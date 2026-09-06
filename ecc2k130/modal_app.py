"""Run the ECC2K-130 client on Modal GPUs.

    modal run modal_app.py::validate                 # correctness, on the GPU
    modal run modal_app.py::bench --gpu H100         # throughput
    modal run modal_app.py::autotune --gpu B200      # sweep the build knobs
    modal run modal_app.py::search --gpu H100 --hours 4
    modal run modal_app.py::fanout --gpu H100 --count 8 --hours 4

The image builds a fat binary covering Ampere through Blackwell so any GPU type
works without a rebuild.  The autotuner rebuilds for the local architecture
only, which takes seconds rather than minutes.

Distinguished points and results land in a Modal Volume, so a search can be
stopped and resumed and several containers can contribute to one corpus.
"""

import json
import os
import pathlib
import subprocess
import time

import modal

CUDA_VERSION = "12.8.1"
GENCODE = " ".join(
    "-gencode arch=compute_%s,code=sm_%s" % (a, a) for a in ("80", "86", "89", "90", "100", "120")
)
REMOTE = "/root/ecc2k130"
LOCAL = pathlib.Path(__file__).parent

image = (
    modal.Image.from_registry(
        f"nvidia/cuda:{CUDA_VERSION}-devel-ubuntu24.04", add_python="3.12"
    )
    .entrypoint([])
    .apt_install("build-essential")
    .add_local_dir(
        LOCAL,
        remote_path=REMOTE,
        copy=True,
        ignore=["ecc2k130-cpu", "ecc2k130", "build/*", "__pycache__", "*.pyc"],
    )
    .run_commands(
        # x86-64-v3 keeps the host binary runnable on any Modal machine; the
        # GPU client picks its own word width on the device.
        f"cd {REMOTE} && make cpu MARCH=x86-64-v3",
        f'cd {REMOTE} && make gpu ARCH="{GENCODE}" BATCH=32 THREADS=128',
    )
)

volume = modal.Volume.from_name("ecc2k130", create_if_missing=True)
app = modal.App("ecc2k130")

HOUR = 60 * 60


# ---------------------------------------------------------------------------
def sh(cmd, cwd=REMOTE, timeout=None):
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout + r.stderr


def computeCapability():
    rc, out = sh("nvidia-smi --query-gpu=compute_cap --format=csv,noheader")
    if rc != 0 or not out.strip():
        return "90"
    return out.strip().splitlines()[0].strip().replace(".", "")


def gpuName():
    rc, out = sh("nvidia-smi --query-gpu=name --format=csv,noheader")
    return out.strip().splitlines()[0].strip() if rc == 0 and out.strip() else "unknown"


def buildFor(batch, threads, leaf, arch=None):
    """Rebuild the client for one architecture and one set of knobs."""
    arch = arch or computeCapability()
    if leaf:
        rc, out = sh(f"cd codegen && python3 gen.py --out ../generated --leaf {leaf}")
        if rc != 0:
            return False, out
    gencode = f"-gencode arch=compute_{arch},code=sm_{arch}"
    rc, out = sh(
        f'make -B gpu ARCH="{gencode}" BATCH={batch} THREADS={threads}', timeout=1800
    )
    return rc == 0, out


def parseRate(text):
    """Iterations per second, in millions, from the client's own report.  The
    "finished" line carries the average over the whole run, so prefer it."""
    best = 0.0
    for line in text.splitlines():
        if "M it/s" not in line:
            continue
        head = line.split("M it/s")[0].split()
        if not head:
            continue
        try:
            rate = float(head[-1])
        except ValueError:
            continue
        if line.strip().startswith("finished"):
            return rate
        best = max(best, rate)
    return best


# ---------------------------------------------------------------------------
@app.function(image=image, gpu="H100", timeout=2 * HOUR, volumes={"/data": volume})
def runValidate():
    """Field arithmetic, orbit invariants, solver, and end-to-end discrete
    logarithms recovered on the GPU itself."""
    out = ["device: " + gpuName(), "compute capability: " + computeCapability(), ""]
    rc, t = sh("./ecc2k130-cpu --test")
    out.append(t.strip())
    if rc != 0:
        return "\n".join(out) + "\nHOST VALIDATION FAILED"

    out.append("\n--- end-to-end on the GPU ---")
    ok = True
    for curve, instances, threads in (("23", 4, 1024), ("41", 4, 4096)):
        for i in range(instances):
            rc, t = sh(
                f"./ecc2k130 --curve {curve} --instance {i} --threads {threads} "
                f"--steps 32 --verify 4"
            )
            line = [l for l in t.splitlines() if "planted" in l or "MISMATCH" in l]
            good = any("yes" in l for l in line)
            ok = ok and good
            out.append(f"curve {curve} instance {i}: " + ("; ".join(line) if line else t.strip()[-200:]))
    out.append("GPU END TO END: " + ("all instances solved" if ok else "FAILED"))
    return "\n".join(out)


@app.function(image=image, gpu="H100", timeout=1 * HOUR)
def runBench(batch=32, threads=128, leaf=17, steps=64, launches=20, workers=0, rebuild=True):
    """Throughput on the challenge curve."""
    info = {"gpu": gpuName(), "cc": computeCapability(), "batch": batch,
            "threads": threads, "leaf": leaf}
    if rebuild:
        ok, log = buildFor(batch, threads, leaf)
        if not ok:
            info["error"] = log[-2000:]
            return info
    cmd = f"./ecc2k130 --curve 131 --bench --steps {steps} --launches {launches} --verify 0"
    if workers:
        cmd += f" --threads {workers}"
    rc, out = sh(cmd, timeout=1800)
    info["rate"] = parseRate(out)
    info["raw"] = out.strip()[-1200:]
    return info


@app.function(image=image, gpu="H100", timeout=4 * HOUR, volumes={"/data": volume})
def runAutotune(batches="8,16,32,64", threadCounts="64,128,256", leaves="17,33",
                steps=64, launches=12):
    """Sweep the build-time knobs on the real device and report the best."""
    results = []
    arch = computeCapability()
    name = gpuName()
    for leaf in [int(x) for x in leaves.split(",") if x]:
        for threads in [int(x) for x in threadCounts.split(",") if x]:
            for batch in [int(x) for x in batches.split(",") if x]:
                t0 = time.time()
                ok, log = buildFor(batch, threads, leaf, arch)
                if not ok:
                    results.append({"batch": batch, "threads": threads, "leaf": leaf,
                                    "rate": 0.0, "error": log[-400:]})
                    continue
                rc, out = sh(
                    f"./ecc2k130 --curve 131 --bench --steps {steps} "
                    f"--launches {launches} --verify 0",
                    timeout=1800,
                )
                rate = parseRate(out)
                results.append({"batch": batch, "threads": threads, "leaf": leaf,
                                "rate": rate, "buildSeconds": round(time.time() - t0, 1)})
                print(f"leaf {leaf} threads {threads} batch {batch}: {rate:.3f} M it/s")
    results.sort(key=lambda r: -r.get("rate", 0.0))
    report = {"gpu": name, "cc": arch, "results": results, "best": results[0] if results else None}
    os.makedirs("/data/autotune", exist_ok=True)
    path = f"/data/autotune/{name.replace(' ', '_')}.json"
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2)
    volume.commit()
    return report


@app.function(image=image, gpu="H100", timeout=24 * HOUR, volumes={"/data": volume})
def runSearch(hours=1.0, curve=131, batch=32, threads=128, leaf=17, dpWeight=-1,
              runId=1, steps=256, workers=0, rebuild=True):
    """Collect distinguished points into the volume until the time budget runs
    out.  Records are (seed, hash); a collision is resolved by recomputing both
    walks from their seeds."""
    if rebuild:
        ok, log = buildFor(batch, threads, leaf)
        if not ok:
            return {"error": log[-2000:]}
    name = gpuName()
    os.makedirs("/data/dp", exist_ok=True)
    dpFile = f"/data/dp/curve{curve}-run{runId}.txt"
    launchSeconds = 30.0
    rc, out = sh(f"./ecc2k130 --curve {curve} --bench --steps 8 --launches 4 --verify 0")
    rate = parseRate(out) or 1.0
    perLaunch = max(1, int(rate * 1e6 * launchSeconds / (steps * 1.0)))
    cmd = (f"./ecc2k130 --curve {curve} --steps {steps} --run-id {runId} "
           f"--dp-file {dpFile} --verify 4 --launches 0")
    if dpWeight >= 0:
        cmd += f" --dp-weight {dpWeight}"
    if workers:
        cmd += f" --threads {workers}"
    deadline = time.time() + hours * HOUR
    print(f"{name}: collecting into {dpFile} for {hours} h at ~{rate:.2f} M it/s")
    proc = subprocess.Popen(cmd, shell=True, cwd=REMOTE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    lines = []
    solved = None
    try:
        while proc.poll() is None:
            line = proc.stdout.readline()
            if line:
                lines.append(line.rstrip())
                if "k = " in line:
                    solved = line.strip()
                if len(lines) % 20 == 0:
                    volume.commit()
            if time.time() > deadline:
                proc.terminate()
                break
    finally:
        try:
            proc.wait(timeout=30)
        except Exception:
            proc.kill()
    volume.commit()
    count = 0
    if os.path.exists(dpFile):
        with open(dpFile) as fh:
            count = sum(1 for _ in fh)
    return {"gpu": name, "distinguishedPoints": count, "file": dpFile,
            "solved": solved, "tail": lines[-25:]}


@app.function(image=image, timeout=2 * HOUR, volumes={"/data": volume})
def mergeCorpus(curve=131):
    """Merge every distinguished-point file in the volume and report duplicate
    hashes, which are the candidate collisions."""
    import collections
    seen = {}
    dup = []
    total = 0
    root = "/data/dp"
    if not os.path.isdir(root):
        return {"error": "no distinguished points yet"}
    for fn in sorted(os.listdir(root)):
        if not fn.startswith(f"curve{curve}-"):
            continue
        with open(os.path.join(root, fn)) as fh:
            for line in fh:
                parts = line.split()
                if len(parts) != 2:
                    continue
                total += 1
                seed, h = parts
                if h in seen and seen[h] != seed:
                    dup.append((seen[h], seed, h))
                else:
                    seen[h] = seed
    return {"records": total, "distinct": len(seen), "collisions": dup[:50],
            "collisionCount": len(dup)}


# ---------------------------------------------------------------------------
@app.local_entrypoint()
def validate(gpu: str = "H100"):
    print(runValidate.with_options(gpu=gpu).remote())


@app.local_entrypoint()
def bench(gpu: str = "H100", batch: int = 32, threads: int = 128, leaf: int = 17,
          steps: int = 64, launches: int = 20):
    r = runBench.with_options(gpu=gpu).remote(batch=batch, threads=threads, leaf=leaf,
                                              steps=steps, launches=launches)
    print(json.dumps({k: v for k, v in r.items() if k != "raw"}, indent=2))
    if "raw" in r:
        print(r["raw"])


@app.local_entrypoint()
def autotune(gpu: str = "H100", batches: str = "8,16,32,64",
             threadCounts: str = "64,128,256", leaves: str = "17,33"):
    r = runAutotune.with_options(gpu=gpu).remote(batches=batches,
                                                 threadCounts=threadCounts, leaves=leaves)
    print(json.dumps(r, indent=2))


@app.local_entrypoint()
def search(gpu: str = "H100", hours: float = 1.0, curve: int = 131, batch: int = 32,
           threads: int = 128, leaf: int = 17, dpWeight: int = -1, runId: int = 1):
    r = runSearch.with_options(gpu=gpu).remote(hours=hours, curve=curve, batch=batch,
                                               threads=threads, leaf=leaf,
                                               dpWeight=dpWeight, runId=runId)
    print(json.dumps(r, indent=2))


@app.local_entrypoint()
def fanout(gpu: str = "H100", count: int = 4, hours: float = 1.0, curve: int = 131,
           batch: int = 32, threads: int = 128, leaf: int = 17, dpWeight: int = -1):
    """Run `count` independent searchers, each with its own run id so their
    seeds never collide, then merge what they produced."""
    fn = runSearch.with_options(gpu=gpu)
    calls = [fn.spawn(hours=hours, curve=curve, batch=batch, threads=threads, leaf=leaf,
                      dpWeight=dpWeight, runId=i + 1) for i in range(count)]
    for c in calls:
        print(json.dumps(c.get(), indent=2))
    print(json.dumps(mergeCorpus.remote(curve=curve), indent=2))


@app.local_entrypoint()
def merge(curve: int = 131):
    print(json.dumps(mergeCorpus.remote(curve=curve), indent=2))
