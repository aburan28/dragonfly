# ECC2K-130

A GPU-oriented client for the Certicom ECC2K-130 challenge: Pollard rho with the
Frobenius-based iteration function of Bailey et al., bitsliced over the permuted
type-II optimal normal basis of `GF(2^131)`.

The whole arithmetic layer is produced by a code generator that verifies every
routine it emits against an independent model of the field. The same source
compiles for CUDA (32-bit lanes) and for the CPU (64-bit lanes), so the exact
code that would run on a GPU is what the test suite exercises.

## Status

| | |
|---|---|
| Field arithmetic, iteration function, solver | implemented and tested |
| End-to-end discrete logarithms | recovered on `GF(2^23)` and `GF(2^41)` |
| CPU client | measured, 12.6 M iterations/s per core |
| CUDA client | host and device both compile; **never run on a GPU here** |
| Modal integration | validate, benchmark, autotune, search, fan out |

No GPU was available while this was written, so the CUDA path is verified by
compiling it — host and device halves, plus `ptxas` register allocation — and
by running the identical arithmetic on the CPU. `modal_app.py` exists to close
that gap: it builds the client on a Modal GPU, runs the same validation there,
recovers discrete logarithms on the device, and autotunes the build knobs
against real hardware.

## The problem

Solve `Q = [k]P` on the Koblitz curve `y^2 + xy = x^3 + 1` over `F_{2^131}`,
where `|E| = 4l` and

```
l = 680564733841876926932320129493409985129        (129-bit prime)
```

Expected work is `2^60.9` iterations, producing about `2^35.6` distinguished
points.

## Design

### Word width

One word operation advances `W` walks at once. The device uses 32-bit words,
and the host picks the widest it has: AVX-512 gives 512 lanes and, through
`vpternlogd`, the same single-instruction three-input logic that `LOP3.LUT`
gives on the GPU. The arithmetic source is identical for all three widths;
only `bitslice.h` changes.

### Field representation

An element is `m` words in the permuted type-II optimal normal basis
`gamma_i = zeta^i + zeta^-i`, `i = 1..m`, where `zeta` is a primitive 263rd root
of unity. Two properties drive the whole design:

* squaring is the index permutation `i -> fold(2i)`, so it costs nothing when
  loops are unrolled, and
* the Hamming weight is invariant under squaring and under negation, so the
  iteration function is well defined on orbits of size `2m` without ever
  computing a canonical representative inside the loop.

The generator models this field as the symmetric vectors over `Z/263` modulo the
all-ones vector, with multiplication as cyclic convolution. That is a complete
and obviously correct model of `GF(2^131)`, and it is what every generated
routine is checked against.

Multiplication follows Bernstein-Lange: convert both operands to the optimal
polynomial basis `{1, c, c^2, ...}` with `c = gamma_1`, multiply as polynomials,
convert the 261-coefficient product back. Both conversions are generated from
the substitution `c = z + 1/z`, which splits by parity as
`H(c) = Heven(c^2) + c * Hodd(c^2)`; since squaring is an index permutation this
gives an `O(K log K)` conversion instead of the `O(K*m)` of a dense matrix.

### Iteration function

```
R_{i+1} = sigma^j(R_i) + R_i,    j = 3 + ((HW(x_{R_i}) / 2) mod 8)
```

with a point distinguished when `HW(x) <= 34`. The `sigma^j` selection is
branch-free: three conditional squarings, each one `LOP3` per word, and `j` is
never materialised.

### Batching and memory traffic

Each thread owns `ECC_BATCH` slots; a slot is 32 walks in the bit lanes of one
word. An iteration is two passes: the first computes weights, the
distinguished-point test and the running product of the addition denominators;
the second does one field inversion for the whole batch and then the affine
additions.

Pass 2 recomputes `sigma^j(x)` and `sigma^j(y)` from three stored weight bits
instead of reading back stored denominators. That trades 786 instructions for
1048 bytes of traffic per slot, which is worth it here — see the memory
analysis below.

### Distinguished points and restarts

A walk that reports is restarted in place from `R = Q + sum c_i sigma^i(P)`,
with `c` a 128-bit string from a PRF of the walk seed, computed with the same
bitsliced arithmetic and merged into the finished lanes under a mask. No linear
combination of `P` and `Q` is tracked in the loop; the server recomputes both
walks from their seeds when two of them collide, which is what keeps the inner
loop free of conditional counter updates.

Reports are `(seed, endpoint)`. Collision resolution recomputes each walk
counting how often each `sigma^j + 1` was applied, giving
`endpoint = [mu](alpha_0 P + Q)` with `mu = prod_j (1 + s^j)^{n_j}`, matches the
two endpoints up to Frobenius and negation, and solves for `k`.

## Results

### Validation

`make test` runs 52 checks across `GF(2^23)`, `GF(2^41)`, `GF(2^83)` and
`GF(2^131)`; all pass.

* every generated routine is checked against the independent field model inside
  the generator, on 64 random inputs at a time, before it is written out;
* the bitsliced multiply, square, inverse, `sigma^7` and Hamming weight are
  differential-tested against the reference implementation, which shares no code
  with them;
* the iteration function commutes with Frobenius and with negation, weight is
  constant on an orbit, and subgroup x-coordinates have even weight;
* the bitsliced start point matches the reference, and the tracked scalar
  reproduces it;
* the collision solver recovers a planted discrete logarithm;
* the distinguished-point comparison is checked exhaustively over all weights
  and cutoffs;
* the challenge point converts out of the normal basis back to Certicom's
  published polynomial-basis hex, `05 1C99BFA6 F18DE467 C80C23B9 8C7994AA`.

Longer runs check the reporting path rather than the algebra. On `GF(2^83)`,
12.6 M iterations produced 172 distinguished points at a rate within 30% of the
predicted one, and 60 of them were recomputed from their seeds by the reference
implementation with no mismatch, including walks that had been restarted. On
the challenge curve itself, run with a deliberately loose cutoff of 34 -> 50 so
that points actually appear, 3.1 M iterations produced 12368 reports, 25 of
which were recomputed and matched exactly.

### End-to-end

`make break-small` recovers planted discrete logarithms through the complete
pipeline — bitsliced walk, distinguished points, store, collision, recomputation
with exponent tracking, linear algebra mod `l`, and verification that
`[k]P = Q`. On `GF(2^41)` (`l` is 39 bits, about `2^16.6` expected iterations)
each instance takes well under a second on four cores.

### Throughput

Measured on one core of a 2.1 GHz Xeon, `GF(2^131)`, median of five runs:

| word | lanes | batch | iterations/s | cycles/iteration |
|---|---|---|---|---|
| `uint64_t` | 64 | 16 | 4.42 M | 475 |
| AVX2 | 256 | 32 | 10.7 M | 196 |
| AVX-512 | 512 | 32 | 12.6 M | 167 |

Four threads reach 40 M iterations/s. For comparison, the 2009 hand-written
qhasm implementation reached 533 cycles/iteration on a Core 2 with 128-bit
vectors.

The batch size matters more than cache pressure would suggest: going from 4 to
32 walks per inversion is worth 60% because it drives the amortised inversion
from 25% of the multiplication budget down to 3%.

| Karatsuba leaf | 9 | 17 (default) | 33 |
|---|---|---|---|
| instructions per 131-bit multiply | 8859 | 8859 | 10701 |

### Device resources (offline, `ptxas` 12.9, no GPU)

For `sm_120` at `ECC_BATCH=16`, leaf 17:

| | walk kernel | reseed kernel |
|---|---|---|
| registers | 255 | 255 |
| stack frame | 16440 B | 15400 B |
| spill stores / loads | 5160 B / 7024 B | 1556 B / 1564 B |

`mulLeaf`, `multPrep`, `toOnb`, `hamming` and `sigmaJ` each compile with zero
spills. The remaining spills are in the Karatsuba glue and the kernel body,
which hold several 131-word intermediates at once. The per-thread stack frame
is the number to watch on real hardware: it is local memory, so occupancy
depends on how much of it stays in L1.

## Findings that update the implementation guide

**LOP3 fusion is worth 1.65x, and it changes the Karatsuba tradeoff.** A
131-bit multiplication costs 14554 two-input bit operations but only 8859
instructions once `xor(xor(a,b),c)` and `xor(a, and(b,c))` are fused into single
three-input operations. Because the fused form charges nothing for the XOR that
accompanies an AND, *schoolbook leaves beat Karatsuba leaves*: recursing all the
way down costs 18819 bit operations and 14770 instructions, while stopping at a
schoolbook leaf costs more bit operations and fewer instructions. The guide's
§5.1 question — whether to re-evaluate Karatsuba depth — resolves toward
shallower recursion than the bit-operation count suggests.

**The conversions are not a problem.** Generated from the parity recursion and
fused, `multPrep` is 331 instructions and the double-size inverse conversion is
903, against 325 and 909 for the hand-designed chains in the papers. The
Hamming weight tree is 268 instructions against 654 bit operations, because a
full adder is two `LOP3`s.

**Code size, not arithmetic, is the wall — and it is avoidable.** Compiling the
batch with everything inlined produced 285,096 PTX instructions for one kernel;
`ptxas` had not finished register allocation after five minutes and 5 GB of
resident memory. Marking the multiplication, inversion, conversions, leaf,
weight tree and `sigma^j` as real subroutines and refusing to unroll the batch
loop brings this to 36,553 PTX lines, a 3.5 s front-end compile and a 2.4 s
`ptxas` run. This is the §9 toolchain risk, and it is entirely a consequence of
inlining policy: no reverse-engineered assembler is needed.

**Keep walk restarts out of the hot kernel.** Building a fresh start point
costs 128 point additions, and inlining that call chain into the walk kernel
cost 5.8 KB of per-thread stack frame (22264 down to 16440 bytes) for a path
taken roughly once in 2^25 iterations per lane. Reporting a point now only
marks its lane, and a separate kernel — launched between walk kernels, and
skipped entirely when nothing reported — revives marked lanes. On the CPU the
same change was worth about 15% of throughput.

**Every generated routine register-allocates cleanly.** For `sm_120`, `mulLeaf`,
`multPrep`, `toOnb`, `hamming` and `sigmaJ` all compile to zero spill stores and
zero spill loads. What spills is the Karatsuba glue inside `mul` and the kernel
body, which hold several 131-word intermediates at once.

**On current hardware this loop is close to memory bound, which it was not in
2009.** With denominators recomputed rather than stored, one walk-step moves
about 98 bytes (state in and out, plus the Montgomery product chain) and costs
about 1700 word instructions. A 5090-class part is balanced at roughly 29
instructions per byte and this design sits at about 18, so memory is the tighter
of the two constraints by about 1.6x — where the 2009 GTX 295 implementation
spent only 11.9% of its cycles on DRAM. Compute grew about 100x since then and
bandwidth about 8x. Anything that reduces bytes per walk-step is now worth more
than anything that reduces bit operations, which is the opposite of the tradeoff
the original design faced.

## Running on Modal

`modal_app.py` builds the client into a CUDA image (a fat binary covering
Ampere through Blackwell) and exposes five entry points:

```
modal run modal_app.py::validate --gpu H100        # correctness, on the device
modal run modal_app.py::bench --gpu B200           # throughput
modal run modal_app.py::autotune --gpu RTX-PRO-6000
modal run modal_app.py::search --gpu H100 --hours 4
modal run modal_app.py::fanout --gpu H100 --count 8 --hours 4
modal run modal_app.py::merge                      # collisions across all runs
```

`validate` runs the whole test suite on the GPU machine and then recovers
planted discrete logarithms with the CUDA engine itself, so a GPU run proves the
same thing the CPU run does. `autotune` rebuilds for the local compute
capability only and sweeps batch size, block size and Karatsuba leaf, writing
the ranking to a Modal Volume. `search` collects distinguished points into that
same volume, so runs are resumable and several containers contribute to one
corpus; each gets its own run id, which keeps their seed spaces disjoint.
`merge` scans the corpus for repeated hashes, which are the candidate
collisions.

`RTX-PRO-6000` is the interesting target: it is the GB202 part the guide
identifies as the best value for this workload, since none of a datacenter
GPU's tensor silicon is reachable from binary-field arithmetic.

## Build

```
make generate     # regenerate the arithmetic headers (needs python3 only)
make cpu          # CPU client
make test         # validation suite
make break-small  # end-to-end discrete logarithms on the small curves
make bench        # throughput on the challenge curve
make gpu          # CUDA client (needs nvcc)
make check-cuda   # type-check host and device with clang, no GPU or nvcc needed
make ptx          # device compile + ptxas report, no GPU needed
```

Build-time knobs: `BATCH` (walks batched per inversion, default 32), `THREADS`
(CUDA block size), `MARCH` (host architecture; `native` picks the widest word),
and `--leaf` to the generator (Karatsuba leaf size).

```
./ecc2k130-cpu --curve 131 --bench
./ecc2k130-cpu --curve 41 --instance 3      # recover a planted discrete log
./ecc2k130 --curve 131 --dp-file dps.txt    # collect distinguished points
```

## What is not done

* No GPU run from here. Throughput on real hardware is unmeasured; the Modal
  app is the way to get it. The §6 layout question — one thread per bitsliced
  multiply versus 32 threads cooperating — is only partly answered: the register
  data says the leaf fits comfortably, but the 16.4 KB per-thread stack frame
  means occupancy needs measurement.
* The multiplier is optimal only within the Karatsuba family. Toom-3 over
  GF(2), which is where Bernstein's 11961-bit-operation chain comes from, is not
  implemented; a search over balanced and unbalanced Karatsuba splits and the
  guide's 128+3 decomposition found nothing better than 8859 instructions.
* No server. Distinguished points can be written to a file and reloaded, but
  there is no UDP protocol, no hash-routed sharding, and no multi-machine
  merging.
* The start-point PRF is a mixing function rather than AES. Any client that
  wants to interoperate with a different implementation must agree on it.
* Checkpointing and multi-GPU are not implemented.
