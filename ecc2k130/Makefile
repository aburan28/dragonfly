# ECC2K-130 client.
#
#   make generate     regenerate the arithmetic headers from codegen/
#   make cpu          CPU build (64-bit lanes, OpenMP)
#   make test         run the validation suite
#   make gpu          CUDA build (needs nvcc)
#   make ptx          device-only compile + ptxas report (needs clang or nvcc,
#                     but no GPU) -- register, spill and code-size numbers
#
# Build-time knobs:
#   BATCH    walks batched into one field inversion per thread (default 16)
#   THREADS  CUDA block size (default 128)
#   ARCH     CUDA architectures

NVCC    ?= nvcc
CXX     ?= g++
CLANG   ?= clang
PTXAS   ?= ptxas
CUDAPATH ?= /usr/local/cuda
BATCH   ?= 16
THREADS ?= 128
ARCH    ?= -gencode arch=compute_80,code=sm_80 \
           -gencode arch=compute_86,code=sm_86 \
           -gencode arch=compute_89,code=sm_89 \
           -gencode arch=compute_90,code=sm_90 \
           -gencode arch=compute_120,code=sm_120
SMARCH  ?= sm_90

HDRS = include/bitslice.h include/fieldbs.h include/walk.h include/kernel.h \
       include/ref.h include/solver.h include/bigmod.h include/curveparams.h
GEN  = generated/eccF131.h generated/eccF83.h generated/eccF41.h generated/eccF23.h

DEFS = -DECC_BATCH=$(BATCH) -DECC_THREADS=$(THREADS)

all: cpu

generate:
	cd codegen && python3 gen.py --out ../generated

cpu: ecc2k130-cpu

ecc2k130-cpu: src/main.cu $(HDRS) $(GEN)
	$(CXX) -O3 -std=c++17 -march=native -fopenmp $(DEFS) -DECC_NO_CUDA \
	    -Wall -Wno-unused-function -x c++ src/main.cu -o $@

gpu: ecc2k130

ecc2k130: src/main.cu $(HDRS) $(GEN)
	$(NVCC) -O3 -std=c++17 $(ARCH) $(DEFS) -lineinfo -Xcompiler -O3 \
	    -Xcompiler -fopenmp src/main.cu -o $@ -lgomp

test: ecc2k130-cpu
	./ecc2k130-cpu --test

# End-to-end: recover planted discrete logarithms on the small curves.
break-small: ecc2k130-cpu
	@for i in 0 1 2 3 4 5 6 7; do \
	    ./ecc2k130-cpu --curve 23 --instance $$i --threads 1 --steps 32 --verify 2 | grep -E "planted|MISMATCH"; \
	done
	@for i in 0 1 2 3; do \
	    ./ecc2k130-cpu --curve 41 --instance $$i --threads 4 --steps 64 --verify 2 | grep -E "planted|MISMATCH"; \
	done

bench: ecc2k130-cpu
	./ecc2k130-cpu --curve 131 --bench --steps 32 --launches 8 --verify 0

# Offline device compile: no GPU needed, reports registers, spills and size.
ptx: src/ptxspike.cu $(HDRS) $(GEN)
	$(CLANG) -x cuda --cuda-device-only --cuda-gpu-arch=$(SMARCH) --cuda-path=$(CUDAPATH) \
	    -nocudalib -Wno-unknown-cuda-version -O3 -std=c++17 $(DEFS) -S -o build/spike.ptx src/ptxspike.cu
	$(PTXAS) -arch=$(SMARCH) -O3 -v build/spike.ptx -o build/spike.cubin

clean:
	rm -f ecc2k130 ecc2k130-cpu build/spike.ptx build/spike.cubin

.PHONY: all generate cpu gpu test break-small bench ptx clean
