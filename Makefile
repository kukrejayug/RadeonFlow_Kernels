.PHONY: test dis
test:
	cmake --build build --target gemm gemm_checker gemm3
	./build/gemm_checker -b

dis:
	roc-obj -d build/libgemm.so

prof:
	cmake --build build --target gemm3
	/opt/rocm/bin/rocprof-compute profile -n gemm -k gemm_kernel -- ./build/gemm3
	/opt/rocm/bin/rocprof-compute analyze -p workloads/gemm/MI300 > workloads/prof.log

gui:
	/opt/rocm/bin/rocprof-compute analyze -p workloads/gemm/MI300 --gui

roc:
	../rocprofiler-sdk-build/bin/rocprofv3 -i input.txt -- ./build/gemm3

roc-help:
	../rocprofiler-sdk-build/bin/rocprofv3 -L

opt:
	cmake --build build --target gemm gemm_checker
	LLVM_DEBUG=amdgpu-promote-alloca /opt/rocm/llvm/bin/opt build/gemm_launcher-hip-amdgcn-amd-amdhsa-gfx942.ll \
		-passes="amdgpu-promote-alloca-to-vector" \
		--amdgpu-promote-alloca-to-vector-limit=512 \
		--disable-promote-alloca-to-vector=0 \
		-print-after-all \
		-S -o opt.ll \
		2> opt.log

trans:
	cmake --build build --target trans
	./build/trans
	roc-obj -d build/trans

eval:
	cmake --build build --target mla
	POPCORN_FD=1 PYTHONPATH=$(realpath build):$(realpath tests/mla/submit):$(realpath tests/mla):$(PYTHONPATH) python tests/mla/eval.py benchmark tests/mla/test.txt

benchmark:
	cmake --build build --target mla
	POPCORN_FD=1 PYTHONPATH=$(realpath build):$(realpath tests/mla/submit):$(realpath tests/mla):$(PYTHONPATH) python tests/mla/eval.py benchmark tests/mla/benchmark.txt


rank:
	cmake --build build --target mla
	POPCORN_FD=1 PYTHONPATH=$(realpath build):$(realpath tests/mla/submit):$(realpath tests/mla):$(PYTHONPATH) python tests/mla/eval.py leaderboard tests/mla/rank.txt


ref:
	POPCORN_FD=1 PYTHONPATH=$(realpath build):$(realpath tests/mla/ref):$(realpath tests/mla):$(PYTHONPATH) python tests/mla/eval.py benchmark tests/mla/test.txt
