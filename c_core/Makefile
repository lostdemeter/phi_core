CC = gcc
CFLAGS = -O2 -std=c11 -Wall -Wextra -Werror -I. -fopenmp -mavx512f -mavx512bw -mavx512vl -mavx512dq
# NOTE: AVX512 baseline (measured box); scalar fallback = same sources
# without -mavx512* (simd.h swap documented per lowering contract).

all: check

# header-only + object check (kernels gate through model repos' vectors)
check:
	$(CC) $(CFLAGS) -c -o /tmp/phi_check_bridge.o bridge.c
	$(CC) $(CFLAGS) -c -o /tmp/phi_check_conv.o conv.c
	$(CC) $(CFLAGS) -c -o /tmp/phi_check_fconv.o fconv.c
	$(CC) $(CFLAGS) -c -o /tmp/phi_check_geo.o geo.c
	$(CC) $(CFLAGS) -c -o /tmp/phi_check_arith.o arith.c
	nvcc -O2 -std=c++17 -I. -c -o /tmp/phi_check_convcu.o conv.cu
	nvcc -O2 -std=c++17 -I. -c -o /tmp/phi_check_ops.o ops.cu
	@echo "c_core compiles on all substrates"

test_bridge: test_bridge.c bridge.c
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^)

test_conv: test_conv.c conv.c bridge.c
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^)

test_geo: test_geo.c geo.c bridge.c
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^)

test_arith: test_arith.c arith.c bridge.c
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^)

test: test_bridge test_conv test_geo test_arith
	python3 run_bridge_test.py && python3 run_conv_test.py && python3 run_geo_test.py && python3 run_arith_test.py

trap:
	for f in *.c *.h *.cu *.cuh; do sed -e 's|//.*||' "$$f"; done | grep -nE "math\.h|[^_a-zA-Z](float|double)[ \t*(]|(float|double)\(" && echo FLOAT-FOUND && exit 1 || echo "no float in C sources"

luts:
	python3 gen_luts.py

clean:
	rm -f /tmp/phi_check_*.o test_bridge test_conv test_geo test_arith

.PHONY: all check test trap luts clean
