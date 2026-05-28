# Makefile for gpu-ga-kernel
# Target: CUDA 11.5 with nvcc

CUDA_PATH ?= /usr/local/cuda
NVCC      = $(CUDA_PATH)/bin/nvcc

# Architecture flags - support common GPUs (adjust as needed)
# sm_70: Volta, sm_75: Turing, sm_80: Ampere, sm_86: Ampere RTX
ARCH_FLAGS = -gencode arch=compute_80,code=sm_80

# Compiler flags
CXXFLAGS   = -std=c++14 -O3 -Xcompiler -Wall,-Wextra
NVCCFLAGS  = $(CXXFLAGS) $(ARCH_FLAGS)

# For CUDA 11.5 compatibility
NVCCFLAGS += --expt-relaxed-constexpr

# Debug build support
ifeq ($(DEBUG),1)
  NVCCFLAGS += -g -G -O0
endif

# Source files
CU_SRCS    = ga_kernel.cu test_bench.cu
CU_OBJS    = $(CU_SRCS:.cu=.o)

# Targets
TEST_BIN   = test_bench
LIB_OBJ    = ga_kernel.o

.PHONY: all clean test

all: $(TEST_BIN)

# Pattern rule for .cu -> .o
%.o: %.cu ga_types.cuh ga_kernel.cuh
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

# Link the test bench
$(TEST_BIN): $(CU_OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

# Static library target
libga_kernel.a: $(LIB_OBJ)
	nv-ar rcs $@ $^

test: $(TEST_BIN)
	./$(TEST_BIN)

clean:
	rm -f $(CU_OBJS) $(TEST_BIN) libga_kernel.a
