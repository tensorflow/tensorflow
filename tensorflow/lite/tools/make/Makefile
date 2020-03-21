# Make uses /bin/sh by default, which is incompatible with the bashisms seen
# below.
SHELL := /bin/bash

# Find where we're running from, so we can store generated files here.
ifeq ($(origin MAKEFILE_DIR), undefined)
	MAKEFILE_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
endif

# Try to figure out the host system
HOST_OS :=
ifeq ($(OS),Windows_NT)
	HOST_OS = windows
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		HOST_OS := linux
	endif
	ifeq ($(UNAME_S),Darwin)
		HOST_OS := osx
	endif
endif

HOST_ARCH := $(shell if uname -m | grep -q i[345678]86; then echo x86_32; else uname -m; fi)

# Override these on the make command line to target a specific architecture. For example:
# make -f tensorflow/lite/tools/make/Makefile TARGET=rpi TARGET_ARCH=armv7l
TARGET := $(HOST_OS)
TARGET_ARCH := $(HOST_ARCH)

INCLUDES := \
-I. \
-I$(MAKEFILE_DIR)/../../../../../ \
-I$(MAKEFILE_DIR)/../../../../../../ \
-I$(MAKEFILE_DIR)/downloads/ \
-I$(MAKEFILE_DIR)/downloads/eigen \
-I$(MAKEFILE_DIR)/downloads/absl \
-I$(MAKEFILE_DIR)/downloads/gemmlowp \
-I$(MAKEFILE_DIR)/downloads/neon_2_sse \
-I$(MAKEFILE_DIR)/downloads/farmhash/src \
-I$(MAKEFILE_DIR)/downloads/flatbuffers/include \
-I$(MAKEFILE_DIR)/downloads/fp16/include \
-I$(OBJDIR)
# This is at the end so any globally-installed frameworks like protobuf don't
# override local versions in the source tree.
INCLUDES += -I/usr/local/include

# These are the default libraries needed, but they can be added to or
# overridden by the platform-specific settings in target makefiles.
LIBS := \
-lstdc++ \
-lpthread \
-lm \
-lz

# There are no rules for compiling objects for the host system (since we don't
# generate things like the protobuf compiler that require that), so all of
# these settings are for the target compiler.
CFLAGS := -O3 -DNDEBUG -fPIC
CXXFLAGS := $(CFLAGS) --std=c++11 $(EXTRA_CXXFLAGS)
LDOPTS := -L/usr/local/lib
ARFLAGS := -r
TARGET_TOOLCHAIN_PREFIX :=
CC_PREFIX :=

ifeq ($(HOST_OS),windows)
CXXFLAGS += -fext-numeric-literals -D__LITTLE_ENDIAN__
endif

# Auto-detect optimization opportunity if building natively.
ifeq ($(HOST_OS),$(TARGET))
ifeq ($(HOST_ARCH),$(TARGET_ARCH))
ifeq ($(TARGET_ARCH),armv7l)
ifneq ($(shell cat /proc/cpuinfo | grep Features | grep neon),)
  ifneq ($(shell cat /proc/cpuinfo | grep Features | grep vfpv4),)
    CXXFLAGS += -mfpu=neon-vfpv4
  else
    CXXFLAGS += -mfpu=neon
  endif
endif # ifeq ($(TARGET_ARCH),armv7l)
endif # ifeq ($(HOST_ARCH),$(TARGET_ARCH))
endif # ifeq ($(HOST_OS),$(TARGET))
endif

# This library is the main target for this makefile. It will contain a minimal
# runtime that can be linked in to other programs.
LIB_NAME := libtensorflow-lite.a

# Benchmark static library and binary
BENCHMARK_LIB_NAME := benchmark-lib.a
BENCHMARK_BINARY_NAME := benchmark_model
BENCHMARK_PERF_OPTIONS_BINARY_NAME := benchmark_model_performance_options

# A small example program that shows how to link against the library.
MINIMAL_SRCS := \
	tensorflow/lite/examples/minimal/minimal.cc

# What sources we want to compile, must be kept in sync with the main Bazel
# build files.

PROFILER_SRCS := \
	tensorflow/lite/profiling/memory_info.cc \
	tensorflow/lite/profiling/platform_profiler.cc \
	tensorflow/lite/profiling/time.cc

PROFILE_SUMMARIZER_SRCS := \
	tensorflow/lite/profiling/profile_summarizer.cc \
	tensorflow/lite/profiling/profile_summary_formatter.cc \
	tensorflow/core/util/stats_calculator.cc

CMD_LINE_TOOLS_SRCS := \
	tensorflow/lite/tools/command_line_flags.cc

CORE_CC_ALL_SRCS := \
$(wildcard tensorflow/lite/*.cc) \
$(wildcard tensorflow/lite/*.c) \
$(wildcard tensorflow/lite/c/*.c) \
$(wildcard tensorflow/lite/core/*.cc) \
$(wildcard tensorflow/lite/core/api/*.cc) \
$(wildcard tensorflow/lite/experimental/resource/*.cc) \
$(wildcard tensorflow/lite/experimental/ruy/*.cc)
ifneq ($(BUILD_TYPE),micro)
CORE_CC_ALL_SRCS += \
$(wildcard tensorflow/lite/kernels/*.cc) \
$(wildcard tensorflow/lite/kernels/internal/*.cc) \
$(wildcard tensorflow/lite/kernels/internal/optimized/*.cc) \
$(wildcard tensorflow/lite/kernels/internal/reference/*.cc) \
$(wildcard tensorflow/lite/tools/optimize/sparsity/*.cc) \
$(PROFILER_SRCS) \
tensorflow/lite/tools/make/downloads/farmhash/src/farmhash.cc \
tensorflow/lite/tools/make/downloads/fft2d/fftsg.c \
tensorflow/lite/tools/make/downloads/fft2d/fftsg2d.c \
tensorflow/lite/tools/make/downloads/flatbuffers/src/util.cpp
CORE_CC_ALL_SRCS += \
	$(shell find tensorflow/lite/tools/make/downloads/absl/absl/ \
	             -type f -name \*.cc | grep -v test | grep -v benchmark | grep -v synchronization | grep -v debugging | grep -v hash | grep -v flags | grep -v random)
endif
# Remove any duplicates.
CORE_CC_ALL_SRCS := $(sort $(CORE_CC_ALL_SRCS))
CORE_CC_EXCLUDE_SRCS := \
$(wildcard tensorflow/lite/*test.cc) \
$(wildcard tensorflow/lite/*/*test.cc) \
$(wildcard tensorflow/lite/*/*/benchmark.cc) \
$(wildcard tensorflow/lite/*/*/example*.cc) \
$(wildcard tensorflow/lite/*/*/test*.cc) \
$(wildcard tensorflow/lite/*/*/*test.cc) \
$(wildcard tensorflow/lite/*/*/*tool.cc) \
$(wildcard tensorflow/lite/*/*/*/*test.cc) \
$(wildcard tensorflow/lite/kernels/*test_main.cc) \
$(wildcard tensorflow/lite/kernels/*test_util*.cc) \
tensorflow/lite/experimental/ruy/tune_tool.cc \
$(MINIMAL_SRCS)

BUILD_WITH_MMAP ?= true
ifeq ($(BUILD_TYPE),micro)
	BUILD_WITH_MMAP=false
endif
ifeq ($(BUILD_TYPE),windows)
	BUILD_WITH_MMAP=false
endif
ifeq ($(BUILD_WITH_MMAP),true)
	CORE_CC_EXCLUDE_SRCS += tensorflow/lite/mmap_allocation.cc
else
	CORE_CC_EXCLUDE_SRCS += tensorflow/lite/mmap_allocation_disabled.cc
endif

BUILD_WITH_RUY ?= false
ifeq ($(TARGET_ARCH),aarch64)
	BUILD_WITH_RUY=true
endif
ifeq ($(BUILD_WITH_RUY),true)
  CXXFLAGS += -DTFLITE_WITH_RUY
endif

# Not to include XNNPACK.
CXXFLAGS += -DTFLITE_WITHOUT_XNNPACK

BUILD_WITH_NNAPI ?= false
ifeq ($(BUILD_WITH_NNAPI),true)
	CORE_CC_ALL_SRCS += tensorflow/lite/delegates/nnapi/nnapi_delegate.cc
	CORE_CC_ALL_SRCS += tensorflow/lite/delegates/nnapi/quant_lstm_sup.cc
	CORE_CC_ALL_SRCS += tensorflow/lite/nnapi/nnapi_implementation.cc
	CORE_CC_ALL_SRCS += tensorflow/lite/nnapi/nnapi_util.cc
	LIBS += -lrt
else
	CORE_CC_ALL_SRCS += tensorflow/lite/delegates/nnapi/nnapi_delegate_disabled.cc
	CORE_CC_ALL_SRCS += tensorflow/lite/nnapi/nnapi_implementation_disabled.cc
endif

ifeq ($(TARGET),ios)
	CORE_CC_EXCLUDE_SRCS += tensorflow/lite/minimal_logging_android.cc
	CORE_CC_EXCLUDE_SRCS += tensorflow/lite/minimal_logging_default.cc
else
	CORE_CC_EXCLUDE_SRCS += tensorflow/lite/minimal_logging_android.cc
	CORE_CC_EXCLUDE_SRCS += tensorflow/lite/minimal_logging_ios.cc
endif

# Filter out all the excluded files.
TF_LITE_CC_SRCS := $(filter-out $(CORE_CC_EXCLUDE_SRCS), $(CORE_CC_ALL_SRCS))

# Benchmark sources
BENCHMARK_SRCS_DIR := tensorflow/lite/tools/benchmark
EVALUATION_UTILS_SRCS := \
  tensorflow/lite/tools/evaluation/utils.cc
BENCHMARK_ALL_SRCS := \
	$(wildcard $(BENCHMARK_SRCS_DIR)/*.cc) \
	$(PROFILE_SUMMARIZER_SRCS) \
	$(CMD_LINE_TOOLS_SRCS) \
	$(EVALUATION_UTILS_SRCS)

BENCHMARK_MAIN_SRC := $(BENCHMARK_SRCS_DIR)/benchmark_main.cc
BENCHMARK_PERF_OPTIONS_SRC := \
	$(BENCHMARK_SRCS_DIR)/benchmark_tflite_performance_options_main.cc
BENCHMARK_LIB_SRCS := $(filter-out \
	$(wildcard $(BENCHMARK_SRCS_DIR)/*_test.cc) \
	$(BENCHMARK_MAIN_SRC) \
	$(BENCHMARK_PERF_OPTIONS_SRC) \
	$(BENCHMARK_SRCS_DIR)/benchmark_plus_flex_main.cc \
	$(BENCHMARK_SRCS_DIR)/gpu_delegate_provider.cc \
	$(BENCHMARK_SRCS_DIR)/hexagon_delegate_provider.cc \
	$(BENCHMARK_SRCS_DIR)/nnapi_delegate_provider.cc \
	$(BENCHMARK_SRCS_DIR)/xnnpack_delegate_provider.cc, \
	$(BENCHMARK_ALL_SRCS))

# These target-specific makefiles should modify or replace options like
# CXXFLAGS or LIBS to work for a specific targetted architecture. All logic
# based on platforms or architectures should happen within these files, to
# keep this main makefile focused on the sources and dependencies.
include $(wildcard $(MAKEFILE_DIR)/targets/*_makefile.inc)

ALL_SRCS := \
	$(MINIMAL_SRCS) \
	$(PROFILER_SRCS) \
	$(PROFILER_SUMMARIZER_SRCS) \
	$(TF_LITE_CC_SRCS) \
	$(BENCHMARK_LIB_SRCS) \
	$(CMD_LINE_TOOLS_SRCS)

# Where compiled objects are stored.
TARGET_OUT_DIR ?= $(TARGET)_$(TARGET_ARCH)
GENDIR := $(MAKEFILE_DIR)/gen/$(TARGET_OUT_DIR)/
OBJDIR := $(GENDIR)obj/
BINDIR := $(GENDIR)bin/
LIBDIR := $(GENDIR)lib/

LIB_PATH := $(LIBDIR)$(LIB_NAME)
BENCHMARK_LIB := $(LIBDIR)$(BENCHMARK_LIB_NAME)
BENCHMARK_BINARY := $(BINDIR)$(BENCHMARK_BINARY_NAME)
BENCHMARK_PERF_OPTIONS_BINARY := $(BINDIR)$(BENCHMARK_PERF_OPTIONS_BINARY_NAME)
MINIMAL_BINARY := $(BINDIR)minimal

CXX := $(CC_PREFIX)${TARGET_TOOLCHAIN_PREFIX}g++
CC := $(CC_PREFIX)${TARGET_TOOLCHAIN_PREFIX}gcc
AR := $(CC_PREFIX)${TARGET_TOOLCHAIN_PREFIX}ar

MINIMAL_OBJS := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(MINIMAL_SRCS))))

LIB_OBJS := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(patsubst %.cpp,%.o,$(TF_LITE_CC_SRCS)))))

BENCHMARK_MAIN_OBJ := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(BENCHMARK_MAIN_SRC))))

BENCHMARK_PERF_OPTIONS_OBJ := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(BENCHMARK_PERF_OPTIONS_SRC))))

BENCHMARK_LIB_OBJS := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(BENCHMARK_LIB_SRCS))))

# For normal manually-created TensorFlow Lite C++ source files.
$(OBJDIR)%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJDIR)%.o: %.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# For normal manually-created TensorFlow Lite C source files.
$(OBJDIR)%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@
$(OBJDIR)%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# The target that's compiled if there's no command-line arguments.
all: $(LIB_PATH)  $(MINIMAL_BINARY) $(BENCHMARK_BINARY) $(BENCHMARK_PERF_OPTIONS_BINARY)

# The target that's compiled for micro-controllers
micro: $(LIB_PATH)

# Hack for generating schema file bypassing flatbuffer parsing
tensorflow/lite/schema/schema_generated.h:
	@cp -u tensorflow/lite/schema/schema_generated.h.OPENSOURCE tensorflow/lite/schema/schema_generated.h

# Gathers together all the objects we've compiled into a single '.a' archive.
$(LIB_PATH): tensorflow/lite/schema/schema_generated.h $(LIB_OBJS)
	@mkdir -p $(dir $@)
	$(AR) $(ARFLAGS) $(LIB_PATH) $(LIB_OBJS)

lib: $(LIB_PATH)

$(MINIMAL_BINARY): $(MINIMAL_OBJS) $(LIB_PATH)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $(MINIMAL_BINARY) $(MINIMAL_OBJS) \
	$(LIBFLAGS) $(LIB_PATH) $(LDFLAGS) $(LIBS)

minimal: $(MINIMAL_BINARY)

$(BENCHMARK_LIB) : $(LIB_PATH) $(BENCHMARK_LIB_OBJS)
	@mkdir -p $(dir $@)
	$(AR) $(ARFLAGS) $(BENCHMARK_LIB) $(LIB_OBJS) $(BENCHMARK_LIB_OBJS)

benchmark_lib: $(BENCHMARK_LIB)

$(BENCHMARK_BINARY) : $(BENCHMARK_MAIN_OBJ) $(BENCHMARK_LIB)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $(BENCHMARK_BINARY) $(BENCHMARK_MAIN_OBJ) \
	$(LIBFLAGS) -Wl,--whole-archive $(BENCHMARK_LIB) -Wl,--no-whole-archive $(LDFLAGS) $(LIBS)

$(BENCHMARK_PERF_OPTIONS_BINARY) : $(BENCHMARK_PERF_OPTIONS_OBJ) $(BENCHMARK_LIB)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $(BENCHMARK_PERF_OPTIONS_BINARY) $(BENCHMARK_PERF_OPTIONS_OBJ) \
	$(LIBFLAGS) $(BENCHMARK_LIB) $(LDFLAGS) $(LIBS)

benchmark: $(BENCHMARK_BINARY) $(BENCHMARK_PERF_OPTIONS_BINARY)

libdir:
	@echo $(LIBDIR)

# Gets rid of all generated files.
clean:
	rm -rf $(MAKEFILE_DIR)/gen

# Gets rid of target files only, leaving the host alone. Also leaves the lib
# directory untouched deliberately, so we can persist multiple architectures
# across builds for iOS and Android.
cleantarget:
	rm -rf $(OBJDIR)
	rm -rf $(BINDIR)

$(DEPDIR)/%.d: ;
.PRECIOUS: $(DEPDIR)/%.d

-include $(patsubst %,$(DEPDIR)/%.d,$(basename $(ALL_SRCS)))
