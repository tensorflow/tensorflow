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

HOST_ARCH := $(shell if [[ $(shell uname -m) =~ i[345678]86 ]]; then echo x86_32; else echo $(shell uname -m); fi)

# Override these on the make command line to target a specific architecture. For example:
# make -f tensorflow/contrib/lite/Makefile CROSS=rpi TARGET_ARCH=armv7l
CROSS := $(HOST_OS)
TARGET_ARCH := $(HOST_ARCH)

# These are the default libraries needed, but they can be added to or
# overridden by the platform-specific settings in target_makefiles.
LIBS := \
-lstdc++ \
-lpthread \
-lm \
-lz

# There are no rules for compiling objects for the host system (since we don't
# generate things like the protobuf compiler that require that), so all of
# these settings are for the target compiler.
CXXFLAGS := -O3 -DNDEBUG
CCFLAGS := ${CXXFLAGS}
CXXFLAGS += --std=c++11
CFLAGS :=
LDOPTS := -L/usr/local/lib
ARFLAGS := -r
TARGET_TOOLCHAIN_PREFIX :=
CC_PREFIX :=

# These target-specific makefiles should modify or replace options like
# CXXFLAGS or LIBS to work for a specific targetted architecture. All logic
# based on platforms or architectures should happen within these files, to
# keep this main makefile focused on the sources and dependencies.
include $(wildcard $(MAKEFILE_DIR)/target_makefiles/*_makefile.inc)

# Where compiled objects are stored.
GENDIR := $(MAKEFILE_DIR)/gen/$(CROSS)_$(TARGET_ARCH)/
OBJDIR := $(GENDIR)obj/
BINDIR := $(GENDIR)bin/
LIBDIR := $(GENDIR)lib/

INCLUDES := \
-I. \
-I$(MAKEFILE_DIR)/../../../ \
-I$(MAKEFILE_DIR)/downloads/ \
-I$(MAKEFILE_DIR)/downloads/eigen \
-I$(MAKEFILE_DIR)/downloads/gemmlowp \
-I$(MAKEFILE_DIR)/downloads/neon_2_sse \
-I$(MAKEFILE_DIR)/downloads/farmhash/src \
-I$(MAKEFILE_DIR)/downloads/flatbuffers/include \
-I$(OBJDIR)
# This is at the end so any globally-installed frameworks like protobuf don't
# override local versions in the source tree.
INCLUDES += -I/usr/local/include

CXX := $(CC_PREFIX) ${TARGET_TOOLCHAIN_PREFIX}g++
CC := $(CC_PREFIX) ${TARGET_TOOLCHAIN_PREFIX}gcc
AR := $(CC_PREFIX) ${TARGET_TOOLCHAIN_PREFIX}ar

# This library is the main target for this makefile. It will contain a minimal
# runtime that can be linked in to other programs.
LIB_NAME := libtensorflow-lite.a
LIB_PATH := $(LIBDIR)$(LIB_NAME)

# A small example program that shows how to link against the library.
MINIMAL_PATH := $(BINDIR)minimal

# Benchmark static library and binary
BENCHMARK_LIB_NAME := benchmark-lib.a
BENCHMARK_BINARY_NAME := benchmark_model
BENCHMARK_LIB := $(LIBDIR)$(BENCHMARK_LIB_NAME)
BENCHMARK_BINARY := $(BINDIR)$(BENCHMARK_BINARY_NAME)

MINIMAL_SRCS := \
tensorflow/contrib/lite/examples/minimal/minimal.cc
MINIMAL_OBJS := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(MINIMAL_SRCS))))

# What sources we want to compile, must be kept in sync with the main Bazel
# build files.

PROFILER_SRCS := \
	tensorflow/contrib/lite/profiling/time.cc
PROFILE_SUMMARIZER_SRCS := \
	tensorflow/contrib/lite/profiling/profile_summarizer.cc \
	tensorflow/core/util/stats_calculator.cc

CORE_CC_ALL_SRCS := \
$(wildcard tensorflow/contrib/lite/*.cc) \
$(wildcard tensorflow/contrib/lite/*.c)
ifneq ($(BUILD_TYPE),micro)
CORE_CC_ALL_SRCS += \
$(wildcard tensorflow/contrib/lite/kernels/*.cc) \
$(wildcard tensorflow/contrib/lite/kernels/internal/*.cc) \
$(wildcard tensorflow/contrib/lite/kernels/internal/optimized/*.cc) \
$(wildcard tensorflow/contrib/lite/kernels/internal/reference/*.cc) \
$(PROFILER_SRCS) \
$(wildcard tensorflow/contrib/lite/kernels/*.c) \
$(wildcard tensorflow/contrib/lite/kernels/internal/*.c) \
$(wildcard tensorflow/contrib/lite/kernels/internal/optimized/*.c) \
$(wildcard tensorflow/contrib/lite/kernels/internal/reference/*.c) \
$(wildcard tensorflow/contrib/lite/downloads/farmhash/src/farmhash.cc) \
$(wildcard tensorflow/contrib/lite/downloads/fft2d/fftsg.c)
endif
# Remove any duplicates.
CORE_CC_ALL_SRCS := $(sort $(CORE_CC_ALL_SRCS))
CORE_CC_EXCLUDE_SRCS := \
$(wildcard tensorflow/contrib/lite/*test.cc) \
$(wildcard tensorflow/contrib/lite/*/*test.cc) \
$(wildcard tensorflow/contrib/lite/*/*/*test.cc) \
$(wildcard tensorflow/contrib/lite/*/*/*/*test.cc) \
$(wildcard tensorflow/contrib/lite/kernels/test_util.cc) \
$(MINIMAL_SRCS)
ifeq ($(BUILD_TYPE),micro)
CORE_CC_EXCLUDE_SRCS += \
tensorflow/contrib/lite/model.cc \
tensorflow/contrib/lite/nnapi_delegate.cc
endif
# Filter out all the excluded files.
TF_LITE_CC_SRCS := $(filter-out $(CORE_CC_EXCLUDE_SRCS), $(CORE_CC_ALL_SRCS))
# File names of the intermediate files target compilation generates.
TF_LITE_CC_OBJS := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(TF_LITE_CC_SRCS))))
LIB_OBJS := $(TF_LITE_CC_OBJS)

# Benchmark sources
BENCHMARK_SRCS_DIR := tensorflow/contrib/lite/tools/benchmark
BENCHMARK_ALL_SRCS := $(TFLITE_CC_SRCS) \
	$(wildcard $(BENCHMARK_SRCS_DIR)/*.cc) \
	$(PROFILE_SUMMARIZER_SRCS)

BENCHMARK_SRCS := $(filter-out \
	$(wildcard $(BENCHMARK_SRCS_DIR)/*_test.cc), \
    $(BENCHMARK_ALL_SRCS))

BENCHMARK_OBJS := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(BENCHMARK_SRCS))))

# For normal manually-created TensorFlow C++ source files.
$(OBJDIR)%.o: %.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
# For normal manually-created TensorFlow C++ source files.
$(OBJDIR)%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CCFLAGS) $(INCLUDES) -c $< -o $@

# The target that's compiled if there's no command-line arguments.
all: $(LIB_PATH)  $(MINIMAL_PATH) $(BENCHMARK_BINARY)

# The target that's compiled for micro-controllers
micro: $(LIB_PATH)

# Gathers together all the objects we've compiled into a single '.a' archive.
$(LIB_PATH): $(LIB_OBJS)
	@mkdir -p $(dir $@)
	$(AR) $(ARFLAGS) $(LIB_PATH) $(LIB_OBJS)

$(MINIMAL_PATH): $(MINIMAL_OBJS) $(LIB_PATH)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $(MINIMAL_PATH) $(MINIMAL_OBJS) \
	$(LIBFLAGS) $(LIB_PATH) $(LDFLAGS) $(LIBS)


$(BENCHMARK_LIB) : $(LIB_PATH) $(BENCHMARK_OBJS)
	@mkdir -p $(dir $@)
	$(AR) $(ARFLAGS) $(BENCHMARK_LIB) $(LIB_OBJS) $(BENCHMARK_OBJS)

benchmark_lib: $(BENCHMARK_LIB)
$(info $(BENCHMARK_BINARY))
$(BENCHMARK_BINARY) : $(BENCHMARK_LIB)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $(BENCHMARK_BINARY) \
	$(LIBFLAGS) $(BENCHMARK_LIB) $(LDFLAGS) $(LIBS)

benchmark: $(BENCHMARK_BINARY)

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

-include $(patsubst %,$(DEPDIR)/%.d,$(basename $(TF_CC_SRCS)))
