# Find where we're running from, so we can store generated files here.
ifeq ($(origin MAKEFILE_DIR), undefined)
	MAKEFILE_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
endif

# Try to figure out the host system
HOST_OS :=
ifeq ($(OS),Windows_NT)
	HOST_OS = WINDOWS
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
	        HOST_OS := LINUX
	endif
	ifeq ($(UNAME_S),Darwin)
		HOST_OS := OSX
	endif
endif

HOST_ARCH := $(shell if [[ $(shell uname -m) =~ i[345678]86 ]]; then echo x86_32; else echo $(shell uname -m); fi)

# Self-hosting
TARGET_ARCH := ${HOST_ARCH}

# Cross compiling
ifeq ($(CROSS),rpi)
  TARGET_ARCH := armv7l
  TARGET_TOOLCHAIN_PREFIX := arm-linux-gnueabihf-
endif

ifeq ($(CROSS),riscv)
  TARGET_ARCH := riscv
  TARGET_TOOLCHAIN_PREFIX := riscv32-unknown-elf-
endif
ifeq ($(CROSS),stm32f7)
  TARGET_ARCH := armf7
  TARGET_TOOLCHAIN_PREFIX := arm-none-eabi-
endif
ifeq ($(CROSS),stm32f1)
  TARGET_ARCH := armm1
  TARGET_TOOLCHAIN_PREFIX := arm-none-eabi-
endif

# Where compiled objects are stored.
OBJDIR := $(MAKEFILE_DIR)/gen/obj/
BINDIR := $(MAKEFILE_DIR)/gen/bin/
LIBDIR := $(MAKEFILE_DIR)/gen/lib/
GENDIR := $(MAKEFILE_DIR)/gen/obj/

LIBS :=
ifeq ($(TARGET_ARCH),x86_64)
        CXXFLAGS += -fPIC -DGEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK -pthread # -msse4.2
endif

ifeq ($(TARGET_ARCH),armv7l)
        CXXFLAGS += -mfpu=neon -pthread -fPIC
	LIBS += -ldl
endif

ifeq ($(TARGET_ARCH),riscv)
#        CXXFLAGS += -march=gap8
        CXXFLAGS += -DTFLITE_MCU
	LIBS += -ldl
	BUILD_TYPE := micro
endif

ifeq ($(TARGET_ARCH),armf7)
        CXXFLAGS += -DGEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK -DTFLITE_MCU
        CXXFLAGS += -fno-rtti -fmessage-length=0 -fno-exceptions -fno-builtin -ffunction-sections -fdata-sections
        CXXFLAGS += -funsigned-char -MMD
        CXXFLAGS += -mcpu=cortex-m7 -mthumb -mfpu=fpv5-sp-d16 -mfloat-abi=softfp
        CXXFLAGS += '-std=gnu++11' '-fno-rtti' '-Wvla' '-c' '-Wall' '-Wextra' '-Wno-unused-parameter' '-Wno-missing-field-initializers' '-fmessage-length=0' '-fno-exceptions' '-fno-builtin' '-ffunction-sections' '-fdata-sections' '-funsigned-char' '-MMD' '-fno-delete-null-pointer-checks' '-fomit-frame-pointer' '-Os'
	LIBS += -ldl
	BUILD_TYPE := micro
endif
ifeq ($(TARGET_ARCH),armm1)
        CXXFLAGS += -DGEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK -mcpu=cortex-m1 -mthumb -DTFLITE_MCU
        CXXFLAGS += -fno-rtti -fmessage-length=0 -fno-exceptions -fno-builtin -ffunction-sections -fdata-sections
        CXXFLAGS += -funsigned-char -MMD
	LIBS += -ldl
endif

# Settings for the host compiler.
CXX := $(CC_PREFIX) ${TARGET_TOOLCHAIN_PREFIX}g++
CXXFLAGS += --std=c++11 -O3 -DNDEBUG
CCFLAGS := ${CXXFLAGS}
CC := $(CC_PREFIX) ${TARGET_TOOLCHAIN_PREFIX}gcc
AR := $(CC_PREFIX) ${TARGET_TOOLCHAIN_PREFIX}ar
CFLAGS :=
LDOPTS :=
LDOPTS += -L/usr/local/lib
ARFLAGS := -r

INCLUDES := \
-I. \
-I$(MAKEFILE_DIR)/../../../ \
-I$(MAKEFILE_DIR)/downloads/ \
-I$(MAKEFILE_DIR)/downloads/eigen \
-I$(MAKEFILE_DIR)/downloads/gemmlowp \
-I$(MAKEFILE_DIR)/downloads/neon_2_sse \
-I$(MAKEFILE_DIR)/downloads/farmhash/src \
-I$(MAKEFILE_DIR)/downloads/flatbuffers/include \
-I$(GENDIR)
# This is at the end so any globally-installed frameworks like protobuf don't
# override local versions in the source tree.
INCLUDES += -I/usr/local/include

LIBS += \
-lstdc++ \
-lpthread \
-lm \
-lz

# If we're on Linux, also link in the dl library.
ifeq ($(HOST_OS),LINUX)
	LIBS += -ldl
endif

include $(MAKEFILE_DIR)/ios_makefile.inc
include $(MAKEFILE_DIR)/rpi_makefile.inc

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
