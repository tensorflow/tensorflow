
MAKEFILE_DIR := tensorflow/lite/experimental/micro/tools/make

# Pull in some convenience functions.
include $(MAKEFILE_DIR)/helper_functions.inc

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
# make -f tensorflow/lite/Makefile TARGET=rpi TARGET_ARCH=armv7l
TARGET := $(HOST_OS)
TARGET_ARCH := $(HOST_ARCH)

# Specify TAGS on the command line to add a particular set of specialized
# implementations, for example TAGS="CMSIS disco_f746ng" to target a Discovery
# STM32F746NG board, using the CMSIS library's implementations where possible.
ALL_TAGS := $(TAGS) $(TARGET)

INCLUDES := \
-I. \
-I$(MAKEFILE_DIR)/downloads/ \
-I$(MAKEFILE_DIR)/downloads/gemmlowp \
-I$(MAKEFILE_DIR)/downloads/flatbuffers/include

# These are the include paths added to any generated project file.
PROJECT_INCLUDES := \
. \
third_party/gemmlowp \
third_party/flatbuffers/include

TEST_SCRIPT := tensorflow/lite/experimental/micro/testing/test_linux_binary.sh

MICROLITE_LIBS := -lm

# There are no rules for compiling objects for the host system (since we don't
# generate things like the protobuf compiler that require that), so all of
# these settings are for the target compiler.
CXXFLAGS := -O3 -DNDEBUG
CXXFLAGS += --std=c++11 -g -DTF_LITE_STATIC_MEMORY
CCFLAGS := -DNDEBUG -g -DTF_LITE_STATIC_MEMORY
LDOPTS := -L/usr/local/lib
ARFLAGS := -r
TARGET_TOOLCHAIN_PREFIX :=
CC_PREFIX :=

# This library is the main target for this makefile. It will contain a minimal
# runtime that can be linked in to other programs.
MICROLITE_LIB_NAME := libtensorflow-microlite.a

MICROLITE_TEST_SRCS := \
$(wildcard tensorflow/lite/experimental/micro/*test.cc) \
$(wildcard tensorflow/lite/experimental/micro/kernels/*test.cc)

MICROLITE_TEST_HDRS := \
$(wildcard tensorflow/lite/experimental/micro/testing/*.h)

MICROLITE_CC_BASE_SRCS := \
$(wildcard tensorflow/lite/experimental/micro/*.cc) \
$(wildcard tensorflow/lite/experimental/micro/kernels/*.cc) \
tensorflow/lite/c/c_api_internal.c \
tensorflow/lite/core/api/error_reporter.cc \
tensorflow/lite/core/api/flatbuffer_conversions.cc \
tensorflow/lite/core/api/op_resolver.cc \
tensorflow/lite/kernels/kernel_util.cc \
tensorflow/lite/kernels/internal/quantization_util.cc
MICROLITE_CC_SRCS := $(filter-out $(MICROLITE_TEST_SRCS), $(MICROLITE_CC_BASE_SRCS))
MICROLITE_CC_SRCS := $(call specialize,$(MICROLITE_CC_SRCS))

MICROLITE_CC_HDRS := \
$(wildcard tensorflow/lite/experimental/micro/*.h) \
$(wildcard tensorflow/lite/experimental/micro/kernels/*.h) \
LICENSE \
tensorflow/lite/c/c_api_internal.h \
tensorflow/lite/c/builtin_op_data.h \
tensorflow/lite/core/api/error_reporter.h \
tensorflow/lite/core/api/flatbuffer_conversions.h \
tensorflow/lite/core/api/op_resolver.h \
tensorflow/lite/kernels/kernel_util.h \
tensorflow/lite/kernels/op_macros.h \
tensorflow/lite/kernels/padding.h \
tensorflow/lite/kernels/internal/common.h \
tensorflow/lite/kernels/internal/compatibility.h \
tensorflow/lite/kernels/internal/reference/depthwiseconv_float.h \
tensorflow/lite/kernels/internal/reference/depthwiseconv_uint8.h \
tensorflow/lite/kernels/internal/reference/fully_connected.h \
tensorflow/lite/kernels/internal/reference/softmax.h \
tensorflow/lite/kernels/internal/round.h \
tensorflow/lite/kernels/internal/tensor_ctypes.h \
tensorflow/lite/kernels/internal/types.h \
tensorflow/lite/kernels/internal/quantization_util.h \
tensorflow/lite/schema/schema_generated.h \
tensorflow/lite/version.h

THIRD_PARTY_CC_HDRS := \
third_party/gemmlowp/fixedpoint/fixedpoint.h \
third_party/gemmlowp/fixedpoint/fixedpoint_sse.h \
third_party/gemmlowp/internal/detect_platform.h \
third_party/gemmlowp/LICENSE \
third_party/flatbuffers/include/flatbuffers/base.h \
third_party/flatbuffers/include/flatbuffers/stl_emulation.h \
third_party/flatbuffers/include/flatbuffers/flatbuffers.h \
third_party/flatbuffers/LICENSE.txt

MAKE_PROJECT_FILES := \
  README_MAKE.md \
  Makefile

MBED_PROJECT_FILES := \
  README_MBED.md \
  mbed-os.lib \
  mbed_app.json

KEIL_PROJECT_FILES := \
  README_KEIL.md \
  keil_project.uvprojx

# These target-specific makefiles should modify or replace options like
# CXXFLAGS or LIBS to work for a specific targetted architecture. All logic
# based on platforms or architectures should happen within these files, to
# keep this main makefile focused on the sources and dependencies.
include $(wildcard $(MAKEFILE_DIR)/targets/*_makefile.inc)

ALL_TAGS += $(TARGET_ARCH)

ALL_SRCS := \
	$(MICROLITE_CC_SRCS) \
	$(MICROLITE_TEST_SRCS)

# Where compiled objects are stored.
GENDIR := $(MAKEFILE_DIR)/gen/$(TARGET)_$(TARGET_ARCH)/
OBJDIR := $(GENDIR)obj/
BINDIR := $(GENDIR)bin/
LIBDIR := $(GENDIR)lib/
PRJDIR := $(GENDIR)prj/

MICROLITE_LIB_PATH := $(LIBDIR)$(MICROLITE_LIB_NAME)

CXX := $(CC_PREFIX)${TARGET_TOOLCHAIN_PREFIX}g++
CC := $(CC_PREFIX)${TARGET_TOOLCHAIN_PREFIX}gcc
AR := $(CC_PREFIX)${TARGET_TOOLCHAIN_PREFIX}ar

# Load the examples.
include $(wildcard tensorflow/lite/experimental/micro/examples/*/Makefile.inc)

MICROLITE_LIB_OBJS := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(MICROLITE_CC_SRCS))))

# For normal manually-created TensorFlow C++ source files.
$(OBJDIR)%.o: %.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# For normal manually-created TensorFlow C source files.
$(OBJDIR)%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CCFLAGS) $(INCLUDES) -c $< -o $@

# For normal manually-created TensorFlow ASM source files.
$(OBJDIR)%.o: %.S
	@mkdir -p $(dir $@)
	$(CC) $(CCFLAGS) $(INCLUDES) -c $< -o $@

# The target that's compiled if there's no command-line arguments.
all: $(MICROLITE_LIB_PATH)

microlite: $(MICROLITE_LIB_PATH)

# Hack for generating schema file bypassing flatbuffer parsing
tensorflow/lite/schema/schema_generated.h:
	@cp -u tensorflow/lite/schema/schema_generated.h.OPENSOURCE tensorflow/lite/schema/schema_generated.h

# Gathers together all the objects we've compiled into a single '.a' archive.
$(MICROLITE_LIB_PATH): tensorflow/lite/schema/schema_generated.h $(MICROLITE_LIB_OBJS)
	@mkdir -p $(dir $@)
	$(AR) $(ARFLAGS) $(MICROLITE_LIB_PATH) $(MICROLITE_LIB_OBJS)

$(BINDIR)%_test : $(OBJDIR)%_test.o $(MICROLITE_LIB_PATH)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $@ $< \
	$(LIBFLAGS) $(MICROLITE_LIB_PATH) $(LDFLAGS) $(MICROLITE_LIBS)

$(BINDIR)%.test_target: $(BINDIR)%_test
	$(TEST_SCRIPT) $< '~~~ALL TESTS PASSED~~~'

# Generate standalone makefile projects for all of the test targets.
$(foreach TEST_TARGET,$(MICROLITE_TEST_SRCS),\
$(eval $(call microlite_test,$(notdir $(basename $(TEST_TARGET))),$(TEST_TARGET))))

test: $(MICROLITE_TEST_TARGETS)

# Gets rid of all generated files.
clean:
	rm -rf $(MAKEFILE_DIR)/gen

$(DEPDIR)/%.d: ;
.PRECIOUS: $(DEPDIR)/%.d
.PRECIOUS: $(BINDIR)%_test

-include $(patsubst %,$(DEPDIR)/%.d,$(basename $(ALL_SRCS)))
