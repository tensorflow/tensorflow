MAKEFILE_DIR := tensorflow/lite/experimental/micro/tools/make

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

# Look for platform or target-specific implementation files to replace reference
# implementations with, given a tag. These are expected to occur in subfolders
# of a directory where a reference implementation exists, and have the same
# interface and header file. For example,
# tensorflow/lite/experimental/micro/examples/micro_speech/audio_provider.cc
# defines a module for supplying audio data, but since no platform or OS can be
# presumed, it just always returns zeroes for its samples. The MacOS-specific
# tensorflow/lite/experimental/micro/examples/micro_speech/osx/audio_provider.cc
# has an implementation that relies on CoreAudio, and there are equivalent
# versions for other operating systems. 
# All lists of source files are put through this substitution process with the
# tags of their target OS and architecture, so that implementations can be added
# by simply placing them in the file tree, with no changes to the build files
# needed.
# One confusing thing about this implementation is that we're using wildcard to
# act as a 'does file exist?' function, rather than expanding an expression.
# Wildcard will return an empty string if given a plain file path with no actual
# wildcards, if the file doesn't exist, so taking the first word of the list
# between that and the reference path will pick the specialized one if it's
# available.
substitute_specialized_implementation = \
  $(firstword $(wildcard $(dir $(1))$(2)/$(notdir $(1))) $(wildcard $(1)))
substitute_specialized_implementations = \
  $(foreach source,$(1),$(call substitute_specialized_implementation,$(source),$(2)))
# Here we're first looking for specialized implementations in ref_dir/$(TARGET)
# and then ref_dir/$(TARGET_ARCH), before falling back to ref_dir's
# implementation.
# The argument to this function should be a list of space-separated file paths,
# with any wildcards already expanded.
specialize = \
  $(call substitute_specialized_implementations,$(call substitute_specialized_implementations,$(1),$(TARGET)),$(TARGET_ARCH))

INCLUDES := \
-I. \
-I$(MAKEFILE_DIR)/../../../../../ \
-I$(MAKEFILE_DIR)/../../../../../../ \
-I$(MAKEFILE_DIR)/../../../../../../../ \
-I$(MAKEFILE_DIR)/downloads/ \
-I$(MAKEFILE_DIR)/downloads/gemmlowp \
-I$(MAKEFILE_DIR)/downloads/flatbuffers/include \
-I$(OBJDIR)
# This is at the end so any globally-installed frameworks like protobuf don't
# override local versions in the source tree.
INCLUDES += -I/usr/local/include

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

# These target-specific makefiles should modify or replace options like
# CXXFLAGS or LIBS to work for a specific targetted architecture. All logic
# based on platforms or architectures should happen within these files, to
# keep this main makefile focused on the sources and dependencies.
include $(wildcard $(MAKEFILE_DIR)/targets/*_makefile.inc)

ALL_SRCS := \
	$(MICROLITE_CC_SRCS) \
	$(MICROLITE_TEST_SRCS)

# Where compiled objects are stored.
GENDIR := $(MAKEFILE_DIR)/gen/$(TARGET)_$(TARGET_ARCH)/
OBJDIR := $(GENDIR)obj/
BINDIR := $(GENDIR)bin/
LIBDIR := $(GENDIR)lib/

MICROLITE_LIB_PATH := $(LIBDIR)$(MICROLITE_LIB_NAME)

CXX := $(CC_PREFIX)${TARGET_TOOLCHAIN_PREFIX}g++
CC := $(CC_PREFIX)${TARGET_TOOLCHAIN_PREFIX}gcc
AR := $(CC_PREFIX)${TARGET_TOOLCHAIN_PREFIX}ar

# Load the examples.
include $(wildcard tensorflow/lite/experimental/micro/examples/*/Makefile.inc)

MICROLITE_LIB_OBJS := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(MICROLITE_CC_SRCS))))

MICROLITE_TEST_TARGETS := $(addprefix $(BINDIR), \
$(patsubst %_test.cc,%.test_target,$(MICROLITE_TEST_SRCS)))

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

# snease: Add %.bin rule here since BINDIR is now defined
# These are microcontroller-specific rules for converting the ELF output
# of the linker into a binary image that can be loaded directly.
OBJCOPY := $(TARGET_TOOLCHAIN_PREFIX)objcopy
$(BINDIR)%.bin: $(BINDIR)%
	@mkdir -p $(dir $@)
	$(OBJCOPY) $< $@ -O binary

test: test_micro_speech $(MICROLITE_TEST_TARGETS)

# Gets rid of all generated files.
clean:
	rm -rf $(MAKEFILE_DIR)/gen

$(DEPDIR)/%.d: ;
.PRECIOUS: $(DEPDIR)/%.d
.PRECIOUS: $(BINDIR)%_test

-include $(patsubst %,$(DEPDIR)/%.d,$(basename $(ALL_SRCS)))
