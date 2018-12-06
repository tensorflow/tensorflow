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

INCLUDES := \
-I. \
-I$(MAKEFILE_DIR)/../../../../../ \
-I$(MAKEFILE_DIR)/../../../../../../ \
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

# Test binary for the microcontroller speech model.
MICRO_SPEECH_TEST_SRCS := \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_speech_test.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/tiny_conv_model_data.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/no_features_data.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/yes_features_data.cc

# Test binary for the streaming microcontroller speech model.
PUSHBUTTON_MICRO_SPEECH_TEST_SRCS := \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_speech_test.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/tiny_conv_model_data.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/no_features_data.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/yes_features_data.cc

# Test binary for the streaming microcontroller speech model.
PUSHBUTTON_CMSIS_SPEECH_TEST_SRCS := \
tensorflow/lite/experimental/micro/examples/micro_speech/micro_speech_test.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/tiny_conv_model_data.cc

# Test binary for the microcontroller speech model.
PREPROCESSOR_TEST_SRCS := \
tensorflow/lite/experimental/micro/examples/micro_speech/preprocessor_test.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/no_30ms_sample_data.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/yes_30ms_sample_data.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/no_power_spectrum_data.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/yes_power_spectrum_data.cc

PREPROCESSOR_REFERENCE_TEST_SRCS = \
$(PREPROCESSOR_TEST_SRCS) \
tensorflow/lite/experimental/micro/examples/micro_speech/preprocessor.cc

PREPROCESSOR_FIXED_TEST_SRCS += \
$(PREPROCESSOR_TEST_SRCS) \
tensorflow/lite/experimental/micro/examples/micro_speech/fixed_point/preprocessor.cc

PREPROCESSOR_1K_SRCS := \
tensorflow/lite/experimental/micro/examples/micro_speech/apollo3/preprocessor_1k.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/CMSIS/sin_1k.cc


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

# These target-specific makefiles should modify or replace options like
# CXXFLAGS or LIBS to work for a specific targetted architecture. All logic
# based on platforms or architectures should happen within these files, to
# keep this main makefile focused on the sources and dependencies.
include $(wildcard $(MAKEFILE_DIR)/targets/*_makefile.inc)

ALL_SRCS := \
	$(MICRO_SPEECH_TEST_SRCS) \
	$(PUSHBUTTON_MICRO_SPEECH_TEST_SRCS) \
	$(PUSHBUTTON_CMSIS_SPEECH_TEST_SRCS) \
	$(PREPROCESSOR_REFERENCE_TEST_SRCS) \
	$(PREPROCESSOR_FIXED_TEST_SRCS) \
	$(MICROLITE_CC_SRCS) \
	$(PREPROCESSOR_1K_MICRO_TEST_SRCS) \
	$(PREPROCESSOR_1K_CMSIS_TEST_SRCS) \
	$(MICROLITE_TEST_SRCS)

# Where compiled objects are stored.
GENDIR := $(MAKEFILE_DIR)/gen/$(TARGET)_$(TARGET_ARCH)/
OBJDIR := $(GENDIR)obj/
BINDIR := $(GENDIR)bin/
LIBDIR := $(GENDIR)lib/

MICROLITE_LIB_PATH := $(LIBDIR)$(MICROLITE_LIB_NAME)

MICRO_SPEECH_TEST_BINARY := $(BINDIR)micro_speech_test
PUSHBUTTON_MICRO_SPEECH_TEST_BINARY := $(BINDIR)pushbutton_micro_speech_test
PUSHBUTTON_CMSIS_SPEECH_TEST_BINARY := $(BINDIR)pushbutton_cmsis_speech_test
PREPROCESSOR_REFERENCE_TEST_BINARY := $(BINDIR)preprocessor_reference_test
PREPROCESSOR_FIXED_TEST_BINARY := $(BINDIR)preprocessor_fixed_test
PREPROCESSOR_1K_MICRO_TEST_BINARY := $(BINDIR)preprocessor_1k_micro_test
PREPROCESSOR_1K_CMSIS_TEST_BINARY := $(BINDIR)preprocessor_1k_cmsis_test

CXX := $(CC_PREFIX)${TARGET_TOOLCHAIN_PREFIX}g++
CC := $(CC_PREFIX)${TARGET_TOOLCHAIN_PREFIX}gcc
AR := $(CC_PREFIX)${TARGET_TOOLCHAIN_PREFIX}ar

MICRO_SPEECH_TEST_OBJS := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(MICRO_SPEECH_TEST_SRCS))))

PUSHBUTTON_MICRO_SPEECH_TEST_OBJS := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(PUSHBUTTON_MICRO_SPEECH_TEST_SRCS))))

PUSHBUTTON_CMSIS_SPEECH_TEST_OBJS := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(PUSHBUTTON_CMSIS_SPEECH_TEST_SRCS))) \
arm_bitreversal2.o)

PREPROCESSOR_REFERENCE_TEST_OBJS := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(PREPROCESSOR_REFERENCE_TEST_SRCS))))

PREPROCESSOR_FIXED_TEST_OBJS := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(PREPROCESSOR_FIXED_TEST_SRCS))))

PREPROCESSOR_1K_MICRO_TEST_OBJS := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(PREPROCESSOR_1K_MICRO_TEST_SRCS))))

PREPROCESSOR_1K_CMSIS_TEST_OBJS := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(PREPROCESSOR_1K_CMSIS_TEST_SRCS))) \
arm_bitreversal2.o)

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

# The target that's compiled if there's no command-line arguments.
all: $(MICROLITE_LIB_PATH) $(MICRO_SPEECH_TEST_BINARY) $(PREPROCESSOR_TEST_BINARY)

microlite: $(MICROLITE_LIB_PATH)

# Hack for generating schema file bypassing flatbuffer parsing
tensorflow/lite/schema/schema_generated.h:
	@cp -u tensorflow/lite/schema/schema_generated.h.OPENSOURCE tensorflow/lite/schema/schema_generated.h

# Gathers together all the objects we've compiled into a single '.a' archive.
$(MICROLITE_LIB_PATH): tensorflow/lite/schema/schema_generated.h $(MICROLITE_LIB_OBJS)
	@mkdir -p $(dir $@)
	$(AR) $(ARFLAGS) $(MICROLITE_LIB_PATH) $(MICROLITE_LIB_OBJS)

$(MICRO_SPEECH_TEST_BINARY): $(MICRO_SPEECH_TEST_OBJS) $(MICROLITE_LIB_PATH)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $(MICRO_SPEECH_TEST_BINARY) $(MICRO_SPEECH_TEST_OBJS) \
	$(LIBFLAGS) $(MICROLITE_LIB_PATH) $(LDFLAGS) $(MICROLITE_LIBS)

micro_speech_test: $(MICRO_SPEECH_TEST_BINARY)
micro_speech_test_bin: $(MICRO_SPEECH_TEST_BINARY).bin

test_micro_speech: $(MICRO_SPEECH_TEST_BINARY)
	$(TEST_SCRIPT) $(MICRO_SPEECH_TEST_BINARY) '~~~ALL TESTS PASSED~~~'

$(PUSHBUTTON_MICRO_SPEECH_TEST_BINARY): $(PUSHBUTTON_MICRO_SPEECH_TEST_OBJS) $(MICROLITE_LIB_PATH)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $(PUSHBUTTON_MICRO_SPEECH_TEST_BINARY) $(PUSHBUTTON_MICRO_SPEECH_TEST_OBJS) \
	$(LIBFLAGS) $(MICROLITE_LIB_PATH) $(LDFLAGS) $(MICROLITE_LIBS)

pushbutton_micro_speech_test: $(PUSHBUTTON_MICRO_SPEECH_TEST_BINARY)
pushbutton_micro_speech_test_bin: $(PUSHBUTTON_MICRO_SPEECH_TEST_BINARY).bin

test_pushbutton_micro_speech: $(PUSHBUTTON_MICRO_SPEECH_TEST_BINARY)
	$(TEST_SCRIPT) $(PUSHBUTTON_MICRO_SPEECH_TEST_BINARY) '~~~ALL TESTS PASSED~~~'


$(PUSHBUTTON_CMSIS_SPEECH_TEST_BINARY): $(PUSHBUTTON_CMSIS_SPEECH_TEST_OBJS) $(MICROLITE_LIB_PATH)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $(PUSHBUTTON_CMSIS_SPEECH_TEST_BINARY) $(PUSHBUTTON_CMSIS_SPEECH_TEST_OBJS) \
	$(LIBFLAGS) $(MICROLITE_LIB_PATH) $(LDFLAGS) $(MICROLITE_LIBS)

pushbutton_cmsis_speech_test: $(PUSHBUTTON_CMSIS_SPEECH_TEST_BINARY)
pushbutton_cmsis_speech_test_bin: $(PUSHBUTTON_CMSIS_SPEECH_TEST_BINARY).bin

test_pushbutton_cmsis_speech: $(PUSHBUTTON_CMSIS_SPEECH_TEST_BINARY)
	$(TEST_SCRIPT) $(PUSHBUTTON_CMSIS_SPEECH_TEST_BINARY) '~~~ALL TESTS PASSED~~~'

$(PREPROCESSOR_REFERENCE_TEST_BINARY): $(PREPROCESSOR_REFERENCE_TEST_OBJS) $(MICROLITE_LIB_PATH)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $(PREPROCESSOR_REFERENCE_TEST_BINARY) $(PREPROCESSOR_REFERENCE_TEST_OBJS) \
	$(LIBFLAGS) $(MICROLITE_LIB_PATH) $(LDFLAGS) $(MICROLITE_LIBS)

preprocessor_reference_test: $(PREPROCESSOR_REFERENCE_TEST_BINARY)
preprocessor_reference_test_bin: $(PREPROCESSOR_REFERENCE_TEST_BINARY).bin

test_preprocessor_reference: $(PREPROCESSOR_REFERENCE_TEST_BINARY)
	$(TEST_SCRIPT) $(PREPROCESSOR_REFERENCE_TEST_BINARY) '~~~ALL TESTS PASSED~~~'

$(PREPROCESSOR_FIXED_TEST_BINARY): $(PREPROCESSOR_FIXED_TEST_OBJS) $(MICROLITE_LIB_PATH)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $(PREPROCESSOR_FIXED_TEST_BINARY) $(PREPROCESSOR_FIXED_TEST_OBJS) \
	$(LIBFLAGS) $(MICROLITE_LIB_PATH) $(LDFLAGS) $(MICROLITE_LIBS)

preprocessor_fixed_test: $(PREPROCESSOR_FIXED_TEST_BINARY)
preprocessor_fixed_test_bin: $(PREPROCESSOR_FIXED_TEST_BINARY).bin

test_preprocessor_fixed: $(PREPROCESSOR_FIXED_TEST_BINARY)
	$(TEST_SCRIPT) $(PREPROCESSOR_FIXED_TEST_BINARY) '~~~ALL TESTS PASSED~~~'

$(PREPROCESSOR_1K_MICRO_TEST_BINARY): $(PREPROCESSOR_1K_MICRO_TEST_OBJS) $(MICROLITE_LIB_PATH)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
  	-o $(PREPROCESSOR_1K_MICRO_TEST_BINARY) $(PREPROCESSOR_1K_MICRO_TEST_OBJS) \
  	$(LIBFLAGS) $(MICROLITE_LIB_PATH) $(LDFLAGS) $(MICROLITE_LIBS)

preprocessor_1k_micro_test: $(PREPROCESSOR_1K_MICRO_TEST_BINARY)
preprocessor_1k_micro_test_bin: $(PREPROCESSOR_1K_MICRO_TEST_BINARY).bin

test_preprocessor_1k_micro: $(PREPROCESSOR_1K_MICRO_TEST_BINARY)
	$(TEST_SCRIPT) $(PREPROCESSOR_1K_MICRO_TEST_BINARY) '~~~ALL TESTS PASSED~~~'


$(PREPROCESSOR_1K_CMSIS_TEST_BINARY): $(PREPROCESSOR_1K_CMSIS_TEST_OBJS) $(MICROLITE_LIB_PATH)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
  	-o $(PREPROCESSOR_1K_CMSIS_TEST_BINARY) $(PREPROCESSOR_1K_CMSIS_TEST_OBJS) \
  	$(LIBFLAGS) $(MICROLITE_LIB_PATH) $(LDFLAGS) $(MICROLITE_LIBS)

preprocessor_1k_cmsis_test: $(PREPROCESSOR_1K_CMSIS_TEST_BINARY)
preprocessor_1k_cmsis_test_bin: $(PREPROCESSOR_1K_CMSIS_TEST_BINARY).bin

test_preprocessor_1k_cmsis: $(PREPROCESSOR_1K_CMSIS_TEST_BINARY)
	$(TEST_SCRIPT) $(PREPROCESSOR_1K_CMSIS_TEST_BINARY) '~~~ALL TESTS PASSED~~~'


$(OBJDIR)arm_bitreversal2.o:
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $(CMSIS_SRC_DIR)/TransformFunctions/arm_bitreversal2.S -o $(OBJDIR)arm_bitreversal2.o

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

$(info $(MICROLITE_TEST_TARGETS))

test: test_micro_speech $(MICROLITE_TEST_TARGETS)

# Gets rid of all generated files.
clean:
	rm -rf $(MAKEFILE_DIR)/gen

$(DEPDIR)/%.d: ;
.PRECIOUS: $(DEPDIR)/%.d
.PRECIOUS: $(BINDIR)%_test

-include $(patsubst %,$(DEPDIR)/%.d,$(basename $(ALL_SRCS)))
