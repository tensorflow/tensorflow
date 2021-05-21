# Settings for Xtensa toolchain for the hifimini kernels.
# REQUIRED:
#  Environment variables:
#   - XTENSA_BASE  must be set to location of
#     the Xtensa developer tools installation directory.
#  Command line arguments:
#   - XTENSA_TOOLS_VERSION: For example: RI-2019.2-linux
#   - XTENSA_CORE: The name of the Xtensa core to use
#      For example: hifimini

TARGET_ARCH :=
XTENSA_USE_LIBC :=

ifndef XTENSA_BASE
  $(error XTENSA_BASE is undefined)
endif

ifndef XTENSA_TOOLS_VERSION
  $(error XTENSA_TOOLS_VERSION is undefined)
endif

ifndef XTENSA_CORE
  $(error XTENSA_CORE is undefined)
endif

ifeq ($(TARGET_ARCH), )
  $(error TARGET_ARCH must be specified on the command line)
endif

# Create a cflag based on the specified TARGET_ARCH. For example:
#   TARGET_ARCH=hifimini --> -DHIFIMINI
#   TARGET_ARCH=fusion_f1 --> -DFUSION_F1
TARGET_ARCH_DEFINES := -D$(shell echo $(TARGET_ARCH) | tr [a-z] [A-Z])

PLATFORM_FLAGS = \
  -DTF_LITE_MCU_DEBUG_LOG \
  -DTF_LITE_USE_CTIME \
  --xtensa-core=$(XTENSA_CORE) \
  -mcoproc \
  -DMAX_RFFT_PWR=9 \
  -DMIN_RFFT_PWR=MAX_RFFT_PWR \
  $(TARGET_ARCH_DEFINES) \
  -mlongcalls

ifeq ($(BUILD_TYPE), release)
  PLATFORM_FLAGS += -Wno-unused-private-field
endif

export PATH := $(XTENSA_BASE)/tools/$(XTENSA_TOOLS_VERSION)/XtensaTools/bin:$(PATH)
TARGET_TOOLCHAIN_PREFIX := xt-
CXX_TOOL := clang++
CC_TOOL := clang

# Unused exception related symbols make their way into a binary that links
# against TFLM as described in https://github.com/tensorflow/tensorflow/issues/47575.
# We have two options to avoid this. The first involves using -stdlib=libc++ and
# the second involves stubbing out and modifying some of the files in the Xtensa
# toolchain to prevent inclusion of the exception handling code
# (http://b/182209217#comment3). This Makefile supports building TFLM in a way
# that is compatible with either of the two approaches.
ifeq ($(XTENSA_USE_LIBC), true)
  PLATFORM_FLAGS += -stdlib=libc++
else
  # TODO(b/150240249): Do not filter-out -fno-rtti once that works for the
  # Xtensa toolchain.
  CXXFLAGS := $(filter-out -fno-rtti, $(CXXFLAGS))
endif

CXXFLAGS += $(PLATFORM_FLAGS)
CCFLAGS += $(PLATFORM_FLAGS)

TEST_SCRIPT := tensorflow/lite/micro/testing/test_xtensa_binary.sh

# TODO(b/158651472): Fix the memory_arena_threshold_test
# TODO(b/174707181): Fix the micro_interpreter_test
EXCLUDED_TESTS := \
  tensorflow/lite/micro/micro_interpreter_test.cc \
  tensorflow/lite/micro/memory_arena_threshold_test.cc
MICROLITE_TEST_SRCS := $(filter-out $(EXCLUDED_TESTS), $(MICROLITE_TEST_SRCS))

# TODO(b/156962140): This manually maintained list of excluded examples is
# quite error prone.
EXCLUDED_EXAMPLE_TESTS := \
  tensorflow/lite/micro/examples/hello_world/Makefile.inc \
  tensorflow/lite/micro/examples/image_recognition_experimental/Makefile.inc \
  tensorflow/lite/micro/examples/magic_wand/Makefile.inc \
  tensorflow/lite/micro/examples/network_tester/Makefile.inc
MICRO_LITE_EXAMPLE_TESTS := $(filter-out $(EXCLUDED_EXAMPLE_TESTS), $(MICRO_LITE_EXAMPLE_TESTS))

