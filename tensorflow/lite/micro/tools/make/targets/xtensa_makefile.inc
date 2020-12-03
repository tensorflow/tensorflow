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

ifndef XTENSA_BASE
  $(error XTENSA_BASE is undefined)
endif

ifndef XTENSA_TOOLS_VERSION
  $(error XTENSA_TOOLS_VERSION is undefined)
endif

ifndef XTENSA_CORE
  $(error XTENSA_CORE is undefined)
endif

PLATFORM_FLAGS = \
  -DTF_LITE_MCU_DEBUG_LOG \
  -DTF_LITE_USE_CTIME \
  --xtensa-core=$(XTENSA_CORE) \
  -mcoproc \
  -DXTENSA \
  -DMAX_RFFT_PWR=9 \
  -DMIN_RFFT_PWR=MAX_RFFT_PWR

ifeq ($(BUILD_TYPE), release)
  PLATFORM_FLAGS += -Wno-unused-private-field
endif

export PATH := $(XTENSA_BASE)/tools/$(XTENSA_TOOLS_VERSION)/XtensaTools/bin:$(PATH)
TARGET_TOOLCHAIN_PREFIX := xt-
CXX_TOOL := clang++
CC_TOOL := clang

CXXFLAGS += $(PLATFORM_FLAGS)
CCFLAGS += $(PLATFORM_FLAGS)

# TODO(b/150240249): Do not remove -fno-rtti once that works for the Xtensa toolchain.
CXXFLAGS := $(filter-out -fno-rtti, $(CXXFLAGS))

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
  tensorflow/lite/micro/examples/micro_speech/Makefile.inc \
  tensorflow/lite/micro/examples/network_tester/Makefile.inc \
  tensorflow/lite/micro/examples/person_detection/Makefile.inc \
  tensorflow/lite/micro/examples/person_detection_experimental/Makefile.inc
MICRO_LITE_EXAMPLE_TESTS := $(filter-out $(EXCLUDED_EXAMPLE_TESTS), $(MICRO_LITE_EXAMPLE_TESTS))

