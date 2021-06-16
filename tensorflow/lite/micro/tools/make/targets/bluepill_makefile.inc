export PATH := $(MAKEFILE_DIR)/downloads/gcc_embedded/bin/:$(PATH)
TARGET_ARCH := cortex-m3
TARGET_TOOLCHAIN_PREFIX := arm-none-eabi-

DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/arm_gcc_download.sh ${MAKEFILE_DIR}/downloads)
ifneq ($(DOWNLOAD_RESULT), SUCCESS)
  $(error Something went wrong with the GCC download: $(DOWNLOAD_RESULT))
endif

DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/renode_download.sh ${MAKEFILE_DIR}/downloads)
ifneq ($(DOWNLOAD_RESULT), SUCCESS)
  $(error Something went wrong with the renode download: $(DOWNLOAD_RESULT))
endif

DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/ext_libs/cmsis_download.sh ${MAKEFILE_DIR}/downloads)
ifneq ($(DOWNLOAD_RESULT), SUCCESS)
  $(error Something went wrong with the CMSIS download: $(DOWNLOAD_RESULT))
endif

DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/ext_libs/stm32_bare_lib_download.sh ${MAKEFILE_DIR}/downloads)
ifneq ($(DOWNLOAD_RESULT), SUCCESS)
  $(error Something went wrong with the STM32 Bare Lib download: $(DOWNLOAD_RESULT))
endif

PLATFORM_FLAGS = \
  -DTF_LITE_MCU_DEBUG_LOG \
  -mcpu=cortex-m3 \
  -mthumb \
  -Wno-vla \
  -Wno-strict-aliasing \
  -Wno-shadow \
  -Wno-type-limits \
  -fomit-frame-pointer \
  -nostdlib

# TODO(b/168334217): Currently we always add -DNDEBUG because the build is
# broken w/o it. Remove this workaround once the issue is resolved.
PLATFORM_FLAGS += -DNDEBUG

# TODO(#46937): Remove once initialization of global variables is sorted out.
PLATFORM_FLAGS += -DRENODE

CXXFLAGS += $(PLATFORM_FLAGS) -fno-use-cxa-atexit
CCFLAGS += $(PLATFORM_FLAGS)

LDFLAGS += \
  -T $(MAKEFILE_DIR)/targets/bluepill/bluepill.lds \
  -Wl,-Map=$(MAKEFILE_DIR)/gen/$(TARGET).map,--cref

# Additional include paths needed for the stm_32_bare_lib only.
INCLUDES += \
  -isystem$(MAKEFILE_DIR)/downloads/cmsis/CMSIS/Core/Include/ \
  -I$(MAKEFILE_DIR)/downloads/stm32_bare_lib/include

MICROLITE_CC_SRCS += \
  $(wildcard $(MAKEFILE_DIR)/downloads/stm32_bare_lib/source/*.c) \
  $(wildcard $(MAKEFILE_DIR)/downloads/stm32_bare_lib/source/*.cc)
EXCLUDED_SRCS := \
  $(MAKEFILE_DIR)/downloads/stm32_bare_lib/source/debug_log.c
MICROLITE_CC_SRCS := $(filter-out $(EXCLUDED_SRCS), $(MICROLITE_CC_SRCS))

# TODO(b/158651472): Fix the memory_arena_threshold_test
# TODO(b/143286954): Figure out why some tests fail and enable once the issues
# are resolved.
EXCLUDED_TESTS := \
  tensorflow/lite/micro/micro_interpreter_test.cc \
  tensorflow/lite/micro/micro_allocator_test.cc \
  tensorflow/lite/micro/memory_helpers_test.cc \
  tensorflow/lite/micro/memory_arena_threshold_test.cc
MICROLITE_TEST_SRCS := $(filter-out $(EXCLUDED_TESTS), $(MICROLITE_TEST_SRCS))

EXCLUDED_EXAMPLE_TESTS := \
  tensorflow/lite/micro/examples/magic_wand/Makefile.inc \
  tensorflow/lite/micro/examples/micro_speech/Makefile.inc \
  tensorflow/lite/micro/examples/image_recognition_experimental/Makefile.inc
MICRO_LITE_EXAMPLE_TESTS := $(filter-out $(EXCLUDED_EXAMPLE_TESTS), $(MICRO_LITE_EXAMPLE_TESTS))

TEST_SCRIPT := tensorflow/lite/micro/testing/test_with_renode.sh

# We are setting this variable to non-zero to allow us to have a custom
# implementation of `make test` for bluepill
TARGET_SPECIFIC_MAKE_TEST := 1
test: build
	$(TEST_SCRIPT) $(BINDIR) $(TEST_PASS_STRING) $(TARGET)
