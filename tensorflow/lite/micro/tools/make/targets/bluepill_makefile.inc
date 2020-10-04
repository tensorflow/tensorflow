# Settings for Blue Pill platforms.
ifeq ($(TARGET), bluepill)
  export PATH := $(MAKEFILE_DIR)/downloads/gcc_embedded/bin/:$(PATH)
  TARGET_ARCH := cortex-m3
  TARGET_TOOLCHAIN_PREFIX := arm-none-eabi-

  $(eval $(call add_third_party_download,$(GCC_EMBEDDED_URL),$(GCC_EMBEDDED_MD5),gcc_embedded,))
  $(eval $(call add_third_party_download,$(CMSIS_URL),$(CMSIS_MD5),cmsis,patch_cmsis))
  $(eval $(call add_third_party_download,$(STM32_BARE_LIB_URL),$(STM32_BARE_LIB_MD5),stm32_bare_lib,))

  PLATFORM_FLAGS = \
    -DGEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK \
    -DTF_LITE_MCU_DEBUG_LOG \
    -fmessage-length=0 \
    -fno-exceptions \
    -fno-unwind-tables \
    -ffunction-sections \
    -fdata-sections \
    -funsigned-char \
    -MMD \
    -mcpu=cortex-m3 \
    -mthumb \
    -Wall \
    -Wextra \
    -Wno-vla \
    -Wno-unused-parameter \
    -Wno-strict-aliasing \
    -Wno-shadow \
    -Wno-type-limits \
    -fno-delete-null-pointer-checks \
    -fomit-frame-pointer \
    -nostdlib \
    -g \
    -Os

  # TODO(b/168334217): Currently we always add -DNDEBUG because the build is
  # broken w/o it. Remove this workaround once the issue is resolved.
  PLATFORM_FLAGS += -DNDEBUG

  CXXFLAGS += $(PLATFORM_FLAGS) -fno-rtti -fno-threadsafe-statics \
              -fno-use-cxa-atexit
  CCFLAGS += $(PLATFORM_FLAGS)
  LDFLAGS += \
    -T $(MAKEFILE_DIR)/targets/bluepill/bluepill.lds \
    -Wl,-Map=$(MAKEFILE_DIR)/gen/$(TARGET).map,--cref \
    -Wl,--gc-sections
  BUILD_TYPE := micro
  MICROLITE_LIBS := \
    -lm
  INCLUDES += \
    -isystem$(MAKEFILE_DIR)/downloads/cmsis/CMSIS/Core/Include/ \
    -I$(MAKEFILE_DIR)/downloads/stm32_bare_lib/include
  MICROLITE_CC_SRCS += \
    $(wildcard $(MAKEFILE_DIR)/downloads/stm32_bare_lib/source/*.c) \
    $(wildcard $(MAKEFILE_DIR)/downloads/stm32_bare_lib/source/*.cc)
  EXCLUDED_SRCS := \
    $(MAKEFILE_DIR)/downloads/stm32_bare_lib/source/debug_log.c
  MICROLITE_CC_SRCS := $(filter-out $(EXCLUDED_SRCS), $(MICROLITE_CC_SRCS))
  TEST_SCRIPT := tensorflow/lite/micro/testing/test_bluepill_binary.sh

  # TODO(b/143286954): Figure out why some tests fail and enable ince the issues
  # are resolved.
  EXCLUDED_TESTS := \
    tensorflow/lite/micro/micro_interpreter_test.cc \
    tensorflow/lite/micro/micro_allocator_test.cc \
    tensorflow/lite/micro/memory_helpers_test.cc \
    tensorflow/lite/micro/memory_arena_threshold_test.cc \
    tensorflow/lite/micro/kernels/circular_buffer_test.cc
  MICROLITE_TEST_SRCS := $(filter-out $(EXCLUDED_TESTS), $(MICROLITE_TEST_SRCS))

  EXCLUDED_EXAMPLE_TESTS := \
    tensorflow/lite/micro/examples/magic_wand/Makefile.inc \
    tensorflow/lite/micro/examples/micro_speech/Makefile.inc \
    tensorflow/lite/micro/examples/image_recognition_experimental/Makefile.inc
  MICRO_LITE_EXAMPLE_TESTS := $(filter-out $(EXCLUDED_EXAMPLE_TESTS), $(MICRO_LITE_EXAMPLE_TESTS))


# These are microcontroller-specific rules for converting the ELF output
# of the linker into a binary image that can be loaded directly.
OBJCOPY := $(TARGET_TOOLCHAIN_PREFIX)objcopy

$(BINDIR)/%.bin: $(BINDIR)/%
	@mkdir -p $(dir $@)
	$(OBJCOPY) $< $@ -O binary

endif
