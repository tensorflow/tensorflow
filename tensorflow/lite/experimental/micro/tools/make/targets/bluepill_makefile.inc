# Settings for Blue Pill platforms.
ifeq ($(TARGET), bluepill)
  TARGET_ARCH := cortex-m3
  TARGET_TOOLCHAIN_PREFIX := arm-none-eabi-

  PLATFORM_FLAGS = \
    -DGEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK \
    -DTF_LITE_STATIC_MEMORY \
    -DTF_LITE_MCU_DEBUG_LOG \
    -fno-rtti \
    -fmessage-length=0 \
    -fno-exceptions \
    -fno-unwind-tables \
    -fno-builtin \
    -ffunction-sections \
    -fdata-sections \
    -funsigned-char \
    -MMD \
    -mcpu=cortex-m3 \
    -mthumb \
    -std=gnu++11 \
    -Wvla \
    -Wall \
    -Wextra \
    -Wno-unused-parameter \
    -Wno-missing-field-initializers \
    -Wno-write-strings \
    -Wno-sign-compare \
    -fno-delete-null-pointer-checks \
    -fomit-frame-pointer \
    -fpermissive \
    -nostdlib \
    -g \
    -Os
  CXXFLAGS += $(PLATFORM_FLAGS)
  CCFLAGS += $(PLATFORM_FLAGS)
  LDFLAGS += \
    -T $(MAKEFILE_DIR)/downloads/stm32_bare_lib/stm32_linker_layout.lds \
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
    TEST_SCRIPT := tensorflow/lite/experimental/micro/testing/test_bluepill_binary.sh
  # These are tests that don't currently work on the blue pill.
  EXCLUDED_TESTS := \
    tensorflow/lite/experimental/micro/micro_interpreter_test.cc \
    tensorflow/lite/experimental/micro/simple_tensor_allocator_test.cc
  MICROLITE_TEST_SRCS := $(filter-out $(EXCLUDED_TESTS), $(MICROLITE_TEST_SRCS))

# These are microcontroller-specific rules for converting the ELF output
# of the linker into a binary image that can be loaded directly.
OBJCOPY := $(TARGET_TOOLCHAIN_PREFIX)objcopy

$(BINDIR)/%.bin: $(BINDIR)/%
	@mkdir -p $(dir $@)
	$(OBJCOPY) $< $@ -O binary

endif