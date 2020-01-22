# Settings for Blue Pill platforms.
ifeq ($(TARGET), bluepill)
  export PATH := $(MAKEFILE_DIR)/downloads/gcc_embedded/bin/:$(PATH)
  TARGET_ARCH := cortex-m3
  TARGET_TOOLCHAIN_PREFIX := arm-none-eabi-

  $(eval $(call add_third_party_download,$(GCC_EMBEDDED_URL),$(GCC_EMBEDDED_MD5),gcc_embedded,))
  $(eval $(call add_third_party_download,$(CMSIS_URL),$(CMSIS_MD5),cmsis,))
  $(eval $(call add_third_party_download,$(STM32_BARE_LIB_URL),$(STM32_BARE_LIB_MD5),stm32_bare_lib,))

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
    -Wsign-compare \
    -Wdouble-promotion \
    -Wshadow \
    -Wunused-variable \
    -Wno-unused-parameter \
    -Wno-missing-field-initializers \
    -Wno-write-strings \
    -fno-delete-null-pointer-checks \
    -fomit-frame-pointer \
    -fpermissive \
    -nostdlib \
    -g \
    -Os
  CXXFLAGS += $(PLATFORM_FLAGS)
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
  # These are tests that don't currently work on the blue pill.
  EXCLUDED_TESTS := \
    tensorflow/lite/micro/micro_interpreter_test.cc \
    tensorflow/lite/micro/simple_tensor_allocator_test.cc
  MICROLITE_TEST_SRCS := $(filter-out $(EXCLUDED_TESTS), $(MICROLITE_TEST_SRCS))

# These are microcontroller-specific rules for converting the ELF output
# of the linker into a binary image that can be loaded directly.
OBJCOPY := $(TARGET_TOOLCHAIN_PREFIX)objcopy

$(BINDIR)/%.bin: $(BINDIR)/%
	@mkdir -p $(dir $@)
	$(OBJCOPY) $< $@ -O binary

endif
