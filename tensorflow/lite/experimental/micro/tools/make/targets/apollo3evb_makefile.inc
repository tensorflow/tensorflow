# Settings for apollo3 evb platforms.
ifeq ($(TARGET), apollo3evb)
  TARGET_ARCH := cortex-m4
  TARGET_TOOLCHAIN_PREFIX := arm-none-eabi-
  # Download the Ambiq Apollo3 SDK and set this variable to find the header
  # files:
  APOLLO3_SDK := /ssd/ambiq/AmbiqSuite\ SDK\ for\ Apollo3/Apollo3-SDK-2018.08.13/
  # Need a pointer to the GNU ARM toolchain for crtbegin.o for the fp functions
  # with the softfp interfaces.
  GCC_ARM := /ssd/gnu_arm_toolchain/gcc-arm-none-eabi-7-2018-q2-update/

  PLATFORM_FLAGS = \
    -DPART_apollo3 \
    -DAM_PACKAGE_BGA \
    -DAM_PART_APOLLO3 \
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
    -mcpu=cortex-m4 \
    -mthumb \
    -mfpu=fpv4-sp-d16 \
    -mfloat-abi=softfp \
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
    -mthumb -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=softfp \
    -nostartfiles -static \
    -Wl,--gc-sections -Wl,--entry,Reset_Handler \
    -Wl,--start-group -lm -lc -lgcc -Wl,--end-group \
    -fno-exceptions \
    -nostdlib --specs=nano.specs -t -lstdc++ -lc -lnosys -lm \
    -Wl,-T,$(MAKEFILE_DIR)/targets/apollo3evb/apollo3evb.ld \
    -Wl,-Map=$(MAKEFILE_DIR)/gen/$(TARGET).map,--cref
  BUILD_TYPE := micro
  # The apollo3evb libs should be copied from the SDK after building them.
  MICROLITE_LIBS := \
    $(MAKEFILE_DIR)/targets/apollo3evb/libam_bsp.a \
    $(MAKEFILE_DIR)/targets/apollo3evb/libam_hal.a \
    $(GCC_ARM)/lib/gcc/arm-none-eabi/7.3.1/thumb/v7e-m/fpv4-sp/softfp/crtbegin.o \
    -lm
  INCLUDES += \
    -isystem$(MAKEFILE_DIR)/downloads/cmsis/CMSIS/Core/Include/ \
    -I$(GCC_ARM)/arm-none-eabi/ \
    -I$(APOLLO3_SDK)/mcu/apollo3/ \
    -I$(APOLLO3_SDK)/CMSIS/AmbiqMicro/Include/ \
    -I$(APOLLO3_SDK)/boards/apollo3_evb/bsp/ \
    -I$(APOLLO3_SDK)/devices/ \
    -I$(APOLLO3_SDK)/utils/

  # The startup_gcc.c file is an altered version of the examples/hello_world/gcc/startup_gcc.c
  # file from Ambiq:
  #   - Increase the stack size from 1k to 20k
  #   - Change the application entry call from main() to _main()
  # The am_*.c files should be copied from the Ambiq Apollo3 SDK
  # _main.c contains application and target specific initialization, like
  # setting clock speed, default uart setups, etc. and an implementation
  # of the DebugLog interfaces.
  MICROLITE_CC_SRCS += \
    $(MAKEFILE_DIR)/targets/apollo3evb/startup_gcc.c \
    $(MAKEFILE_DIR)/targets/apollo3evb/_main.c \
    $(MAKEFILE_DIR)/targets/apollo3evb/am_util_delay.c \
    $(MAKEFILE_DIR)/targets/apollo3evb/am_util_faultisr.c \
    $(MAKEFILE_DIR)/targets/apollo3evb/am_util_id.c \
    $(MAKEFILE_DIR)/targets/apollo3evb/am_util_stdio.c

  TEST_SCRIPT := tensorflow/lite/experimental/log_test/test_apollo3evb_binary.sh
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
