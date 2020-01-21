# Settings for RISCV 32-bit MCU toolchain.
ifeq ($(TARGET), riscv32_mcu)
  TARGET_ARCH := riscv32_mcu
  TARGET_TOOLCHAIN_PREFIX := riscv64-unknown-elf-

  $(eval $(call add_third_party_download,$(RISCV_TOOLCHAIN_URL),$(RISCV_TOOLCHAIN_MD5),riscv_toolchain,))
  $(eval $(call add_third_party_download,$(SIFIVE_FE310_LIB_URL),$(SIFIVE_FE310_LIB_MD5),sifive_fe310_lib,))

  PLATFORM_FLAGS = \
    -march=rv32imac \
    -mabi=ilp32 \
    -mcmodel=medany \
    -mexplicit-relocs \
    -fno-builtin-printf \
    -fno-exceptions \
    -DTF_LITE_MCU_DEBUG_LOG \
    -DTF_LITE_USE_GLOBAL_ROUND \
    -fno-unwind-tables \
    -fno-builtin \
    -ffunction-sections \
    -fdata-sections \
    -funsigned-char \
    -Wvla \
    -Wall \
    -Wextra \
    -Wsign-compare \
    -Wno-unused-parameter \
    -Wno-missing-field-initializers \
    -Wno-write-strings \
    -fno-delete-null-pointer-checks \
    -fomit-frame-pointer \
    -Os

  CXXFLAGS += $(PLATFORM_FLAGS) \
    -fpermissive \
    -fno-rtti \
    --std=gnu++11

  CCFLAGS += $(PLATFORM_FLAGS)

  BUILD_TYPE := micro

  INCLUDES += \
    -I$(MAKEFILE_DIR)/downloads/sifive_fe310_lib/bsp/include \
    -I$(MAKEFILE_DIR)/downloads/sifive_fe310_lib/bsp/drivers/ \
    -I$(MAKEFILE_DIR)/downloads/sifive_fe310_lib/bsp/env \
    -I$(MAKEFILE_DIR)/downloads/sifive_fe310_lib/bsp/env/freedom-e300-hifive1

  MICROLITE_CC_SRCS += \
    $(wildcard tensorflow/lite/micro/riscv32_mcu/*.cc)

  LDFLAGS += \
   -T$(MAKEFILE_DIR)/downloads/sifive_fe310_lib/bsp/env/freedom-e300-hifive1/flash.lds \
   -nostartfiles \
   -L$(MAKEFILE_DIR)/downloads/sifive_fe310_lib/bsp/env \
   --specs=nano.specs

# These are microcontroller-specific rules for converting the ELF output
# of the linker into a binary image that can be loaded directly.
  OBJCOPY := $(TARGET_TOOLCHAIN_PREFIX)objcopy
  $(BINDIR)/%.bin: $(BINDIR)/%
		@mkdir -p $(dir $@)
		$(OBJCOPY) $< $@ -O binary
endif
