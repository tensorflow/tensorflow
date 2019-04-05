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
    -Wno-unused-parameter \
    -Wno-missing-field-initializers \
    -Wno-write-strings \
    -Wno-sign-compare \
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
    $(wildcard tensorflow/lite/experimental/micro/riscv32_mcu/*.cc)
  MICRO_SPEECH_TEST_SRCS += \
    $(wildcard $(MAKEFILE_DIR)/downloads/sifive_fe310_lib/bsp/libwrap/sys/*.c) \
    $(wildcard $(MAKEFILE_DIR)/downloads/sifive_fe310_lib/bsp/libwrap/sys/*.cc) \
    $(MAKEFILE_DIR)/downloads/sifive_fe310_lib/bsp/libwrap/misc/write_hex.c \
    $(MAKEFILE_DIR)/downloads/sifive_fe310_lib/bsp/libwrap/stdlib/malloc.c \
    $(MAKEFILE_DIR)/downloads/sifive_fe310_lib/bsp/env/start.S \
    $(MAKEFILE_DIR)/downloads/sifive_fe310_lib/bsp/env/entry.S \
    $(MAKEFILE_DIR)/downloads/sifive_fe310_lib/bsp/env/freedom-e300-hifive1/init.c
  LIBWRAP_SYMS := malloc free \
                  open lseek read write fstat stat close link unlink \
                  execve fork getpid kill wait \
                  isatty times sbrk _exit puts

  LDFLAGS += $(foreach s,$(LIBWRAP_SYMS),-Wl,--wrap=$(s))
  LDFLAGS += $(foreach s,$(LIBWRAP_SYMS),-Wl,--wrap=_$(s))
  LDFLAGS += -L. -Wl,--start-group -lc -Wl,--end-group
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