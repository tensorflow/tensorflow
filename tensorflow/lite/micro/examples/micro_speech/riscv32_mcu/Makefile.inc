ifeq ($(TARGET), riscv32_mcu)
  # Wrap functions
  MICRO_FE310_LIBWRAP_SRCS := \
    $(wildcard $(MAKEFILE_DIR)/downloads/sifive_fe310_lib/bsp/libwrap/sys/*.c) \
    $(MAKEFILE_DIR)/downloads/sifive_fe310_lib/bsp/libwrap/misc/write_hex.c \
    $(MAKEFILE_DIR)/downloads/sifive_fe310_lib/bsp/libwrap/stdlib/malloc.c

  MICRO_FE310_BSP_ENV_SRCS := \
    $(MAKEFILE_DIR)/downloads/sifive_fe310_lib/bsp/env/start.S \
    $(MAKEFILE_DIR)/downloads/sifive_fe310_lib/bsp/env/entry.S \
    $(MAKEFILE_DIR)/downloads/sifive_fe310_lib/bsp/env/freedom-e300-hifive1/init.c

  MICRO_SPEECH_TEST_SRCS += $(MICRO_FE310_LIBWRAP_SRCS) $(MICRO_FE310_BSP_ENV_SRCS) \
    tensorflow/lite/micro/arduino/abi.cc
  MICRO_SPEECH_SRCS += $(MICRO_FE310_LIBWRAP_SRCS) $(MICRO_FE310_BSP_ENV_SRCS) \
    tensorflow/lite/micro/arduino/abi.cc

  LIBWRAP_SYMS := malloc free \
                  open lseek read write fstat stat close link unlink \
                  execve fork getpid kill wait \
                  isatty times sbrk _exit puts

  LDFLAGS += $(foreach s,$(LIBWRAP_SYMS),-Wl,--wrap=$(s))
  LDFLAGS += $(foreach s,$(LIBWRAP_SYMS),-Wl,--wrap=_$(s))
  LDFLAGS += -L. -Wl,--start-group -lc -Wl,--end-group
endif
