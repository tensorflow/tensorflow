# Settings for RiscV platforms.
ifeq ($(TARGET), riscv)
  TARGET_ARCH := riscv
  TARGET_TOOLCHAIN_PREFIX := riscv32-unknown-elf-

  #CXXFLAGS += -march=gap8
  CXXFLAGS += -DTFLITE_MCU
	LIBS += -ldl
	BUILD_TYPE := micro
endif

ifeq ($(TARGET), linux_riscv64)
  TARGET_ARCH := riscv64
  TARGET_TOOLCHAIN_PREFIX := riscv64-unknown-linux-gnu-
  TARGET_OUT_DIR := linux_riscv64
  LIBS := -lstdc++ -lpthread -lm -ldl -latomic
  LDFLAGS := \
        -Wl,--no-export-dynamic \
        -Wl,--exclude-libs,ALL \
        -Wl,--gc-sections \
        -Wl,--as-needed \
        -lrt
endif
