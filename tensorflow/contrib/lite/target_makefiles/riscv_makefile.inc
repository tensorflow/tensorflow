# Settings for RiscV platforms.
ifeq ($(CROSS), riscv)
  TARGET_ARCH := riscv
  TARGET_TOOLCHAIN_PREFIX := riscv32-unknown-elf-

  #CXXFLAGS += -march=gap8
  CXXFLAGS += -DTFLITE_MCU
	LIBS += -ldl
	BUILD_TYPE := micro
endif
