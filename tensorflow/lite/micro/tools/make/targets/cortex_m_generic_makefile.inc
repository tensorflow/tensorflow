# Generic Makefile target for ARM Cortex M builds.
# For more info see: tensorflow/lite/micro/cortex_m_generic/README.md

FLOAT := soft
ifeq ($(TARGET_ARCH),$(filter $(TARGET_ARCH),cortex-m55))
  CORE=M55
  ARM_LDFLAGS := -Wl,--cpu=8.1-M.Main.mve.fp
  TARGET_SPECIFIC_FLAGS += -D__DSP_PRESENT=1 -D__FPU_PRESENT=1
  FLOAT=hard
  TARGET_ARCH := cortex-m55
  GCC_TARGET_ARCH := $(TARGET_ARCH)
else ifeq ($(TARGET_ARCH),$(filter $(TARGET_ARCH),cortex-m55+nodsp+nofp))
  CORE=M55
  ARM_LDFLAGS := -Wl,--cpu=8.1-M.Main.mve.no_dsp.no_fp
  TARGET_SPECIFIC_FLAGS +=
  TARGET_ARCH := cortex-m55+nodsp+nofp
  GCC_TARGET_ARCH := $(TARGET_ARCH)
else ifeq ($(TARGET_ARCH),$(filter $(TARGET_ARCH),cortex-m55+nofp))
  CORE=M55
  ARM_LDFLAGS := -Wl,--cpu=8.1-M.Main.mve.no_fp
  TARGET_SPECIFIC_FLAGS += -D__DSP_PRESENT=1
  TARGET_ARCH := cortex-m55+nofp
  GCC_TARGET_ARCH := $(TARGET_ARCH)
else ifeq ($(TARGET_ARCH),$(filter $(TARGET_ARCH),cortex-m33+nodsp))
  CORE=M33
  ARM_LDFLAGS := -Wl,--cpu=Cortex-M33.no_dsp.no_fp
  TARGET_SPECIFIC_FLAGS +=
  TARGET_ARCH := cortex-m33+nodsp
  GCC_TARGET_ARCH := $(TARGET_ARCH)
else ifeq ($(TARGET_ARCH),$(filter $(TARGET_ARCH),cortex-m33))
  CORE=M33
  ARM_LDFLAGS := -Wl,--cpu=Cortex-M33
  TARGET_SPECIFIC_FLAGS += -D__DSP_PRESENT=1 -D__FPU_PRESENT=1 -D__VTOR_PRESENT=1 -D__FPU_USED=1
  FLOAT=hard
  TARGET_ARCH := cortex-m33
  GCC_TARGET_ARCH := $(TARGET_ARCH)
else ifeq ($(TARGET_ARCH),$(filter $(TARGET_ARCH),cortex-m0))
  CORE=M0
  ARM_LDFLAGS := -Wl,--cpu=Cortex-M0
  TARGET_SPECIFIC_FLAGS +=
  TARGET_ARCH := cortex-m0
  GCC_TARGET_ARCH := $(TARGET_ARCH)
else ifeq ($(TARGET_ARCH),$(filter $(TARGET_ARCH),cortex-m3))
  CORE=M3
  ARM_LDFLAGS := -Wl,--cpu=Cortex-M3
  TARGET_SPECIFIC_FLAGS +=
  TARGET_ARCH := cortex-m3
  GCC_TARGET_ARCH := $(TARGET_ARCH)
else ifeq ($(TARGET_ARCH),$(filter $(TARGET_ARCH),cortex-m4))
  CORE=M4
  ARM_LDFLAGS := -Wl,--cpu=Cortex-M4.no_fp
  TARGET_SPECIFIC_FLAGS+=
  TARGET_ARCH := cortex-m4
  GCC_TARGET_ARCH := $(TARGET_ARCH)
else ifeq ($(TARGET_ARCH),$(filter $(TARGET_ARCH),cortex-m4+fp))
  CORE=M4
  ARM_LDFLAGS := -Wl,--cpu=Cortex-M4
  TARGET_SPECIFIC_FLAGS+=
  FLOAT=hard
  TARGET_ARCH := cortex-m4+fp
  GCC_TARGET_ARCH := cortex-m4
else ifeq ($(TARGET_ARCH),$(filter $(TARGET_ARCH),cortex-m7+fp))
  CORE=M7
  ARM_LDFLAGS := -Wl,--cpu=Cortex-M7
  TARGET_SPECIFIC_FLAGS +=
  FLOAT=hard
  TARGET_ARCH := cortex-m7+fp
  GCC_TARGET_ARCH :=  cortex-m7
else ifeq ($(TARGET_ARCH),$(filter $(TARGET_ARCH),cortex-m7))
  CORE=M7
  ARM_LDFLAGS := -Wl,--cpu=Cortex-M7.no_fp
  TARGET_SPECIFIC_FLAGS +=
  TARGET_ARCH := cortex-m7
  GCC_TARGET_ARCH := $(TARGET_ARCH)
else
  $(error "$(TARGET) not supported")
endif

# soft-abi=soft disables MVE - use softfp instead for M55.
ifneq ($(filter cortex-m55%,$(TARGET_ARCH)),)
  ifeq ($(FLOAT),soft)
    FLOAT=softfp
  endif
endif

# Toolchain specfic flags
ifeq ($(TOOLCHAIN), armclang))
  CXX_TOOL  := armclang
  CC_TOOL   := armclang
  AR_TOOL   := armar
  LD    := armlink
  FLAGS_ARMC = \
    --target=arm-arm-none-eabi \
    -mcpu=$(TARGET_ARCH)

  CXXFLAGS += $(FLAGS_ARMC)
  CCFLAGS += $(FLAGS_ARMC)
  LDFLAGS += $(ARM_LDFLAGS)

  # Arm Compiler will not link the Math library (see below), therefore we're filtering it out.
  # See Fatal error: L6450U: Cannot find library m:
  # "Arm Compiler is designed to run in a bare metal environment,
  # and automatically includes implementations of these functions,
  # and so no such flag is necessary."
  # https://developer.arm.com/documentation/100891/0611/troubleshooting/general-troubleshooting-advice
  MICROLITE_LIBS := $(filter-out -lm,$(MICROLITE_LIBS))
else ifeq ($(TOOLCHAIN), armgcc))
  ifeq ($(shell which arm-none-eabi-gcc),)
    export PATH := $(MAKEFILE_DIR)/downloads/gcc_embedded/bin/:$(PATH)
    $(eval $(call add_third_party_download,$(GCC_EMBEDDED_URL),$(GCC_EMBEDDED_MD5),gcc_embedded,))
  endif

  TARGET_TOOLCHAIN_PREFIX := arm-none-eabi-
  
  ifneq ($(filter cortex-m55%,$(TARGET_ARCH)),)
    $(error Micro architecure support is not available yet for $(GCC_TARGET_ARCH))
  else
    FLAGS_GCC = -mcpu=$(GCC_TARGET_ARCH)
  endif
  CXXFLAGS += $(FLAGS_GCC)
  CCFLAGS += $(FLAGS_GCC)
  LDFLAGS +=
endif

PLATFORM_FLAGS = \
  -mthumb \
  -mfloat-abi=$(FLOAT) \
  -funsigned-char \
  -mlittle-endian \
  -Wno-type-limits \
  -Wno-unused-private-field \
  -MD \
  -DCPU_$(CORE)=1 \
  $(TARGET_SPECIFIC_FLAGS)

ifneq ($(BUILD_TYPE), release)
  PLATFORM_FLAGS += -DTF_LITE_MCU_DEBUG_LOG
endif

# Common + C/C++ flags
CXXFLAGS += $(PLATFORM_FLAGS)
CCFLAGS += $(PLATFORM_FLAGS)

TEST_SCRIPT :=

