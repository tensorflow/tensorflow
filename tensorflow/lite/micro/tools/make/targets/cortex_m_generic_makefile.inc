# Generic Makefile target for ARM Cortex M builds.
# For more info see: tensorflow/lite/micro/cortex_m_generic/README.md

FLOAT := soft
GCC_TARGET_ARCH := $(TARGET_ARCH)

ifeq ($(TARGET_ARCH), cortex-m0)
  CORE=M0
  ARM_LDFLAGS := -Wl,--cpu=Cortex-M0

else ifeq ($(TARGET_ARCH), cortex-m3)
  CORE=M3
  ARM_LDFLAGS := -Wl,--cpu=Cortex-M3

else ifeq ($(TARGET_ARCH), cortex-m33)
  CORE=M33
  ARM_LDFLAGS := -Wl,--cpu=Cortex-M33
  TARGET_SPECIFIC_FLAGS += -D__DSP_PRESENT=1 -D__FPU_PRESENT=1 -D__VTOR_PRESENT=1 -D__FPU_USED=1
  FLOAT=hard

else ifeq ($(TARGET_ARCH), cortex-m33+nodsp)
  CORE=M33
  ARM_LDFLAGS := -Wl,--cpu=Cortex-M33.no_dsp.no_fp

else ifeq ($(TARGET_ARCH), cortex-m4)
  CORE=M4
  ARM_LDFLAGS := -Wl,--cpu=Cortex-M4.no_fp
  GCC_TARGET_ARCH := cortex-m4+nofp

else ifeq ($(TARGET_ARCH), cortex-m4+fp)
  CORE=M4
  ARM_LDFLAGS := -Wl,--cpu=Cortex-M4
  TARGET_SPECIFIC_FLAGS += -D__FPU_PRESENT=1
  FLOAT=hard
  GCC_TARGET_ARCH := cortex-m4

else ifeq ($(TARGET_ARCH), cortex-m55)
  CORE=M55
  ARM_LDFLAGS := -Wl,--cpu=8.1-M.Main.mve.fp
  TARGET_SPECIFIC_FLAGS += -D__DSP_PRESENT=1 -D__FPU_PRESENT=1
  FLOAT=hard

else ifeq ($(TARGET_ARCH), cortex-m55+nodsp+nofp)
  CORE=M55
  ARM_LDFLAGS := -Wl,--cpu=8.1-M.Main.mve.no_dsp.no_fp

else ifeq ($(TARGET_ARCH), cortex-m55+nofp)
  CORE=M55
  ARM_LDFLAGS := -Wl,--cpu=8.1-M.Main.mve.no_fp
  TARGET_SPECIFIC_FLAGS += -D__DSP_PRESENT=1

else ifeq ($(TARGET_ARCH), cortex-m7)
  CORE=M7
  ARM_LDFLAGS := -Wl,--cpu=Cortex-M7.no_fp
  GCC_TARGET_ARCH := cortex-m7+nofp

else ifeq ($(TARGET_ARCH), cortex-m7+fp)
  CORE=M7
  ARM_LDFLAGS := -Wl,--cpu=Cortex-M7
  FLOAT=hard
  GCC_TARGET_ARCH := cortex-m7

else
  $(error "TARGET_ARCH=$(TARGET_ARCH) is not supported")
endif

ifneq ($(filter cortex-m55%,$(TARGET_ARCH)),)
  # soft-abi=soft disables MVE - use softfp instead for M55.
  ifeq ($(FLOAT),soft)
    FLOAT=softfp
  endif
endif

# Toolchain specfic flags
ifeq ($(TOOLCHAIN), armclang)
  CXX_TOOL  := armclang
  CC_TOOL   := armclang
  AR_TOOL   := armar
  LD        := armlink

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

else ifeq ($(TOOLCHAIN), gcc)
  export PATH := $(MAKEFILE_DIR)/downloads/gcc_embedded/bin/:$(PATH)
  DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/arm_gcc_download.sh ${MAKEFILE_DIR}/downloads)
  ifneq ($(DOWNLOAD_RESULT), SUCCESS)
    $(error Something went wrong with the GCC download: $(DOWNLOAD_RESULT))
  endif

  TARGET_TOOLCHAIN_PREFIX := arm-none-eabi-

  FLAGS_GCC = -mcpu=$(GCC_TARGET_ARCH) -mfpu=auto
  CXXFLAGS += $(FLAGS_GCC)
  CCFLAGS += $(FLAGS_GCC)

else
  $(error "TOOLCHAIN=$(TOOLCHAIN) is not supported.")
endif

PLATFORM_FLAGS = \
  -DTF_LITE_MCU_DEBUG_LOG \
  -mthumb \
  -mfloat-abi=$(FLOAT) \
  -funsigned-char \
  -mlittle-endian \
  -Wno-type-limits \
  -Wno-unused-private-field \
  -fomit-frame-pointer \
  -MD \
  -DCPU_$(CORE)=1 \
  $(TARGET_SPECIFIC_FLAGS)

# Common + C/C++ flags
CXXFLAGS += $(PLATFORM_FLAGS)
CCFLAGS += $(PLATFORM_FLAGS)

# Needed for the project generation interface.
MICROLITE_CC_HDRS := \
  tensorflow/lite/micro/cortex_m_generic/debug_log_callback.h
