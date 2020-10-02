# Generic Makefile target for ARM Cortex M builds.
# REQUIRED:
#   - TAGS: armclang or armgcc
# For Cortex-M55, ARM Compiler 6.14 or later is required.
# Example:
# make -f tensorflow/lite/micro/tools/make/Makefile TAGS="cmsis-nn armclang"  TARGET=cortex-m55-generic microlite

ifneq ($(filter cortex-%-generic,$(TARGET)),)

  # Pick up cortex-m-generic debug log instead of default.
  ALL_TAGS := $(filter-out $(TARGET),$(ALL_TAGS))
  ALL_TAGS += cortex-m-generic

  FLOAT := soft

  ifeq ($(TARGET),$(filter $(TARGET),cortex-m55-generic))
    CORE=M55
    ARM_LDFLAGS := -Wl,--cpu=8.1-M.Main.mve.fp
    TARGET_SPECIFIC_FLAGS += -D__DSP_PRESENT=1 -D__FPU_PRESENT=1
    FLOAT=hard
    TARGET_ARCH := cortex-m55
  else ifeq ($(TARGET),$(filter $(TARGET),cortex-m55+nodsp+nofp-generic))
    CORE=M55
    ARM_LDFLAGS := -Wl,--cpu=8.1-M.Main.mve.no_dsp.no_fp
    TARGET_SPECIFIC_FLAGS +=
    TARGET_ARCH := cortex-m55+nodsp+nofp
  else ifeq ($(TARGET),$(filter $(TARGET),cortex-m55+nofp-generic))
    CORE=M55
    ARM_LDFLAGS := -Wl,--cpu=8.1-M.Main.mve.no_fp
    TARGET_SPECIFIC_FLAGS += -D__DSP_PRESENT=1
    TARGET_ARCH := cortex-m55+nofp
  else ifeq ($(TARGET),$(filter $(TARGET),cortex-m33+nodsp-generic))
    CORE=M33
    ARM_LDFLAGS := -Wl,--cpu=Cortex-M33.no_dsp.no_fp
    TARGET_SPECIFIC_FLAGS +=
    TARGET_ARCH := cortex-m33+nodsp
  else ifeq ($(TARGET),$(filter $(TARGET),cortex-m33-generic))
    CORE=M33
    ARM_LDFLAGS := -Wl,--cpu=Cortex-M33
    TARGET_SPECIFIC_FLAGS += -D__DSP_PRESENT=1 -D__FPU_PRESENT=1 -D__VTOR_PRESENT=1 -D__FPU_USED=1
    FLOAT=hard
    TARGET_ARCH := cortex-m33
  else ifeq ($(TARGET),$(filter $(TARGET),cortex-m0-generic))
    CORE=M0
    ARM_LDFLAGS := -Wl,--cpu=Cortex-M0
    TARGET_SPECIFIC_FLAGS +=
    TARGET_ARCH := cortex-m0
  else ifeq ($(TARGET),$(filter $(TARGET),cortex-m3-generic))
    CORE=M3
    ARM_LDFLAGS := -Wl,--cpu=Cortex-M3
    TARGET_SPECIFIC_FLAGS +=
    TARGET_ARCH := cortex-m3
  else ifeq ($(TARGET),$(filter $(TARGET),cortex-m4-generic))
    CORE=M4
    ARM_LDFLAGS := -Wl,--cpu=Cortex-M4.no_fp
    TARGET_SPECIFIC_FLAGS+=
    TARGET_ARCH := cortex-m4
  else ifeq ($(TARGET),$(filter $(TARGET),cortex-m7+fp-generic))
    CORE=M7
    ARM_LDFLAGS := -Wl,--cpu=Cortex-M7.fp
    TARGET_SPECIFIC_FLAGS +=
    FLOAT=hard
    TARGET_ARCH := cortex-m7+fp
  else ifeq ($(TARGET),$(filter $(TARGET),cortex-m7-generic))
    CORE=M7
    ARM_LDFLAGS := -Wl,--cpu=Cortex-M7.no_fp
    TARGET_SPECIFIC_FLAGS +=
    TARGET_ARCH := cortex-m7
  else
    $(error "$(TARGET) not supported")
  endif

  # Toolchain specfic flags
  ifeq ($(filter armclang,$(ALL_TAGS)),armclang)
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
  else ifeq ($(filter armgcc,$(ALL_TAGS)),armgcc)
    CXX_TOOL  := arm-none-eabi-gcc
    CC_TOOL   := arm-none-eabi-gcc
    AR_TOOL   := arm-none-eabi-gcc-ar
    LD        := arm-none-eabi-ld
    ifneq ($(filter cortex-m55%,$(TARGET_ARCH)),)
      $(error Micro architecure support is not available yet for $(TARGET_ARCH))
    else
      FLAGS_GCC = -mcpu=$(TARGET_ARCH)
    endif
    CXXFLAGS += $(FLAGS_GCC)
    CCFLAGS += $(FLAGS_GCC)
    LDFLAGS += -Wl,--gc-sections
  endif

  PLATFORM_FLAGS = \
    -DTF_LITE_STATIC_MEMORY \
    -mthumb \
    -mfloat-abi=$(FLOAT) \
    -funsigned-char \
    -mlittle-endian \
    -fno-function-sections \
    -MD \
    -DCPU_$(CORE)=1 \
    $(TARGET_SPECIFIC_FLAGS)

  # Common + C/C++ flags
  CXXFLAGS += $(PLATFORM_FLAGS) -fno-rtti
  CCFLAGS += $(PLATFORM_FLAGS)

  TEST_SCRIPT :=

endif
