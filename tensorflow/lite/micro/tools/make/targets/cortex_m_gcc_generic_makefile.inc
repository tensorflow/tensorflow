# Generic Makefile target for ARM Cortex Mx gcc builds.
ifeq ($(TARGET),$(filter $(TARGET),\
  cortex_m_gcc_generic\
  ))
  TARGET_ARCH := arm
  TARGET_TOOLCHAIN_PREFIX := arm-none-eabi-
  export PATH := $(MAKEFILE_DIR)/downloads/gcc_embedded/bin/:$(PATH)

  $(eval $(call add_third_party_download,$(GCC_EMBEDDED_URL),$(GCC_EMBEDDED_MD5),gcc_embedded,))

  PLATFORM_FLAGS = \
    -DGEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK \
    -DTF_LITE_STATIC_MEMORY \
    -DNDEBUG \
    -DTF_LITE_MCU_DEBUG_LOG \
    -D __FPU_PRESENT=1 \
    -DARM_MATH_CM4 \
    -fmessage-length=0 \
    -fno-exceptions \
    -fno-unwind-tables \
    -ffunction-sections \
    -fdata-sections \
    -funsigned-char \
    -MMD \
    -mcpu=cortex-m4 \
    -mthumb \
    -mfpu=fpv4-sp-d16 \
    -Wvla \
    -Wall \
    -Wextra \
    -Wno-shadow \
    -Wno-missing-field-initializers \
    -Wno-double-promotion \
    -Wno-strict-aliasing \
    -Wno-type-limits \
    -Wno-unused-function \
    -Wno-unused-variable \
    -Wno-unused-parameter \
    -Wno-write-strings \
    -Wno-sign-compare \
    -fno-delete-null-pointer-checks \
    -fomit-frame-pointer \
    -nostdlib \
    -ggdb \
    -O3

ifdef CORTEX_M_CORE
  ifeq ($(CORTEX_M_CORE), M4F)
    PLATFORM_FLAGS += -mfloat-abi=hard
  else ifeq ($(CORTEX_M_CORE), M4)
    PLATFORM_FLAGS += -mfloat-abi=softfp
  else
    $(error invalid target defined in command line option CORTEX_M_CORE=[M4|M4F])
  endif
else
  $(error CORTEX_M_CORE=[M4|M4F] not defined on the command line)
endif

  CXXFLAGS += $(PLATFORM_FLAGS) -std=gnu++11 -fno-rtti -fpermissive -fno-use-cxa-atexit -fno-threadsafe-statics
  CCFLAGS += $(PLATFORM_FLAGS)
  CXXFLAGS := $(CXXFLAGS:-DTF_LITE_STATIC_MEMORY=)
  CCFLAGS := $(CCFLAGS:-DTF_LITE_STATIC_MEMORY=)
  
  INCLUDES += \
    -isystem$(MAKEFILE_DIR)/downloads/cmsis/CMSIS/Core/Include/ \
    -isystem$(MAKEFILE_DIR)/downloads/cmsis/CMSIS/DSP/Include/

endif