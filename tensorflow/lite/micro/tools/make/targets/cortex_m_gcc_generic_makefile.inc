# Generic Makefile target for ARM Cortex Mx gcc builds.
ifeq ($(TARGET), cortex_m_gcc_generic)
  TARGET_ARCH := arm
  TARGET_TOOLCHAIN_PREFIX := arm-none-eabi-
  export PATH := $(MAKEFILE_DIR)/downloads/gcc_embedded/bin/:$(PATH)

  $(eval $(call add_third_party_download,$(GCC_EMBEDDED_URL),$(GCC_EMBEDDED_MD5),gcc_embedded,))

  PLATFORM_FLAGS = \
    -DTF_LITE_MCU_DEBUG_LOG \
    -funsigned-char \
    -mcpu=cortex-m4 \
    -mfpu=fpv4-sp-d16 \
    -mthumb \
    -fomit-frame-pointer

ifeq ($(CORTEX_M_CORE), M4F)
  PLATFORM_FLAGS += -mfloat-abi=hard
else ifeq ($(CORTEX_M_CORE), M4)
  PLATFORM_FLAGS += -mfloat-abi=softfp
else ifeq ($(CORTEX_M_CORE), )
  $(error CORTEX_M_CORE=[M4|M4F] not defined on the command line)
else
  $(error invalid target defined in command line option CORTEX_M_CORE=[M4|M4F])
endif

  CXXFLAGS += $(PLATFORM_FLAGS)
  CCFLAGS += $(PLATFORM_FLAGS)

endif
