# Settings for cortex-m4f generic device, gcc build.
ifeq ($(TARGET),$(filter $(TARGET),\
  cortex-m4f-gcc-generic\
  ))
  export PATH := $(MAKEFILE_DIR)/downloads/gcc_embedded/bin/:$(PATH)
  TARGET_ARCH := cortex-m4
  TARGET_TOOLCHAIN_PREFIX := arm-none-eabi-
  # Need a pointer to the GNU ARM toolchain for crtbegin.o for the fp functions
  # with the hard interfaces.
  GCC_ARM := $(MAKEFILE_DIR)/downloads/gcc_embedded/

  $(eval $(call add_third_party_download,$(GCC_EMBEDDED_URL),$(GCC_EMBEDDED_MD5),gcc_embedded,))
  $(eval $(call add_third_party_download,$(CMSIS_URL),$(CMSIS_MD5),cmsis,patch_cmsis))

  # Use the faster depthwise conv implementation.
  ALL_TAGS += portable_optimized

  PLATFORM_FLAGS = \
    -DGEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK \
    -DTF_LITE_STATIC_MEMORY \
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
    -mfloat-abi=hard \
    -Wall \
    -Wextra \
    -Wno-shadow \
    -Wno-vla \
    -Wno-strict-aliasing \
    -Wno-type-limits \
    -Wno-unused-parameter \
    -Wno-missing-field-initializers \
    -Wno-write-strings \
    -Wno-sign-compare \
    -Wunused-function \
    -fno-delete-null-pointer-checks \
    -fomit-frame-pointer \
    -ggdb \
    -O3
  CXXFLAGS += $(PLATFORM_FLAGS) -std=gnu++11 -fno-rtti -fno-use-cxa-atexit
  CCFLAGS += $(PLATFORM_FLAGS)

  BUILD_TYPE := micro

  MICROLITE_LIBS := \
    -lm
  INCLUDES += \
    -isystem$(MAKEFILE_DIR)/downloads/cmsis/CMSIS/Core/Include/ \
    -isystem$(MAKEFILE_DIR)/downloads/cmsis/CMSIS/DSP/Include/ \
    -I$(GCC_ARM)/arm-none-eabi/ \

  CMSIS_SRC_DIR := $(MAKEFILE_DIR)/downloads/cmsis/CMSIS/DSP/Source
  THIRD_PARTY_CC_SRCS := \
  $(CMSIS_SRC_DIR)/BasicMathFunctions/arm_dot_prod_q15.c \
  $(CMSIS_SRC_DIR)/BasicMathFunctions/arm_mult_q15.c \
  $(CMSIS_SRC_DIR)/TransformFunctions/arm_rfft_init_q15.c \
  $(CMSIS_SRC_DIR)/TransformFunctions/arm_rfft_q15.c \
  $(CMSIS_SRC_DIR)/TransformFunctions/arm_bitreversal2.c \
  $(CMSIS_SRC_DIR)/TransformFunctions/arm_cfft_q15.c \
  $(CMSIS_SRC_DIR)/TransformFunctions/arm_cfft_radix4_q15.c \
  $(CMSIS_SRC_DIR)/CommonTables/arm_const_structs.c \
  $(CMSIS_SRC_DIR)/CommonTables/arm_common_tables.c \
  $(CMSIS_SRC_DIR)/StatisticsFunctions/arm_mean_q15.c \
  $(CMSIS_SRC_DIR)/StatisticsFunctions/arm_max_q7.c

  # These are tests that don't currently work on the generic cortex-m4f.
  EXCLUDED_TESTS :=
  MICROLITE_TEST_SRCS := $(filter-out $(EXCLUDED_TESTS), $(MICROLITE_TEST_SRCS))

endif
