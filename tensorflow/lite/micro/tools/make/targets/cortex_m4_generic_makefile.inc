# Generic Makefile target for ARM Cortex M4 builds.
# REQUIRED:
#   - TOOLCHAIN_PATH: The path to the ARM GCC toolchain to use.

ifeq ($(TARGET), cortex_m4_generic)
  TARGET_ARCH := arm
  TARGET_TOOLCHAIN_PREFIX := arm-none-eabi-
  export PATH := $(TOOLCHAIN_PATH):$(PATH)

  PLATFORM_FLAGS = \
    -DGEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK \
    -DTF_LITE_STATIC_MEMORY \
    -DNDEBUG \
    -DTF_LITE_MCU_DEBUG_LOG \
    -D __FPU_PRESENT=1 \
    -DARM_MATH_CM4 \
    -fno-rtti \
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
    -mfloat-abi=softfp \
    -std=gnu++11 \
    -Wvla \
    -Wall \
    -Wextra \
    -Wno-shadow \
    -Wno-missing-field-initializers \
    -Wno-strict-aliasing \
    -Wno-type-limits \
    -Wno-unused-function \
    -Wno-unused-parameter \
    -fno-delete-null-pointer-checks \
    -fno-threadsafe-statics \
    -fomit-frame-pointer \
    -fno-use-cxa-atexit \
    -O3

  CXXFLAGS += $(PLATFORM_FLAGS)
  CCFLAGS += $(PLATFORM_FLAGS)

  LDFLAGS += -Wl,--gc-sections

endif

