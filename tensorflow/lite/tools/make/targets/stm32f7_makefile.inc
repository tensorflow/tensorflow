# Settings for STM32F7 platforms.
ifeq ($(TARGET), stm32f7)
  TARGET_ARCH := armf7
  TARGET_TOOLCHAIN_PREFIX := arm-none-eabi-

  CXXFLAGS += \
    -DGEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK \
    -DTFLITE_MCU \
    -fno-rtti \
    -fmessage-length=0 \
    -fno-exceptions \
    -fno-builtin \
    -ffunction-sections \
    -fdata-sections \
    -funsigned-char \
    -MMD \
    -mcpu=cortex-m7 \
    -mthumb \
    -mfpu=fpv5-sp-d16 \
    -mfloat-abi=softfp \
    -std=gnu++11 \
    -fno-rtti \
    -Wvla \
    -c \
    -Wall \
    -Wextra \
    -Wno-unused-parameter \
    -Wno-missing-field-initializers \
    -fmessage-length=0 \
    -fno-exceptions \
    -fno-builtin \
    -ffunction-sections \
    -fdata-sections \
    -funsigned-char \
    -MMD \
    -fno-delete-null-pointer-checks \
    -fomit-frame-pointer \
    -Os
 	LIBS += -ldl
	BUILD_TYPE := micro
endif
