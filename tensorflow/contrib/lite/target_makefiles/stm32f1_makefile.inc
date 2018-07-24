# Settings for STM32F1 platforms.
ifeq ($(CROSS), stm32f1)
  TARGET_ARCH := armm1
  TARGET_TOOLCHAIN_PREFIX := arm-none-eabi-

  CXXFLAGS += \
  -DGEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK \
  -mcpu=cortex-m1 \
  -mthumb \
  -DTFLITE_MCU \
  -fno-rtti \
  -fmessage-length=0 \
  -fno-exceptions \
  -fno-builtin \
  -ffunction-sections \
  -fdata-sections \
  -funsigned-char \
  -MMD
	LIBS += -ldl
	BUILD_TYPE := micro
endif
