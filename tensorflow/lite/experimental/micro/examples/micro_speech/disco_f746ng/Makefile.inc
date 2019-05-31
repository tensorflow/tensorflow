# Settings for the Discovery STM32F746NG board.
ifneq ($(filter disco_f746ng,$(ALL_TAGS)),)
  MBED_PROJECT_FILES += \
    AUDIO_DISCO_F746NG.lib \
    BSP_DISCO_F746NG.lib \
    SDRAM_DISCO_F746NG.lib
endif
