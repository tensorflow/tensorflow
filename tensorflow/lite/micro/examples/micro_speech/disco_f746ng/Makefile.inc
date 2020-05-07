# Settings for the Discovery STM32F746NG board.
ifneq ($(filter disco_f746ng,$(ALL_TAGS)),)
  micro_speech_MBED_PROJECT_FILES += \
    AUDIO_DISCO_F746NG.lib \
    BSP_DISCO_F746NG.lib \
    SDRAM_DISCO_F746NG.lib \
    LCD_DISCO_F746NG.lib
endif
