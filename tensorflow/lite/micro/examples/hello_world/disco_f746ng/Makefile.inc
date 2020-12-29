# Settings for the Discovery STM32F746NG board.
ifneq ($(filter disco_f746ng,$(ALL_TAGS)),)
  hello_world_MBED_PROJECT_FILES += \
    BSP_DISCO_F746NG.lib \
    LCD_DISCO_F746NG.lib
endif
