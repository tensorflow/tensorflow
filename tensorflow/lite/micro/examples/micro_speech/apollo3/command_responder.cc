/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if defined(ARDUINO)
#if defined(ARDUINO_SFE_EDGE)

#else
#define ARDUINO_EXCLUDE_CODE
#endif  // defined(ARDUINO_SFE_EDGE)
#endif  // defined(ARDUINO)

#ifndef ARDUINO_EXCLUDE_CODE

#include "tensorflow/lite/micro/examples/micro_speech/command_responder.h"

#include "am_bsp.h"  // NOLINT

#if defined(TFLU_APOLLO3_BOARD_apollo3evb)
void apollo3evb_init(void);
#endif // defined(TFLU_APOLLO3_BOARD_apollo3evb)

#if defined(TFLU_APOLLO3_BOARD_sparkfun_edge)
void sparkfun_edge_init(void);
#endif // defined(TFLU_APOLLO3_BOARD_apollo3evb)

// This implementation will light up the LEDs on the board in response to
// different commands.
void RespondToCommand(tflite::ErrorReporter* error_reporter,
                      int32_t current_time, const char* found_command,
                      uint8_t score, bool is_new_command) {
  static bool is_initialized = false;
  if (!is_initialized) {
#if defined(TFLU_APOLLO3_BOARD_apollo3evb)
    apollo3evb_init();
#endif // defined(TFLU_APOLLO3_BOARD_apollo3evb)
#if defined(TFLU_APOLLO3_BOARD_sparkfun_edge)
    sparkfun_edge_init();
#endif // defined(TFLU_APOLLO3_BOARD_sparkfun_edge)
    is_initialized = true;
  }

  if (is_new_command) {
    TF_LITE_REPORT_ERROR(error_reporter, "Heard %s (%d) @%dms", found_command,
                         score, current_time);
  }

#if defined(TFLU_APOLLO3_BOARD_apollo3evb)
#if USE_MAYA
  if (g_PowerOff) {
    power_down_sequence();
  }
#endif  // USE_MAYA
#endif  // defined(TFLU_APOLLO3_BOARD_apollo3evb)

#if defined(TFLU_APOLLO3_BOARD_sparkfun_edge)
  // Toggle the blue LED every time an inference is performed.
  am_devices_led_toggle(am_bsp_psLEDs, AM_BSP_LED_BLUE);

  // Turn on LEDs corresponding to the detection for the cycle
  am_devices_led_off(am_bsp_psLEDs, AM_BSP_LED_RED);
  am_devices_led_off(am_bsp_psLEDs, AM_BSP_LED_YELLOW);
  am_devices_led_off(am_bsp_psLEDs, AM_BSP_LED_GREEN);
  if (is_new_command) {    
    if (found_command[0] == 'y') {
      am_devices_led_on(am_bsp_psLEDs, AM_BSP_LED_YELLOW);
    }
    if (found_command[0] == 'n') {
      am_devices_led_on(am_bsp_psLEDs, AM_BSP_LED_RED);
    }
    if (found_command[0] == 'u') {
      am_devices_led_on(am_bsp_psLEDs, AM_BSP_LED_GREEN);
    }
  }
#endif // defined(TFLU_APOLLO3_BOARD_sparkfun_edge)
}

#if defined(TFLU_APOLLO3_BOARD_apollo3evb)
#if USE_MAYA
static uint32_t g_PowerOff = 0;
#endif // USE_MAYA

void apollo3evb_init(void) {
  // ARPIT TODO : Implement low power configuration
#if USE_MAYA
  // Make sure SWO/ITM/TPIU is disabled.
  // SBL may not get it completely shut down.
  am_bsp_itm_printf_disable();
#else
  // Initialize the printf interface for AP3B ITM/SWO output.
  am_bsp_itm_printf_enable();
#endif

#ifdef AM_BSP_NUM_LEDS
  //
  // Initialize the LEDs.
  // On the apollo3_evb, when the GPIO outputs are disabled (the default at
  // power up), the FET gates are floating and
  // partially illuminating the LEDs.
  //
  uint32_t ux, ui32GPIONumber;
  for (ux = 0; ux < AM_BSP_NUM_LEDS; ux++) {
    ui32GPIONumber = am_bsp_psLEDs[ux].ui32GPIONumber;

    am_hal_gpio_pinconfig(ui32GPIONumber, g_AM_HAL_GPIO_OUTPUT);
    am_hal_gpio_state_write(ui32GPIONumber, AM_HAL_GPIO_OUTPUT_TRISTATE_DISABLE);
    am_hal_gpio_state_write(ui32GPIONumber, AM_HAL_GPIO_OUTPUT_CLEAR);
  }
#endif  // AM_BSP_NUM_LEDS

  // Configure the LEDs.
  am_devices_led_array_init(am_bsp_psLEDs, AM_BSP_NUM_LEDS);
  // Turn the LEDs off
  for (int ix = 0; ix < AM_BSP_NUM_LEDS; ix++) {
    am_devices_led_off(am_bsp_psLEDs, ix);
  }

#if USE_MAYA
  // Configure Power Button
  am_hal_gpio_pinconfig(AM_BSP_GPIO_BUTTON_POWER, g_AM_BSP_GPIO_BUTTON_POWER);

  // Clear and Enable the GPIO Interrupt (write to clear).
  am_hal_gpio_interrupt_clear(AM_HAL_GPIO_BIT(AM_BSP_GPIO_BUTTON_POWER));
  am_hal_gpio_interrupt_register(AM_BSP_GPIO_BUTTON_POWER,
                                 power_button_handler);
  am_hal_gpio_interrupt_enable(AM_HAL_GPIO_BIT(AM_BSP_GPIO_BUTTON_POWER));

  // Enable GPIO interrupts to the NVIC.
  NVIC_EnableIRQ(GPIO_IRQn);
#endif  // USE_MAYA

#if USE_DEBUG_GPIO
  // DEBUG : GPIO flag polling.
  // Configure the GPIOs for flag polling.
  am_hal_gpio_pinconfig(31, g_AM_HAL_GPIO_OUTPUT);  // Slot1 AN pin
  am_hal_gpio_pinconfig(39, g_AM_HAL_GPIO_OUTPUT);  // Slot1 RST pin
  am_hal_gpio_pinconfig(44, g_AM_HAL_GPIO_OUTPUT);  // Slot1 CS pin
  am_hal_gpio_pinconfig(48, g_AM_HAL_GPIO_OUTPUT);  // Slot1 PWM pin

  am_hal_gpio_pinconfig(32, g_AM_HAL_GPIO_OUTPUT);  // Slot2 AN pin
  am_hal_gpio_pinconfig(46, g_AM_HAL_GPIO_OUTPUT);  // Slot2 RST pin
  am_hal_gpio_pinconfig(42, g_AM_HAL_GPIO_OUTPUT);  // Slot2 CS pin
  am_hal_gpio_pinconfig(47, g_AM_HAL_GPIO_OUTPUT);  // Slot2 PWM pin
#endif
}

#if USE_MAYA
extern "C" void power_down_sequence(void) {
  am_hal_gpio_read_type_e eReadType;
  eReadType = AM_HAL_GPIO_INPUT_READ;

  // Reconfigure PDM Pins for low power
  // Drive PDMCLK low so Mics go in standby mode of ~ 10 to 20uA each
  am_hal_gpio_pinconfig(12, g_AM_HAL_GPIO_OUTPUT);
  am_hal_gpio_state_write(12, AM_HAL_GPIO_OUTPUT_SET);

  // Disable PDMDATA pin so no input buffer leakage current from floating pin
  am_hal_gpio_pinconfig(11, g_AM_HAL_GPIO_DISABLE);

  // Disable PDM
  am_hal_pdm_disable(g_pdm_handle);
  am_hal_pdm_power_control(g_pdm_handle, AM_HAL_PDM_POWER_OFF, false);
  am_hal_interrupt_master_disable();

  am_hal_gpio_interrupt_clear(AM_HAL_GPIO_BIT(AM_BSP_GPIO_BUTTON0));
  am_hal_gpio_interrupt_disable(AM_HAL_GPIO_BIT(AM_BSP_GPIO_BUTTON0));
  am_util_delay_ms(200);  // Debounce Delay
  am_hal_gpio_interrupt_clear(AM_HAL_GPIO_BIT(AM_BSP_GPIO_BUTTON0));
  am_hal_gpio_interrupt_enable(AM_HAL_GPIO_BIT(AM_BSP_GPIO_BUTTON0));

  for (int ix = 0; ix < AM_BSP_NUM_LEDS; ix++) {
    am_devices_led_off(am_bsp_psLEDs, ix);
  }

  am_hal_sysctrl_sleep(AM_HAL_SYSCTRL_SLEEP_DEEP);
  // Apollo3 will be < 3uA in deep sleep

  am_hal_reset_control(AM_HAL_RESET_CONTROL_SWPOR, 0);
  // Use Reset to perform clean power-on from sleep
}

//*****************************************************************************
//
// GPIO ISR
//
//*****************************************************************************
extern "C" void am_gpio_isr(void) {
  uint64_t ui64Status;
  // Read and clear the GPIO interrupt status then service the interrupts.
  am_hal_gpio_interrupt_status_get(false, &ui64Status);
  am_hal_gpio_interrupt_clear(ui64Status);
  am_hal_gpio_interrupt_service(ui64Status);
}

extern "C" void power_button_handler(void) { g_PowerOff = 1; }

#endif // USE_MAYA
#endif // defined(TFLU_APOLLO3_BOARD_apollo3evb)

#if defined(TFLU_APOLLO3_BOARD_sparkfun_edge)
void sparkfun_edge_init(void) {
#ifdef AM_BSP_NUM_LEDS
  // Setup LED's as outputs
  am_devices_led_array_init(am_bsp_psLEDs, AM_BSP_NUM_LEDS);
  am_devices_led_array_out(am_bsp_psLEDs, AM_BSP_NUM_LEDS, 0x00000000);
#endif
}
#endif // defined(TFLU_APOLLO3_BOARD_sparkfun_edge)

#endif  // ARDUINO_EXCLUDE_CODE
