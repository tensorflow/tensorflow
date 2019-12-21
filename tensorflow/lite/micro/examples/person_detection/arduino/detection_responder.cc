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

#ifdef ARDUINO_ARDUINO_NANO33BLE

#include "tensorflow/lite/micro/examples/person_detection/detection_responder.h"

#include "Arduino.h"

// Flash the blue LED after each inference
void RespondToDetection(tflite::ErrorReporter* error_reporter,
                        uint8_t person_score, uint8_t no_person_score) {
  static bool is_initialized = false;
  if (!is_initialized) {
    // Pins for the built-in RGB LEDs on the Arduino Nano 33 BLE Sense
    pinMode(LEDR, OUTPUT);
    pinMode(LEDG, OUTPUT);
    pinMode(LEDB, OUTPUT);
    is_initialized = true;
  }

  // Note: The RGB LEDs on the Arduino Nano 33 BLE
  // Sense are on when the pin is LOW, off when HIGH.

  // Switch the person/not person LEDs off
  digitalWrite(LEDG, HIGH);
  digitalWrite(LEDR, HIGH);

  // Flash the blue LED after every inference.
  digitalWrite(LEDB, LOW);
  delay(100);
  digitalWrite(LEDB, HIGH);

  // Switch on the green LED when a person is detected,
  // the red when no person is detected
  if (person_score > no_person_score) {
    digitalWrite(LEDG, LOW);
    digitalWrite(LEDR, HIGH);
  } else {
    digitalWrite(LEDG, HIGH);
    digitalWrite(LEDR, LOW);
  }

  error_reporter->Report("Person score: %d No person score: %d", person_score,
                         no_person_score);
}

#endif // ARDUINO_ARDUINO_NANO33BLE



#ifdef ARDUINO_SFE_EDGE

#include "tensorflow/lite/micro/examples/person_detection/detection_responder.h"

#include "am_bsp.h"  // NOLINT

// This implementation will light up LEDs on the board in response to the
// inference results.
void RespondToDetection(tflite::ErrorReporter* error_reporter,
                        uint8_t person_score, uint8_t no_person_score) {
  static bool is_initialized = false;
  if (!is_initialized) {
    // Setup LED's as outputs.  Leave red LED alone since that's an error
    // indicator for sparkfun_edge in image_provider.
    am_devices_led_init((am_bsp_psLEDs + AM_BSP_LED_BLUE));
    am_devices_led_init((am_bsp_psLEDs + AM_BSP_LED_GREEN));
    am_devices_led_init((am_bsp_psLEDs + AM_BSP_LED_YELLOW));
    is_initialized = true;
  }

  // Toggle the blue LED every time an inference is performed.
  am_devices_led_toggle(am_bsp_psLEDs, AM_BSP_LED_BLUE);

  // Turn on the green LED if a person was detected.  Turn on the yellow LED
  // otherwise.
  am_devices_led_off(am_bsp_psLEDs, AM_BSP_LED_YELLOW);
  am_devices_led_off(am_bsp_psLEDs, AM_BSP_LED_GREEN);
  if (person_score > no_person_score) {
    am_devices_led_on(am_bsp_psLEDs, AM_BSP_LED_GREEN);
  } else {
    am_devices_led_on(am_bsp_psLEDs, AM_BSP_LED_YELLOW);
  }

  error_reporter->Report("Person score: %d No person score: %d", person_score,
                         no_person_score);
}

#endif // ARDUINO_SFE_EDGE