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

#include "tensorflow/lite/experimental/micro/examples/person_detection/detection_responder.h"

#include "Arduino.h"

// Pins for the built-in RGB LEDs on the Arduino Nano 33 BLE Sense
const int led_red = 22;
const int led_green = 23;
const int led_blue = 24;

// Flash the blue LED after each inference
void RespondToDetection(tflite::ErrorReporter* error_reporter,
                        uint8_t person_score, uint8_t no_person_score) {
  static bool is_initialized = false;
  if (!is_initialized) {
    pinMode(led_green, OUTPUT);
    pinMode(led_blue, OUTPUT);
    is_initialized = true;
  }

  // Note: The RGB LEDs on the Arduino Nano 33 BLE
  // Sense are on when the pin is LOW, off when HIGH.

  // Switch the person/not person LEDs off
  digitalWrite(led_green, HIGH);
  digitalWrite(led_red, HIGH);

  // Flash the blue LED after every inference.
  digitalWrite(led_blue, LOW);
  delay(100);
  digitalWrite(led_blue, HIGH);

  // Switch on the green LED when a person is detected,
  // the red when no person is detected
  if (person_score > no_person_score) {
    digitalWrite(led_green, LOW);
    digitalWrite(led_red, HIGH);
  } else {
    digitalWrite(led_green, HIGH);
    digitalWrite(led_red, LOW);
  }

  error_reporter->Report("Person score: %d No person score: %d", person_score,
                         no_person_score);
}
