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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICRO_EXAMPLES_PERSON_DETECTION_HIMAX_DRIVER_PLATFORM_SPARKFUN_EDGE_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICRO_EXAMPLES_PERSON_DETECTION_HIMAX_DRIVER_PLATFORM_SPARKFUN_EDGE_H_

#ifdef __cplusplus
extern "C" {
#endif

#define HM01B0_PIN_D0 24
#define HM01B0_PIN_D1 25
#define HM01B0_PIN_D2 26
#define HM01B0_PIN_D3 27
#define HM01B0_PIN_D4 28
#define HM01B0_PIN_D5 5
#define HM01B0_PIN_D6 6
#define HM01B0_PIN_D7 7
#define HM01B0_PIN_VSYNC 15
#define HM01B0_PIN_HSYNC 22
#define HM01B0_PIN_PCLK 23
#define HM01B0_PIN_TRIG 12
#define HM01B0_PIN_INT 4
#define HM01B0_PIN_SCL 8
#define HM01B0_PIN_SDA 9
#define HM01B0_PIN_DVDD_EN 10

// Define AP3B's CTIMER and output pin for HM01B0 MCLK generation
#define HM01B0_MCLK_GENERATOR_MOD 0
#define HM01B0_MCLK_GENERATOR_SEG AM_HAL_CTIMER_TIMERB
#define HM01B0_PIN_MCLK 13

// Deifne I2C controller and SCL(pin8)/SDA(pin9) are configured automatically.
#define HM01B0_IOM_MODE AM_HAL_IOM_I2C_MODE
#define HM01B0_IOM_MODULE 1
#define HM01B0_I2C_CLOCK_FREQ AM_HAL_IOM_100KHZ

#ifdef __cplusplus
}
#endif

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICRO_EXAMPLES_PERSON_DETECTION_HIMAX_DRIVER_PLATFORM_SPARKFUN_EDGE_H_
