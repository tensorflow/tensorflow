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

#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_PERSON_DETECTION_HIMAX_DRIVER_PLATFORM_SPARKFUN_EDGE_H_
#define TENSORFLOW_LITE_MICRO_EXAMPLES_PERSON_DETECTION_HIMAX_DRIVER_PLATFORM_SPARKFUN_EDGE_H_

#ifdef __cplusplus
extern "C" {
#endif

#define HM01B0_PIN_D0                   AM_BSP_GPIO_CAMERA_HM01B0_D0
#define HM01B0_PIN_D1                   AM_BSP_GPIO_CAMERA_HM01B0_D1
#define HM01B0_PIN_D2                   AM_BSP_GPIO_CAMERA_HM01B0_D2
#define HM01B0_PIN_D3                   AM_BSP_GPIO_CAMERA_HM01B0_D3
#define HM01B0_PIN_D4                   AM_BSP_GPIO_CAMERA_HM01B0_D4
#define HM01B0_PIN_D5                   AM_BSP_GPIO_CAMERA_HM01B0_D5
#define HM01B0_PIN_D6                   AM_BSP_GPIO_CAMERA_HM01B0_D6
#define HM01B0_PIN_D7                   AM_BSP_GPIO_CAMERA_HM01B0_D7
#define HM01B0_PIN_VSYNC                AM_BSP_GPIO_CAMERA_HM01B0_VSYNC
#define HM01B0_PIN_HSYNC                AM_BSP_GPIO_CAMERA_HM01B0_HSYNC
#define HM01B0_PIN_PCLK                 AM_BSP_GPIO_CAMERA_HM01B0_PCLK
#define HM01B0_PIN_SCL                  AM_BSP_CAMERA_HM01B0_I2C_SCL_PIN
#define HM01B0_PIN_SDA                  AM_BSP_CAMERA_HM01B0_I2C_SDA_PIN


// Some boards do not support TRIG or INT pins
#ifdef AM_BSP_GPIO_CAMERA_HM01B0_TRIG
#define HM01B0_PIN_TRIG                 AM_BSP_GPIO_CAMERA_HM01B0_TRIG
#endif // AM_BSP_GPIO_CAMERA_HM01B0_TRIG

#ifdef AM_BSP_GPIO_CAMERA_HM01B0_INT
#define HM01B0_PIN_INT                  AM_BSP_GPIO_CAMERA_HM01B0_INT
#endif // AM_BSP_GPIO_CAMERA_HM01B0_INT


// Define AP3B's CTIMER and output pin for HM01B0 MCLK generation
#define HM01B0_MCLK_GENERATOR_MOD       AM_BSP_CAMERA_HM01B0_MCLK_GEN_MOD
#define HM01B0_MCLK_GENERATOR_SEG       AM_BSP_CAMERA_HM01B0_MCLK_GEN_SEG
#define HM01B0_PIN_MCLK                 AM_BSP_CAMERA_HM01B0_MCLK_PIN

// Deifne I2C controller and SCL(pin8)/SDA(pin9) are configured automatically.
#define HM01B0_IOM_MODE                 AM_HAL_IOM_I2C_MODE
#define HM01B0_IOM_MODULE               AM_BSP_CAMERA_HM01B0_I2C_IOM
#define HM01B0_I2C_CLOCK_FREQ           AM_HAL_IOM_100KHZ


#ifdef __cplusplus
}
#endif

#endif  // TENSORFLOW_LITE_MICRO_EXAMPLES_PERSON_DETECTION_HIMAX_DRIVER_PLATFORM_SPARKFUN_EDGE_H_
