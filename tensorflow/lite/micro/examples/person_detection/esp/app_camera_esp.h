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

#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_PERSON_DETECTION_ESP_APP_CAMERA_ESP_H_
#define TENSORFLOW_LITE_MICRO_EXAMPLES_PERSON_DETECTION_ESP_APP_CAMERA_ESP_H_

#include "esp_camera.h"
#include "esp_log.h"
#include "esp_system.h"

/**
 * PIXFORMAT_RGB565,    // 2BPP/RGB565
 * PIXFORMAT_YUV422,    // 2BPP/YUV422
 * PIXFORMAT_GRAYSCALE, // 1BPP/GRAYSCALE
 * PIXFORMAT_JPEG,      // JPEG/COMPRESSED
 * PIXFORMAT_RGB888,    // 3BPP/RGB888
 */
#define CAMERA_PIXEL_FORMAT PIXFORMAT_GRAYSCALE

/*
 * FRAMESIZE_96X96,    // 96x96
 * FRAMESIZE_QQVGA,    // 160x120
 * FRAMESIZE_QQVGA2,   // 128x160
 * FRAMESIZE_QCIF,     // 176x144
 * FRAMESIZE_HQVGA,    // 240x176
 * FRAMESIZE_QVGA,     // 320x240
 * FRAMESIZE_CIF,      // 400x296
 * FRAMESIZE_VGA,      // 640x480
 * FRAMESIZE_SVGA,     // 800x600
 * FRAMESIZE_XGA,      // 1024x768
 * FRAMESIZE_SXGA,     // 1280x1024
 * FRAMESIZE_UXGA,     // 1600x1200
 */
#define CAMERA_FRAME_SIZE FRAMESIZE_96X96

#if CONFIG_CAMERA_MODEL_WROVER_KIT
#define PWDN_GPIO_NUM -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 21
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27

#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 19
#define Y4_GPIO_NUM 18
#define Y3_GPIO_NUM 5
#define Y2_GPIO_NUM 4
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22

#elif CONFIG_CAMERA_MODEL_ESP_EYE
#define PWDN_GPIO_NUM -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 4
#define SIOD_GPIO_NUM 18
#define SIOC_GPIO_NUM 23

#define Y9_GPIO_NUM 36
#define Y8_GPIO_NUM 37
#define Y7_GPIO_NUM 38
#define Y6_GPIO_NUM 39
#define Y5_GPIO_NUM 35
#define Y4_GPIO_NUM 14
#define Y3_GPIO_NUM 13
#define Y2_GPIO_NUM 34
#define VSYNC_GPIO_NUM 5
#define HREF_GPIO_NUM 27
#define PCLK_GPIO_NUM 25

#elif CONFIG_CAMERA_MODEL_M5STACK_PSRAM
#define PWDN_GPIO_NUM -1
#define RESET_GPIO_NUM 15
#define XCLK_GPIO_NUM 27
#define SIOD_GPIO_NUM 25
#define SIOC_GPIO_NUM 23

#define Y9_GPIO_NUM 19
#define Y8_GPIO_NUM 36
#define Y7_GPIO_NUM 18
#define Y6_GPIO_NUM 39
#define Y5_GPIO_NUM 5
#define Y4_GPIO_NUM 34
#define Y3_GPIO_NUM 35
#define Y2_GPIO_NUM 32
#define VSYNC_GPIO_NUM 22
#define HREF_GPIO_NUM 26
#define PCLK_GPIO_NUM 21

#elif CONFIG_CAMERA_MODEL_M5STACK_WIDE
#define PWDN_GPIO_NUM -1
#define RESET_GPIO_NUM 15
#define XCLK_GPIO_NUM 27
#define SIOD_GPIO_NUM 22
#define SIOC_GPIO_NUM 23

#define Y9_GPIO_NUM 19
#define Y8_GPIO_NUM 36
#define Y7_GPIO_NUM 18
#define Y6_GPIO_NUM 39
#define Y5_GPIO_NUM 5
#define Y4_GPIO_NUM 34
#define Y3_GPIO_NUM 35
#define Y2_GPIO_NUM 32
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 26
#define PCLK_GPIO_NUM 21

#elif CONFIG_CAMERA_MODEL_AI_THINKER
#define PWDN_GPIO_NUM 32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 0
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27

#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 21
#define Y4_GPIO_NUM 19
#define Y3_GPIO_NUM 18
#define Y2_GPIO_NUM 5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22

#elif CONFIG_CAMERA_MODEL_CUSTOM
#define PWDN_GPIO_NUM CONFIG_CAMERA_PIN_PWDN
#define RESET_GPIO_NUM CONFIG_CAMERA_PIN_RESET
#define XCLK_GPIO_NUM CONFIG_CAMERA_PIN_XCLK
#define SIOD_GPIO_NUM CONFIG_CAMERA_PIN_SIOD
#define SIOC_GPIO_NUM CONFIG_CAMERA_PIN_SIOC

#define Y9_GPIO_NUM CONFIG_CAMERA_PIN_Y9
#define Y8_GPIO_NUM CONFIG_CAMERA_PIN_Y8
#define Y7_GPIO_NUM CONFIG_CAMERA_PIN_Y7
#define Y6_GPIO_NUM CONFIG_CAMERA_PIN_Y6
#define Y5_GPIO_NUM CONFIG_CAMERA_PIN_Y5
#define Y4_GPIO_NUM CONFIG_CAMERA_PIN_Y4
#define Y3_GPIO_NUM CONFIG_CAMERA_PIN_Y3
#define Y2_GPIO_NUM CONFIG_CAMERA_PIN_Y2
#define VSYNC_GPIO_NUM CONFIG_CAMERA_PIN_VSYNC
#define HREF_GPIO_NUM CONFIG_CAMERA_PIN_HREF
#define PCLK_GPIO_NUM CONFIG_CAMERA_PIN_PCLK

#else
#error "No camera module configured, please configure in menuconfig"
#endif

#define XCLK_FREQ 20000000

#ifdef __cplusplus
extern "C" {
#endif

int app_camera_init();

#ifdef __cplusplus
}
#endif
#endif  // TENSORFLOW_LITE_MICRO_EXAMPLES_PERSON_DETECTION_ESP_APP_CAMERA_ESP_H_
