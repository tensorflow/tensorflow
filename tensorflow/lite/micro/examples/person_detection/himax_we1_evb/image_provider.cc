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
#define ARDUINO_EXCLUDE_CODE
#endif  // defined(ARDUINO)

#ifndef ARDUINO_EXCLUDE_CODE

#include "tensorflow/lite/micro/examples/person_detection/image_provider.h"

#include "hx_drv_tflm.h"  // NOLINT
#include "tensorflow/lite/micro/examples/person_detection/model_settings.h"

hx_drv_sensor_image_config_t g_pimg_config;

TfLiteStatus GetImage(tflite::ErrorReporter* error_reporter, int image_width,
                      int image_height, int channels, int8_t* image_data) {
  static bool is_initialized = false;

  if (!is_initialized) {
    if (hx_drv_sensor_initial(&g_pimg_config) != HX_DRV_LIB_PASS) {
      return kTfLiteError;
    }
    is_initialized = true;
  }

  hx_drv_sensor_capture(&g_pimg_config);

  hx_drv_image_rescale((uint8_t*)g_pimg_config.raw_address,
                       g_pimg_config.img_width, g_pimg_config.img_height,
                       image_data, image_width, image_height);

  return kTfLiteOk;
}

#endif  // ARDUINO_EXCLUDE_CODE
