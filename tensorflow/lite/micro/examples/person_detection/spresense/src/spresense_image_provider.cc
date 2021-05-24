/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

// The SPRESENSE_CONFIG_H is defined on compiler option.
// It contains "nuttx/config.h" from Spresense SDK to see the configurated
// parameters.
#include SPRESENSE_CONFIG_H
#include "spresense_image_provider.h"

#include "tensorflow/lite/micro/examples/person_detection/image_provider.h"
#include "tensorflow/lite/micro/examples/person_detection/model_settings.h"

TfLiteStatus GetImage(tflite::ErrorReporter* error_reporter, int image_width,
                      int image_height, int channels, int8_t* image_data) {
  if (spresense_getimage((unsigned char*)image_data) == 0) {
    return kTfLiteOk;
  } else {
    return kTfLiteError;
  }
}
