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

#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_IMAGE_RECOGNITION_EXPERIMENTAL_STM32F746_DISCOVERY_IMAGE_UTIL_H_
#define TENSORFLOW_LITE_MICRO_EXAMPLES_IMAGE_RECOGNITION_EXPERIMENTAL_STM32F746_DISCOVERY_IMAGE_UTIL_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

#define NUM_IN_CH 2
#define IN_IMG_WIDTH 160
#define IN_IMG_HEIGHT 120

void ResizeConvertImage(tflite::ErrorReporter* error_reporter,
                        int in_frame_width, int in_frame_height,
                        int num_in_channels, int out_frame_width,
                        int out_frame_height, int channels,
                        const uint8_t* in_frame, uint8_t* out_frame);

#endif  // TENSORFLOW_LITE_MICRO_EXAMPLES_IMAGE_RECOGNITION_EXPERIMENTAL_STM32F746_DISCOVERY_IMAGE_UTIL_H_
