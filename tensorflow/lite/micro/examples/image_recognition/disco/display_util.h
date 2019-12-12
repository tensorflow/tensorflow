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

#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_RECOGNITION_DISCO_DISPLAY_UTIL_H_
#define TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_RECOGNITION_DISCO_DISPLAY_UTIL_H_

#include <stdint.h>

void init_lcd();

void display_image_rgb888(int x_dim, int y_dim, const uint8_t* image_data,
                          int x_loc, int y_loc);

void display_image_rgb565(int x_dim, int y_dim, const uint8_t* image_data,
                          int x_loc, int y_loc);

void print_prediction(const char* prediction);

void print_confidence(uint8_t max_score);

#endif  // TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_RECOGNITION_DISCO_DISPLAY_UTIL_H_
