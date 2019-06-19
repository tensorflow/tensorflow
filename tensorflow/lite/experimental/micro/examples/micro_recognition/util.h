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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICRO_EXAMPLES_MICRO_RECOGNITION_UTIL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICRO_EXAMPLES_MICRO_RECOGNITION_UTIL_H_

#include <stdint.h>

int get_top_prediction(uint8_t* predictions) {
  int max_score = 0;
  int guess = 0;

  for (int category_index = 0; category_index < 10; category_index++) {
    const uint8_t category_score = predictions[category_index];
    if (category_score > max_score) {
      max_score = category_score;
      guess = category_index;
    }
  }

  return guess;
}

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICRO_EXAMPLES_MICRO_RECOGNITION_UTIL_H_
