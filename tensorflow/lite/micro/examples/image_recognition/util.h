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

#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_IMAGE_RECOGNITION_UTIL_H_
#define TENSORFLOW_LITE_MICRO_EXAMPLES_IMAGE_RECOGNITION_UTIL_H_

#include <stdint.h>
#include <string.h>

#define IMAGE_SIZE 3072
#define CHANNEL_SIZE 1024
#define R_CHANNEL_OFFSET 0
#define G_CHANNEL_OFFSET CHANNEL_SIZE
#define B_CHANNEL_OFFSET (CHANNEL_SIZE * 2)

int get_top_prediction(const uint8_t* predictions, int num_categories) {
  int max_score = predictions[0];
  int guess = 0;

  for (int category_index = 1; category_index < num_categories;
       category_index++) {
    const uint8_t category_score = predictions[category_index];
    if (category_score > max_score) {
      max_score = category_score;
      guess = category_index;
    }
  }

  return guess;
}

void reshape_cifar_image(uint8_t* image_data, int num_bytes) {
  uint8_t temp_data[IMAGE_SIZE];

  memcpy(temp_data, image_data, num_bytes);

  int k = 0;
  for (int i = 0; i < CHANNEL_SIZE; i++) {
    int r_ind = R_CHANNEL_OFFSET + i;
    int g_ind = G_CHANNEL_OFFSET + i;
    int b_ind = B_CHANNEL_OFFSET + i;

    image_data[k] = temp_data[r_ind];
    k++;
    image_data[k] = temp_data[g_ind];
    k++;
    image_data[k] = temp_data[b_ind];
    k++;
  }
}

#endif  // TENSORFLOW_LITE_MICRO_EXAMPLES_IMAGE_RECOGNITION_UTIL_H_
