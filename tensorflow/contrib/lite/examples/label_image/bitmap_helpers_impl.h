/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CONTRIB_LITE_EXAMPLES_LABEL_IMAGE_BITMAP_HELPERS_IMPL_H
#define TENSORFLOW_CONTRIB_LITE_EXAMPLES_LABEL_IMAGE_BITMAP_HELPERS_IMPL_H

#include "tensorflow/contrib/lite/examples/label_image/label_image.h"

namespace tflite {
namespace label_image {

template <class T>
void downsize(T* out, uint8_t* in, int image_height, int image_width,
              int image_channels, int wanted_height, int wanted_width,
              int wanted_channels, Settings* s) {
  for (int y = 0; y < wanted_height; ++y) {
    const int in_y = (y * image_height) / wanted_height;
    uint8_t* in_row = in + (in_y * image_width * image_channels);
    T* out_row = out + (y * wanted_width * wanted_channels);
    for (int x = 0; x < wanted_width; ++x) {
      const int in_x = (x * image_width) / wanted_width;
      uint8_t* in_pixel = in_row + (in_x * image_channels);
      T* out_pixel = out_row + (x * wanted_channels);
      for (int c = 0; c < wanted_channels; ++c) {
        if (s->input_floating)
          out_pixel[c] = (in_pixel[c] - s->input_mean) / s->input_std;
        else
          out_pixel[c] = in_pixel[c];
      }
    }
  }
}

}  // label_image
}  // tflite

#endif  // TENSORFLOW_CONTRIB_LITE_EXAMPLES_LABEL_IMAGE_BITMAP_HELPERS_IMPL_H
