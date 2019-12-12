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

#include "tensorflow/lite/micro/examples/image_recognition/disco/image_util.h"

void ResizeConvertImage(tflite::ErrorReporter* error_reporter,
                        int in_frame_width, int in_frame_height,
                        int num_in_channels, int out_frame_width,
                        int out_frame_height, int channels,
                        const uint8_t* in_image, uint8_t* out_image) {
  // offset so that only the center part of rectangular image is selected for
  // resizing
  int width_offset = ((in_frame_width - in_frame_height) / 2) * num_in_channels;

  int yresize_ratio = (in_frame_height / out_frame_height) * num_in_channels;
  int xresize_ratio = (in_frame_width / out_frame_width) * num_in_channels;
  int resize_ratio =
      (xresize_ratio < yresize_ratio) ? xresize_ratio : yresize_ratio;

  for (int y = 0; y < out_frame_height; y++) {
    for (int x = 0; x < out_frame_width; x++) {
      int orig_img_loc =
          y * in_frame_width * resize_ratio + x * resize_ratio + width_offset;
      // correcting the image inversion here
      int out_img_loc = ((out_frame_height - 1 - y) * out_frame_width +
                         (out_frame_width - 1 - x)) *
                        channels;
      uint8_t pix_lo = in_image[orig_img_loc];
      uint8_t pix_hi = in_image[orig_img_loc + 1];
      // convert RGB565 to RGB888
      out_image[out_img_loc] = (0xF8 & pix_hi);
      out_image[out_img_loc + 1] =
          ((0x07 & pix_hi) << 5) | ((0xE0 & pix_lo) >> 3);
      out_image[out_img_loc + 2] = (0x1F & pix_lo) << 3;
    }
  }
}
