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

#ifndef TENSORFLOW_CONTRIB_LITE_EXAMPLES_LABEL_IMAGE_LABEL_IMAGE_H_
#define TENSORFLOW_CONTRIB_LITE_EXAMPLES_LABEL_IMAGE_LABEL_IMAGE_H_

namespace tflite {
namespace label_image {

uint8_t* read_bmp(const std::string& input_bmp_name, int& width, int& height,
                  int& channels);

template <class T>
void downsize(T* out, uint8_t* in, int image_height, int image_width,
              int image_channels, int wanted_height, int wanted_width,
              int wanted_channels);

template <class T>
void get_top_n(T* prediction, const int prediction_size,
               const size_t num_results, const float threshold,
               std::vector<std::pair<float, int>>* top_results);

}  // label_image
}  // tflite

#endif  // TENSORFLOW_CONTRIB_LITE_EXAMPLES_LABEL_IMAGE_LABEL_IMAGE_H
