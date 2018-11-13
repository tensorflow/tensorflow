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

#ifndef TENSORFLOW_CONTRIB_LITE_EXAMPLES_LABEL_IMAGE_GET_TOP_N_H_
#define TENSORFLOW_CONTRIB_LITE_EXAMPLES_LABEL_IMAGE_GET_TOP_N_H_

#include "tensorflow/contrib/lite/examples/label_image/get_top_n_impl.h"

namespace tflite {
namespace label_image {

template <class T>
void get_top_n(T* prediction, int prediction_size, size_t num_results,
               float threshold, std::vector<std::pair<float, int>>* top_results,
               bool input_floating);

// explicit instantiation so that we can use them otherwhere
template void get_top_n<uint8_t>(uint8_t*, int, size_t, float,
                                 std::vector<std::pair<float, int>>*, bool);
template void get_top_n<float>(float*, int, size_t, float,
                               std::vector<std::pair<float, int>>*, bool);

}  // namespace label_image
}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_EXAMPLES_LABEL_IMAGE_GET_TOP_N_H_
