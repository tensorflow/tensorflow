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

#ifndef TENSORFLOW_LITE_DELEGATES_UTILS_H_
#define TENSORFLOW_LITE_DELEGATES_UTILS_H_

#include <vector>

#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace delegates {

// Given a list(vector<int>) of indices, modifies it in-place to contain
// max_subsets number of continuous subsets. Subsets are selected in descending
// order of their length.
// Resulting vector contains sorted list of pruned indices.
//
// This util can be used by delegates to avoid accepting too many node-subsets.
TfLiteStatus PruneContinuousSubsets(TfLiteContext* context,
                                    const int max_subsets,
                                    std::vector<int>* indices);

}  // namespace delegates
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_UTILS_H_
