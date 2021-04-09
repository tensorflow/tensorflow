/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/platform/tensor_float_32_utils.h"

#include <atomic>

namespace tensorflow {

// Whether TensorFloat-32 should be used where supported.
static std::atomic<bool> tensor_float_32_enabled{true};

void enable_tensor_float_32_execution(bool enabled) {
  tensor_float_32_enabled = enabled;
}

bool tensor_float_32_execution_enabled() { return tensor_float_32_enabled; }

}  // namespace tensorflow
