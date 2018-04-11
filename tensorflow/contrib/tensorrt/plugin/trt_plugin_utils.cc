/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/contrib/tensorrt/plugin/trt_plugin_utils.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT

namespace tensorflow {
namespace tensorrt {

size_t ExtractOpName(const void* serial_data, size_t serial_length,
                     size_t& incremental) {
  incremental = sizeof(size_t);
  if (serial_length < incremental) return 0;
  size_t encoded_op_name = *static_cast<const size_t*>(serial_data);
  return encoded_op_name;
}

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
#endif  // GOOGLE_TENSORRT
