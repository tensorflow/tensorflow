/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_TFRT_KERNELS_STREAM_OPS_UTIL_CONSTANTS_H_
#define TENSORFLOW_CORE_TFRT_KERNELS_STREAM_OPS_UTIL_CONSTANTS_H_

#include <cstddef>

namespace tensorflow {
namespace tfrt_stub {

// Step id and batch id are packed together to a 64 bit integer in the stream
// callback. Step id takes the MSB 32 bit.
inline constexpr size_t kStepIdBitSize = 32;

}  // namespace tfrt_stub
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_KERNELS_STREAM_OPS_UTIL_CONSTANTS_H_
