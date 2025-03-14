// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_TENSOR_BUFFER_CONVERSION_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_TENSOR_BUFFER_CONVERSION_H_

#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/runtime/tensor_buffer.h"

namespace litert::internal {

// Converts the given tensor buffer to the specified buffer type. A new tensor
// buffer is created and returned. This function locks/unlocks the tensor buffer
// and will involve a copy.
// TODO(b/383176413): Investigate zero/fast-copy conversions.
Expected<LiteRtTensorBufferT::Ptr> TensorBufferConvertTo(
    LiteRtTensorBufferType buffer_type, LiteRtTensorBufferT& tensor_buffer);

}  // namespace litert::internal

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_TENSOR_BUFFER_CONVERSION_H_
