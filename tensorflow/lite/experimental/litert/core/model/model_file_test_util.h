// Copyright 2024 Google LLC.
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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_MODEL_MODEL_FILE_TEST_UTIL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_MODEL_MODEL_FILE_TEST_UTIL_H_

#include <functional>

#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/core/util/flatbuffer_tools.h"

namespace litert::internal {

// Callback to get a tfl tensor from it's index.
using GetTflTensor =
    std::function<std::reference_wrapper<const TflTensor>(uint32_t ind)>;

// Compare q-params for having the same type and values.
bool EqualsFbQuantization(const Quantization& litert_quantization,
                          const TflQuantization* tfl_quantization);

// Compare tensor types for having the same shape and element type.
bool EqualsFbTensorType(const TensorType& litert_tensor_type,
                        const TflTensorType& tfl_tensor_type);

// Compare litert op to flatbuffer op along with their input/output tensors
// types and quantization. Takes a callback to lookup tfl tensors the indices
// within the tfl op.
bool EqualsFbOp(const LiteRtOpT& litert_op, const TflOp& tfl_op,
                GetTflTensor get_tfl_tensor);

// Compare litert tensor to flatbuffer tensor for having same types and
// quantization.
bool EqualsFbTensor(const LiteRtTensorT& litert_tensor,
                    const TflTensor& tfl_tensor);

}  // namespace litert::internal

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_MODEL_MODEL_FILE_TEST_UTIL_H_
