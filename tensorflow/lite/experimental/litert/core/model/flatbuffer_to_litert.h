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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_MODEL_FLATBUFFER_TO_LITERT_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_MODEL_FLATBUFFER_TO_LITERT_H_

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/core/util/flatbuffer_tools.h"

namespace litert::internal {

LiteRtStatus IsOpSupported(const TflOp& op);

LiteRtStatus IsBufferSupported(const TflBuffer& buffer);

// Checks if the misc non-type non quantization parts of this tensor are
// supported in the litet model api.
LiteRtStatus IsTensorSupported(const TflTensor& tensor);

LiteRtElementType MapElementType(TflElementType element_type);

Expected<TensorType> MapTensorType(const TflTensorType& tfl_tensor_type);

Expected<Quantization> MapQuantization(const TflQuantization* tfl_quantization,
                                       BufferProvider buffer_provider);

}  // namespace litert::internal

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_MODEL_FLATBUFFER_TO_LITERT_H_
