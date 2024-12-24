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

#include "tensorflow/lite/experimental/litert/runtime/tfl_utils.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "tensorflow/lite/c/c_api_opaque.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/cc/litert_detail.h"
#include "tensorflow/lite/experimental/litert/cc/litert_element_type.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"

namespace litert {
namespace internal {

Expected<ElementType> ConvertElementType(TfLiteType tfl_type) {
  switch (tfl_type) {
    case kTfLiteNoType:
      return ElementType::None;
    case kTfLiteBool:
      return ElementType::Bool;
    case kTfLiteInt4:
      return ElementType::Int4;
    case kTfLiteInt8:
      return ElementType::Int8;
    case kTfLiteInt16:
      return ElementType::Int16;
    case kTfLiteInt32:
      return ElementType::Int32;
    case kTfLiteInt64:
      return ElementType::Int64;
    case kTfLiteUInt8:
      return ElementType::UInt8;
    case kTfLiteUInt16:
      return ElementType::UInt16;
    case kTfLiteUInt32:
      return ElementType::UInt32;
    case kTfLiteUInt64:
      return ElementType::UInt64;
    case kTfLiteFloat16:
      return ElementType::Float16;
    case kTfLiteBFloat16:
      return ElementType::BFloat16;
    case kTfLiteFloat32:
      return ElementType::Float32;
    case kTfLiteFloat64:
      return ElementType::Float64;
    case kTfLiteComplex64:
      return ElementType::Complex64;
    case kTfLiteComplex128:
      return ElementType::Complex128;
    case kTfLiteResource:
      return ElementType::TfResource;
    case kTfLiteString:
      return ElementType::TfString;
    case kTfLiteVariant:
      return ElementType::TfVariant;
    default:
      return Unexpected(kLiteRtStatusErrorInvalidArgument,
                        "Unsupported TfLiteType");
  }
}

Expected<RankedTensorType> ConvertTensorType(
    const TfLiteOpaqueTensor* tfl_opaque_tensor) {
  auto tfl_type = TfLiteOpaqueTensorType(tfl_opaque_tensor);
  auto element_type = ConvertElementType(tfl_type);
  if (!element_type) {
    return Unexpected(element_type.Error());
  }

  size_t rank = TfLiteOpaqueTensorNumDims(tfl_opaque_tensor);
  Dimensions dimensions(rank);
  for (size_t i = 0; i < rank; ++i) {
    dimensions[i] = TfLiteOpaqueTensorDim(tfl_opaque_tensor, i);
  }

  return RankedTensorType(*element_type, Layout(std::move(dimensions)));
}

}  // namespace internal
}  // namespace litert
