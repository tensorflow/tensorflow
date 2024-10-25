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

#include "tensorflow/lite/experimental/lrt/core/utils.h"

#include <cstddef>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tensorflow/lite/experimental/lrt/c/litert_model.h"

namespace litert {
namespace internal {

absl::StatusOr<Ratio> GetElementSize(LiteRtElementType element_type) {
  switch (element_type) {
    case kLiteRtElementTypeInt4:
      return Ratio{1, 2};
    case kLiteRtElementTypeBool:
      return Ratio{1, 1};
    case kLiteRtElementTypeInt8:
    case kLiteRtElementTypeUInt8:
      return Ratio{1, 1};
    case kLiteRtElementTypeInt16:
    case kLiteRtElementTypeUInt16:
    case kLiteRtElementTypeFloat16:
    case kLiteRtElementTypeBFloat16:
      return Ratio{2, 1};
    case kLiteRtElementTypeInt32:
    case kLiteRtElementTypeUInt32:
    case kLiteRtElementTypeFloat32:
      return Ratio{4, 1};
    case kLiteRtElementTypeInt64:
    case kLiteRtElementTypeUInt64:
    case kLiteRtElementTypeFloat64:
      return Ratio{8, 1};
    case kLiteRtElementTypeComplex64:
      return Ratio{16, 1};
    case kLiteRtElementTypeComplex128:
      return Ratio{32, 1};
    default:
      return absl::InvalidArgumentError("Unexpected element type");
  }
}

absl::StatusOr<size_t> GetNumPackedBytes(const LiteRtRankedTensorType& type) {
  auto element_size = GetElementSize(type.element_type);
  if (!element_size.ok()) {
    return element_size.status();
  }

  auto num_elements = GetNumElements(type);
  if (!num_elements.ok()) {
    return num_elements.status();
  }

  return ((*num_elements * element_size->num) + (element_size->denom - 1)) /
         element_size->denom;
}

absl::StatusOr<size_t> GetNumElements(
    const LiteRtRankedTensorType& tensor_type) {
  size_t num_elements = 1;
  for (auto i = 0; i < tensor_type.layout.rank; ++i) {
    auto dim = tensor_type.layout.dimensions[i];
    if (dim < 0) {
      return absl::InvalidArgumentError(
          "Unexpected dynamic tensor passed as input");
    } else if (dim == 0) {
      return absl::InvalidArgumentError("Unexpected 0 tensor dimension");
    }
    num_elements *= dim;
  }

  return num_elements;
}

}  // namespace internal
}  // namespace litert
