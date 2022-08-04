// Copyright 2022 The TensorFlow Authors
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

#include "tensorflow/compiler/xla/tfrt_utils.h"

#include "tensorflow/core/platform/logging.h"

namespace xla {

PrimitiveType TfrtToPrimitiveType(tfrt::DType dtype) {
  switch (dtype) {
    // Unsigned integer types.
    case tfrt::DType::UI8:
      return PrimitiveType::U8;
    case tfrt::DType::UI16:
      return PrimitiveType::U16;
    case tfrt::DType::UI32:
      return PrimitiveType::U32;
    case tfrt::DType::UI64:
      return PrimitiveType::U64;

    // Signed integer types.
    case tfrt::DType::I1:
      return PrimitiveType::PRED;
    case tfrt::DType::I8:
      return PrimitiveType::S8;
    case tfrt::DType::I16:
      return PrimitiveType::S16;
    case tfrt::DType::I32:
      return PrimitiveType::S32;
    case tfrt::DType::I64:
      return PrimitiveType::S64;

    // Floating point types.
    case tfrt::DType::F16:
      return PrimitiveType::F16;
    case tfrt::DType::F32:
      return PrimitiveType::F32;
    case tfrt::DType::F64:
      return PrimitiveType::F64;
    case tfrt::DType::BF16:
      return PrimitiveType::BF16;

    // Complex types.
    case tfrt::DType::Complex64:
      return PrimitiveType::C64;
    case tfrt::DType::Complex128:
      return PrimitiveType::C128;

    default:
      LOG(FATAL) << "Unsupported data type: " << dtype;
  }
}

}  // namespace xla
