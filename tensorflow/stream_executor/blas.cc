/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/stream_executor/blas.h"

#include "absl/strings/str_cat.h"

namespace stream_executor {
namespace blas {

std::string TransposeString(Transpose t) {
  switch (t) {
    case Transpose::kNoTranspose:
      return "NoTranspose";
    case Transpose::kTranspose:
      return "Transpose";
    case Transpose::kConjugateTranspose:
      return "ConjugateTranspose";
    default:
      LOG(FATAL) << "Unknown transpose " << static_cast<int32>(t);
  }
}

std::string UpperLowerString(UpperLower ul) {
  switch (ul) {
    case UpperLower::kUpper:
      return "Upper";
    case UpperLower::kLower:
      return "Lower";
    default:
      LOG(FATAL) << "Unknown upperlower " << static_cast<int32>(ul);
  }
}

std::string DiagonalString(Diagonal d) {
  switch (d) {
    case Diagonal::kUnit:
      return "Unit";
    case Diagonal::kNonUnit:
      return "NonUnit";
    default:
      LOG(FATAL) << "Unknown diagonal " << static_cast<int32>(d);
  }
}

std::string SideString(Side s) {
  switch (s) {
    case Side::kLeft:
      return "Left";
    case Side::kRight:
      return "Right";
    default:
      LOG(FATAL) << "Unknown side " << static_cast<int32>(s);
  }
}

// -- AlgorithmConfig

std::string AlgorithmConfig::ToString() const {
  return absl::StrCat(algorithm_);
}

std::string ComputationTypeString(ComputationType ty) {
  switch (ty) {
    case ComputationType::kF16:
      return "f16";
    case ComputationType::kF32:
      return "f32";
    case ComputationType::kF64:
      return "f64";
    case ComputationType::kI32:
      return "i32";
    case ComputationType::kComplexF32:
      return "complex f32";
    case ComputationType::kComplexF64:
      return "complex f64";
    default:
      LOG(FATAL) << "Unknown ComputationType " << static_cast<int32>(ty);
  }
}

std::ostream& operator<<(std::ostream& os, ComputationType ty) {
  return os << ComputationTypeString(ty);
}

std::string DataTypeString(DataType ty) {
  switch (ty) {
    case DataType::kBF16:
      return "bf16";
    case DataType::kHalf:
      return "f16";
    case DataType::kFloat:
      return "f32";
    case DataType::kDouble:
      return "f64";
    case DataType::kInt8:
      return "i8";
    case DataType::kInt32:
      return "i32";
    case DataType::kComplexFloat:
      return "complex f32";
    case DataType::kComplexDouble:
      return "complex f64";
    default:
      LOG(FATAL) << "Unknown DataType " << static_cast<int32>(ty);
  }
}

std::ostream& operator<<(std::ostream& os, DataType ty) {
  return os << DataTypeString(ty);
}

}  // namespace blas
}  // namespace stream_executor
