/* Copyright 2015 The OpenXLA Authors.

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

#include "xla/stream_executor/blas.h"

#include <cstdint>
#include <optional>
#include <ostream>
#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/stream_executor/blas.pb.h"
#include "xla/stream_executor/device_memory.h"

namespace stream_executor {
namespace blas {

// TODO(ezhulenev): We need a scoped thread local map-like container to make
// sure that we can have multiple BlasSupport instances that do not overwrite
// each others workspaces. For not it's ok as we know that this can't happen.
static thread_local DeviceMemoryBase* workspace_thread_local = nullptr;

BlasSupport::ScopedWorkspace::ScopedWorkspace(BlasSupport* blas,
                                              DeviceMemoryBase* workspace)
    : blas_(blas) {
  blas->SetWorkspace(workspace);
}

BlasSupport::ScopedWorkspace::~ScopedWorkspace() { blas_->ResetWorkspace(); }

DeviceMemoryBase* BlasSupport::GetWorkspace() { return workspace_thread_local; }

void BlasSupport::SetWorkspace(DeviceMemoryBase* workspace) {
  workspace_thread_local = workspace;
}

void BlasSupport::ResetWorkspace() { workspace_thread_local = nullptr; }

std::string TransposeString(Transpose t) {
  switch (t) {
    case Transpose::kNoTranspose:
      return "NoTranspose";
    case Transpose::kTranspose:
      return "Transpose";
    case Transpose::kConjugateTranspose:
      return "ConjugateTranspose";
    default:
      LOG(FATAL) << "Unknown transpose " << static_cast<int32_t>(t);
  }
}

std::string UpperLowerString(UpperLower ul) {
  switch (ul) {
    case UpperLower::kUpper:
      return "Upper";
    case UpperLower::kLower:
      return "Lower";
    default:
      LOG(FATAL) << "Unknown upperlower " << static_cast<int32_t>(ul);
  }
}

std::string DiagonalString(Diagonal d) {
  switch (d) {
    case Diagonal::kUnit:
      return "Unit";
    case Diagonal::kNonUnit:
      return "NonUnit";
    default:
      LOG(FATAL) << "Unknown diagonal " << static_cast<int32_t>(d);
  }
}

std::string SideString(Side s) {
  switch (s) {
    case Side::kLeft:
      return "Left";
    case Side::kRight:
      return "Right";
    default:
      LOG(FATAL) << "Unknown side " << static_cast<int32_t>(s);
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
    case ComputationType::kF16AsF32:
      return "f16 (w/ f32 accumulation)";
    case ComputationType::kBF16AsF32:
      return "bf16 (w/ f32 accumulation)";
    case ComputationType::kTF32AsF32:
      return "tf32 (w/ f32 accumulation)";
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
      LOG(FATAL) << "Unknown DataType " << static_cast<int32_t>(ty);
  }
}

std::ostream& operator<<(std::ostream& os, DataType ty) {
  return os << DataTypeString(ty);
}

xla::BlasTransposeProto ToProto(Transpose t) {
  switch (t) {
    case Transpose::kNoTranspose:
      return xla::BLAS_NO_TRANSPOSE;
    case Transpose::kTranspose:
      return xla::BLAS_TRANSPOSE;
    case Transpose::kConjugateTranspose:
      return xla::BLAS_CONJUGATE_TRANSPOSE;
  }
}

absl::StatusOr<Transpose> FromProto(xla::BlasTransposeProto t) {
  switch (t) {
    case xla::BLAS_NO_TRANSPOSE:
      return Transpose::kNoTranspose;
    case xla::BLAS_TRANSPOSE:
      return Transpose::kTranspose;
    case xla::BLAS_CONJUGATE_TRANSPOSE:
      return Transpose::kConjugateTranspose;
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid BlasTranspose value: ", t));
  }
}

xla::BlasComputationTypeProto ToProto(ComputationType ty) {
  switch (ty) {
    case ComputationType::kF16:
      return xla::BLAS_COMPUTATION_TYPE_F16;
    case ComputationType::kF32:
      return xla::BLAS_COMPUTATION_TYPE_F32;
    case ComputationType::kF64:
      return xla::BLAS_COMPUTATION_TYPE_F64;
    case ComputationType::kI32:
      return xla::BLAS_COMPUTATION_TYPE_I32;
    case ComputationType::kF16AsF32:
      return xla::BLAS_COMPUTATION_TYPE_F16_AS_F32;
    case ComputationType::kBF16AsF32:
      return xla::BLAS_COMPUTATION_TYPE_BF16_AS_F32;
    case ComputationType::kTF32AsF32:
      return xla::BLAS_COMPUTATION_TYPE_TF32_AS_F32;
  }
}

std::optional<ComputationType> FromProto(xla::BlasComputationTypeProto ty) {
  switch (ty) {
    case xla::BLAS_COMPUTATION_TYPE_F16:
      return ComputationType::kF16;
    case xla::BLAS_COMPUTATION_TYPE_F32:
      return ComputationType::kF32;
    case xla::BLAS_COMPUTATION_TYPE_F64:
      return ComputationType::kF64;
    case xla::BLAS_COMPUTATION_TYPE_I32:
      return ComputationType::kI32;
    case xla::BLAS_COMPUTATION_TYPE_F16_AS_F32:
      return ComputationType::kF16AsF32;
    case xla::BLAS_COMPUTATION_TYPE_BF16_AS_F32:
      return ComputationType::kBF16AsF32;
    case xla::BLAS_COMPUTATION_TYPE_TF32_AS_F32:
      return ComputationType::kTF32AsF32;
    case xla::BLAS_COMPUTATION_TYPE_UNSET:
    default:
      return std::nullopt;
  }
}

}  // namespace blas
}  // namespace stream_executor
