/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_CODEGEN_INTRINSIC_CPP_INTRINSIC_DECLARATIONS_H_
#define XLA_CODEGEN_INTRINSIC_CPP_INTRINSIC_DECLARATIONS_H_

#include <vector>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "xla/codegen/intrinsic/cpp/cpp_gen_intrinsics.h"
#include "xla/codegen/intrinsic/intrinsic.h"
#include "xla/codegen/intrinsic/type.h"
#include "xla/xla_data.pb.h"

namespace xla::codegen::intrinsics {

class EigenTanh : public Intrinsic<EigenTanh> {
 public:
  static constexpr absl::string_view kName = "tanh";

  static std::vector<std::vector<Type>> SupportedVectorTypes(
      absl::string_view features) {
    if (!AreEigenIntrinsicsAvailable()) {
      return {};
    }
    return {
        {Type::S(xla::F32)},     {Type::V(xla::F32, 4)}, {Type::V(xla::F32, 8)},
        {Type::V(xla::F32, 16)}, {Type::S(xla::F64)},    {Type::V(xla::F64, 4)},
        {Type::V(xla::F64, 8)},
    };
  }

  static absl::StatusOr<llvm::Function*> CreateDefinition(
      llvm::Module* module, const IntrinsicOptions& options, Type type) {
    return GetCppGenFunction(module, Name(type));
  }
};

class EigenAtan : public Intrinsic<EigenAtan> {
 public:
  static constexpr absl::string_view kName = "atan";

  static std::vector<std::vector<Type>> SupportedVectorTypes(
      absl::string_view features) {
    if (!AreEigenIntrinsicsAvailable()) {
      return {};
    }
    // On ARM NEON, Remez reciprocal division (1.0f / abs_x) can trigger
    // division traps or underflow near zero under hardware Flush-To-Zero (FTZ)
    // execution. We advertise scalar support only so that MLIR automatically
    // unrolls vector lanes to scalar xla.atan.f32/f64, where genuine CPU
    // short-circuit conditional branching (abs_x < 1e-3) bypasses Remez
    // approximation.
    if (absl::StrContains(features, "+neon")) {
      return {
          {Type::S(xla::F32)},
          {Type::S(xla::F64)},
      };
    }
    return {
        {Type::S(xla::F32)},     {Type::V(xla::F32, 4)}, {Type::V(xla::F32, 8)},
        {Type::V(xla::F32, 16)}, {Type::S(xla::F64)},    {Type::V(xla::F64, 4)},
        {Type::V(xla::F64, 8)},
    };
  }

  static absl::StatusOr<llvm::Function*> CreateDefinition(
      llvm::Module* module, const IntrinsicOptions& options, Type type) {
    return GetCppGenFunction(module, Name(type));
  }
};

class EigenSin : public Intrinsic<EigenSin> {
 public:
  static constexpr absl::string_view kName = "sin";

  static std::vector<std::vector<Type>> SupportedVectorTypes(
      absl::string_view features) {
    if (!AreEigenIntrinsicsAvailable()) {
      return {};
    }
    return {
        {Type::S(xla::F32)},     {Type::V(xla::F32, 4)}, {Type::V(xla::F32, 8)},
        {Type::V(xla::F32, 16)}, {Type::S(xla::F64)},    {Type::V(xla::F64, 4)},
        {Type::V(xla::F64, 8)},
    };
  }

  static absl::StatusOr<llvm::Function*> CreateDefinition(
      llvm::Module* module, const IntrinsicOptions& options, Type type) {
    return GetCppGenFunction(module, Name(type));
  }
};

class EigenCos : public Intrinsic<EigenCos> {
 public:
  static constexpr absl::string_view kName = "cos";

  static std::vector<std::vector<Type>> SupportedVectorTypes(
      absl::string_view features) {
    if (!AreEigenIntrinsicsAvailable()) {
      return {};
    }
    return {
        {Type::S(xla::F32)},     {Type::V(xla::F32, 4)}, {Type::V(xla::F32, 8)},
        {Type::V(xla::F32, 16)}, {Type::S(xla::F64)},    {Type::V(xla::F64, 4)},
        {Type::V(xla::F64, 8)},
    };
  }

  static absl::StatusOr<llvm::Function*> CreateDefinition(
      llvm::Module* module, const IntrinsicOptions& options, Type type) {
    return GetCppGenFunction(module, Name(type));
  }
};
}  // namespace xla::codegen::intrinsics

#endif  // XLA_CODEGEN_INTRINSIC_CPP_INTRINSIC_DECLARATIONS_H_
