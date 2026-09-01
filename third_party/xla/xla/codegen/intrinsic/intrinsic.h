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

#ifndef XLA_CODEGEN_INTRINSIC_INTRINSIC_H_
#define XLA_CODEGEN_INTRINSIC_INTRINSIC_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Target/TargetMachine.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "xla/codegen/intrinsic/type.h"
#include "xla/codegen/intrinsic/vec_name_mangler.h"
#include "xla/xla_data.pb.h"

namespace xla::codegen::intrinsics {

enum class DeviceType {
  kAmdCpu,
  kIntelCpu,
  kIntelGpu,
  kArmCpu,
  kSystemZCpu,
  kNvidiaGpu,
  kAmdGpu,
};

struct IntrinsicOptions {
  // CPU features available on the target machine.
  std::string features;

  // The type of device the target machine is running on. Defaults to Intel CPU.
  DeviceType device_type = DeviceType::kIntelCpu;
  // Disables math functions that do not have the same results across e.g.
  // AMD vs. Intel CPUs.
  bool disable_platform_dependent_math = false;

  // Preferred vector width of the target module.
  int prefer_vector_width = 0;

  bool Contains(absl::string_view feature) const {
    return absl::StrContains(features, feature);
  }
};

// Intrinsics are provided by XLA to expose special features (functions) that
// may only be implemented with code generator support.
//
// XLA intrinsics are conceptually similar to intrinsics in LLVM IR, but
// implemented inside XLA, and during the compilation process they are lowered
// into the LLVM IR.
//
// Some of the builting LLVM intrinsics are lowered to external function calls
// (i.e. llvm.tanh.* is lowered to libm call), or to the 'compiler-rt' function
// calls (i.e. f32 to bf16 truncation). Having external function calls on a hot
// path is very expensive, furthermore they are ofter can't be vectorized by the
// LLVM, which also adds a large performance penalty. We choose to implement
// intrinsics critical for performance inside XLA iteself:
//
// 1. Math functions (exp, tanh, etc.) using polynomial approximations, to avoid
//    external functions calls to libm and to get the same numerics as TPU and
//    HLO Interpreter.
//
// 2. Conversion between different floating point types, to avoid expensive
//    external function calls to compiler-rt, and again to get consistent
//    numerics with TPU and HLO Interpreter (handling Nans and infinities).
//
// Similar to LLVM intrinsics, XLA intrinsics are overloaded on the data type(s)
// and vector width(s) of the argument and result.
template <typename Derived>
class Intrinsic {
 public:
  // Whether the simd function is masked.
  static constexpr bool kIsMasked = false;
  // If false, the first argument is the return type.
  static constexpr bool kLastArgIsReturnType = false;
  // How many arguments this function takes.
  static constexpr int8_t kNumArgs = 1;

  static std::string Name(llvm::ArrayRef<Type> types) {
    return ::xla::codegen::intrinsic::GetTypedName(
        Derived::kLastArgIsReturnType, types, Derived::kName);
  }

  template <typename... Args>
  static absl::StatusOr<llvm::Function*> CreateDefinition(llvm::Module* module,
                                                          const Args... types) {
    static_assert(sizeof...(Args) > 0, "At least one argument is required.");
    static_assert((std::is_convertible_v<Args, Type> && ...),
                  "All arguments must be intrinsic::Type.");
    return Derived::CreateDefinition(module, types...);
  }

  static mlir::func::FuncOp GetOrInsertDeclaration(mlir::OpBuilder& b,
                                                   mlir::ModuleOp& module,
                                                   llvm::ArrayRef<Type> types) {
    CHECK(!types.empty()) << "At least one argument is required.";

    std::vector<mlir::Type> ir_types;
    ir_types.reserve(types.size());
    for (const auto& type : types) {
      ir_types.push_back(Type::TypeToIrType(type, *module.getContext()));
    }
    mlir::Type return_type = ir_types.front();
    if (Derived::kLastArgIsReturnType) {
      return_type = ir_types.back();
      ir_types.pop_back();
    }
    mlir::FunctionType type =
        mlir::FunctionType::get(module.getContext(), ir_types, {return_type});

    // Check if the function already exists, and has the correct type.
    std::string name = Name(types);
    if (auto func = module.lookupSymbol<mlir::func::FuncOp>(name);
        func && func.getFunctionType() == type) {
      return func;
    }

    // If not found or type mismatch, create the declaration.
    mlir::OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(module.getBody());

    auto decl = mlir::func::FuncOp::create(b, module.getLoc(), name, type);
    decl.setPrivate();
    return decl;
  }

  static llvm::Function* GetOrInsertDeclaration(llvm::Module* module,
                                                llvm::ArrayRef<Type> types) {
    CHECK(!types.empty()) << "At least one argument is required.";
    std::vector<llvm::Type*> ir_types;
    ir_types.reserve(types.size());
    for (const auto& type : types) {
      ir_types.push_back(Type::TypeToIrType(type, module->getContext()));
    }
    llvm::Type* return_type = ir_types.front();
    if (Derived::kLastArgIsReturnType) {
      return_type = ir_types.back();
      ir_types.pop_back();
    }
    auto* function_type = llvm::FunctionType::get(return_type, ir_types, false);
    return llvm::cast<llvm::Function>(
        module->getOrInsertFunction(Name(types), function_type).getCallee());
  }

  static bool IsSupported(absl::string_view features,
                          llvm::ArrayRef<Type> types) {
    for (const auto& supported_args : Derived::SupportedVectorTypes(features)) {
      if (llvm::ArrayRef<Type>(supported_args) == types) {
        return true;
      }
    }
    return false;
  }
};
}  // namespace xla::codegen::intrinsics

#endif  // XLA_CODEGEN_INTRINSIC_INTRINSIC_H_
