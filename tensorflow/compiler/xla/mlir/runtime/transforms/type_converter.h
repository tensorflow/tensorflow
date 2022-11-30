/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_MLIR_RUNTIME_TRANSFORMS_TYPE_CONVERTER_H_
#define TENSORFLOW_COMPILER_XLA_MLIR_RUNTIME_TRANSFORMS_TYPE_CONVERTER_H_

#include <functional>
#include <memory>

#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "tensorflow/compiler/xla/runtime/types.h"

namespace xla {
namespace runtime {

//===----------------------------------------------------------------------===//
// Type conversion from the compile time types to the run-time types.
//===----------------------------------------------------------------------===//

// Type converter converts MLIR types known at compile time to the corresponding
// types used at run time. It provides default conversions for the canonical
// types (memrefs, tensors, etc...) and allows users to register custom
// conversions for user-defined types.
class TypeConverter {
 public:
  // Conversion function must return run time type corresponding to the compile
  // time type if the conversion is successful, or `nullptr` if failed.
  using ConversionFn = std::function<std::unique_ptr<Type>(mlir::Type)>;

  TypeConverter() = default;

  template <typename... Fns>
  explicit TypeConverter(Fns&&... fn) {
    (AddConversion(std::forward<Fns>(fn)), ...);
  }

  // Adds a type conversion function with a type predicate.
  //
  // Example:
  //
  //   AddConversion([](mlir::TensorType) -> std::unique_ptr<Type> { ... });
  //
  // The conversion function will match only the tensor type, and return empty
  // result for all other types, and the type converter will try the next
  // conversion function (see `Convert` implementation).
  template <typename Fn, typename FnTraits = llvm::function_traits<Fn>>
  void AddConversion(Fn&& fn) {
    using ArgType = typename FnTraits::template arg_t<0>;
    conversions_.emplace_back(
        [fn = std::forward<Fn>(fn)](mlir::Type type) -> std::unique_ptr<Type> {
          if (auto arg = type.dyn_cast<ArgType>()) return fn(arg);
          return {};
        });
  }

  // Converts MLIR element type to the PrimitiveType.
  static absl::StatusOr<PrimitiveType> ConvertElementType(mlir::Type type);

  // Converts MLIR type to the runtime type. Returns error if conversion was not
  // successful and the type has no corresponding run time type.
  absl::StatusOr<std::unique_ptr<Type>> Convert(mlir::Type type) const;

  // Converts MLIR function type to the runtime function type. Returns error if
  // function has unsupported operands or results types.
  absl::StatusOr<FunctionType> Convert(mlir::FunctionType type) const;

 private:
  llvm::SmallVector<ConversionFn> conversions_;
};

}  // namespace runtime
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_MLIR_RUNTIME_TRANSFORMS_TYPE_CONVERTER_H_
