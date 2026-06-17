/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_XLA_CALL_MODULE_ATTRS_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_XLA_CALL_MODULE_ATTRS_H_

#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace TF {

// The main function's name in the serialized stablehlo module embedded in
// XlaCallModule's `module` attribute.
constexpr llvm::StringRef kStablehloMainFunctionName = "main";

// After deserializing the stablehlo functions from XlaCallModule,
// this XlaCallModule attribute refers to the deserialized stablehlo main
// function.
constexpr llvm::StringRef kStablehloEntryFunctionAttrName = "_entry_function";

// The StableHLO version of the serialized stablehlo module embedded in
// XlaCallModule's `module` attribute, set on deserialization.
constexpr llvm::StringRef kStablehloVersionAttrName = "_stablehlo_version";

// Every stablehlo function deserialized from XlaCallModule has this attribute.
constexpr llvm::StringRef kFromXlaCallModuleAttrName = "_from_xla_call_module";

// Name of `tf.XlaCallModule`'s dictionary attribute for keeping the
// deserialized stablehlo module's attributes.
constexpr llvm::StringRef kStablehloModuleAttrsAttrName =
    "_stablehlo_module_attrs";

}  // namespace TF
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_XLA_CALL_MODULE_ATTRS_H_
