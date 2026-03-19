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

#ifndef TENSORFLOW_COMPILER_MLIR_TF2XLA_TRANSFORMS_LEGALIZATION_OP_CONFIG_H_
#define TENSORFLOW_COMPILER_MLIR_TF2XLA_TRANSFORMS_LEGALIZATION_OP_CONFIG_H_

#include "mlir/Support/TypeID.h"  // from @llvm-project

namespace mlir {
namespace hlo {

// Given the type ID, check if it's legalized with MLIR.
bool IsTypeLegalizedWithMlir(const TypeID& type_id);

// Returns true if the op is considered a dynamic padder op.
bool IsDynamicPadderOp(const TypeID& type_id);

// Returns True if this op has a Tf2XLA fallback. Currently, this is not the
// inverse of the !IsOpLegalizedWithMlir, but it should be.
bool HasTf2XlaFallback(const TypeID& type_id);

// Whether this type is allowed to have a TF2XLA fallback.
bool IsOpAllowedTf2xlaFallback(const TypeID& type_id);

// Whether this type is Preferred to use a TF2XLA fallback kernel when using
// the MLIR bridge. If this is true, then the TF2XLA fallback kernel will be
// used over the MLIR lowering.
bool IsOpAllowedTf2xlaPreferred(const TypeID& type_id);

}  // namespace hlo
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TF2XLA_TRANSFORMS_LEGALIZATION_OP_CONFIG_H_
