/* Copyright 2026 The OpenXLA Authors.

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

#ifndef TPU_TYPE_BASE
#define TPU_TYPE_BASE

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

class TPU_Float8EXMYTypeBase<Dialect dialect> : TypeDef<dialect, "Float8EXMY",
                            [DeclareTypeInterfaceMethods<FloatTypeInterface, ["getFloatSemantics"]>]> {
    let mnemonic = "float8_exmy";
    let summary = "EXMY type in a 8 bit container";
    let description = [{
      EXMY type in a 8 bit container. Meaningful bits are aligned to LSB, and
      bits higher than the underlying exmy type in the container are considered
      as ignored. See https://arxiv.org/abs/2405.13938 for more details.
    }];

    let parameters = (ins
      TypeParameter<"::mlir::FloatType", "Underlying EXMY type">:$underlying_type
    );

    let assemblyFormat = [{
      `<` $underlying_type `>`
    }];

    let extraClassDefinition = [{
      const ::llvm::fltSemantics& $cppClass::getFloatSemantics() const {
        return getUnderlyingType().getFloatSemantics();
      }
    }];
}

#endif  // TPU_TYPE_BASE
