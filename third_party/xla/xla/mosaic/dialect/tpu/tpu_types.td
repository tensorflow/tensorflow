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

#ifndef TPU_TYPES
#define TPU_TYPES

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "xla/mosaic/dialect/tpu/tpu.td"

class TPU_Type<string name, string mnemonic_, list<Trait> traits = [],
               string baseCppType = "::mlir::Type">
    : TypeDef<TPU_Dialect, name, traits, baseCppType> {
  let mnemonic = mnemonic_;
}

def TPU_Float8EXMYType : TPU_Type<"Float8EXMY", "float8_exmy",
                            [DeclareTypeInterfaceMethods<FloatTypeInterface, ["getFloatSemantics"]>]> {
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
}

#endif  // TPU_TYPES
