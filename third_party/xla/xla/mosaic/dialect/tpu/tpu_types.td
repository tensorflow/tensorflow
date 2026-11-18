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
include "xla/mosaic/dialect/tpu/tpu_type_base.td"

class TPU_Type<string name, string mnemonic_, list<Trait> traits = [],
               string baseCppType = "::mlir::Type">
    : TypeDef<TPU_Dialect, name, traits, baseCppType> {
  let mnemonic = mnemonic_;
}

def TPU_Float8EXMYType : TPU_Float8EXMYTypeBase<TPU_Dialect>;

#endif  // TPU_TYPES
