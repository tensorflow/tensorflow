/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_IFRT_IR_VIFRT_BASE_TD_
#define XLA_PYTHON_IFRT_IR_VIFRT_BASE_TD_

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

// VIFRT represents the layout only. Therefore, it uses AnyType everywhere.
def Vifrt_AnyType : AnyTypeOf<[AnyType]>;
def Vifrt_AnyAttr : AnyAttrOf<[AnyAttr]>;
def Vifrt_AnyRegion : Region<CPred<"true">, "any region">;

#endif  // XLA_PYTHON_IFRT_IR_VIFRT_BASE_TD_