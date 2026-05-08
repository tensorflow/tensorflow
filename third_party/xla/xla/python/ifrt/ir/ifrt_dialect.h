/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_IFRT_IR_IFRT_DIALECT_H_
#define XLA_PYTHON_IFRT_IR_IFRT_DIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "xla/pjrt/layout_mode.h"
#include "xla/python/ifrt/ir/ifrt_interfaces.h"
#include "xla/python/ifrt/ir/sharding_param.h"
#include "xla/python/ifrt/memory.h"

// Generated definitions.
#include "xla/python/ifrt/ir/ifrt_dialect.h.inc"  // IWYU pragma: export
#define GET_ATTRDEF_CLASSES
#include "xla/python/ifrt/ir/ifrt_attrs.h.inc"  // IWYU pragma: export
#define GET_TYPEDEF_CLASSES
#include "xla/python/ifrt/ir/ifrt_types.h.inc"  // IWYU pragma: export

namespace xla {
namespace ifrt {

// Returns true if the given function is an IFRT function.
bool IsIfrtFunction(mlir::Operation* op);

// Returns the `IfrtArrayType` of the given value.
//
// CHECK fails if the type of the value is not an `IfrtArrayType`.
IfrtArrayType GetArrayType(mlir::Value value);

IfrtArrayType GetArrayType(mlir::Type type);

// Returns the `IfrtShardingParamAttr` of the given `IfrtArrayType`.
//
// CHECK fails if the sharding attribute is not an `IfrtShardingParamAttr`.
IfrtShardingParamAttr GetShardingParamAttr(IfrtArrayType array_type);

// Returns true if the given attribute is has unspecified sharding.
bool IsUnspecifiedSharding(IfrtShardingAttrInterface sharding_attr);

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_IR_IFRT_DIALECT_H_
