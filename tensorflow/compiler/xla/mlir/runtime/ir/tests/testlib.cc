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

#include "tensorflow/compiler/xla/mlir/runtime/ir/tests/testlib.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/DialectImplementation.h"  // from @llvm-project

// clang-format off
#include "tensorflow/compiler/xla/mlir/runtime/ir/tests/testlib_dialect.cc.inc"
#include "tensorflow/compiler/xla/mlir/runtime/ir/tests/testlib_enums.cc.inc"
// clang-format on

#define GET_ATTRDEF_CLASSES
#include "tensorflow/compiler/xla/mlir/runtime/ir/tests/testlib_attrs.cc.inc"

#define GET_TYPEDEF_CLASSES
#include "tensorflow/compiler/xla/mlir/runtime/ir/tests/testlib_types.cc.inc"

namespace xla {
namespace runtime {

void TestlibDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "tensorflow/compiler/xla/mlir/runtime/ir/tests/testlib_attrs.cc.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "tensorflow/compiler/xla/mlir/runtime/ir/tests/testlib_types.cc.inc"
      >();
}

}  // namespace runtime
}  // namespace xla
