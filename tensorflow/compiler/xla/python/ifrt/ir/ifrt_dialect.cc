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

#include "tensorflow/compiler/xla/python/ifrt/ir/ifrt_dialect.h"

#include <cstdint>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/xla/python/ifrt/ir/ifrt_ops.h"

// Generated definitions.
#include "tensorflow/compiler/xla/python/ifrt/ir/ifrt_dialect.cc.inc"
#include "tensorflow/compiler/xla/python/ifrt/ir/sharding_param.h"
#define GET_TYPEDEF_CLASSES
#include "tensorflow/compiler/xla/python/ifrt/ir/ifrt_types.cc.inc"

namespace xla {
namespace ifrt {

void IfrtDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "tensorflow/compiler/xla/python/ifrt/ir/ifrt_types.cc.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/xla/python/ifrt/ir/ifrt_ops.cc.inc"
      >();
}

// static
mlir::LogicalResult IfrtArrayType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emit_error,
    mlir::RankedTensorType global, ShardingParam sharding,
    llvm::ArrayRef<int64_t> devices) {
  llvm::SmallSet<int64_t, 4> device_set;
  for (auto device : devices) {
    if (!device_set.insert(device).second) {
      return emit_error() << "`devices` has duplicated id " << device;
    }
  }

  if (mlir::failed(sharding.verify(emit_error))) {
    return mlir::failure();
  }

  int64_t devices_in_mesh = 1;
  for (const int64_t axis_size : sharding.minor_to_major().axis_sizes) {
    devices_in_mesh *= axis_size;
  }
  if (devices_in_mesh != devices.size()) {
    return emit_error() << "Requires the same amount of `devices` and from "
                           "`sharding`. Actual: "
                        << devices.size() << " vs " << devices_in_mesh;
  }

  return mlir::success();
}

}  // namespace ifrt
}  // namespace xla
