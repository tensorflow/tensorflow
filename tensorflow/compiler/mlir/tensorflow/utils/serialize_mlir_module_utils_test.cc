/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/tensorflow/utils/serialize_mlir_module_utils.h"

#include <string>

#include <gtest/gtest.h>
#include "absl/strings/match.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(SerializeMlirModuleUtilsTest, DebugInfoSerialization) {
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> mlir_module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));

  GetMlirCommonFlags()->tf_mlir_enable_debug_info_serialization = true;
  std::string serialized_module = SerializeMlirModule(*mlir_module);
  EXPECT_TRUE(absl::StrContains(serialized_module, "loc("));

  GetMlirCommonFlags()->tf_mlir_enable_debug_info_serialization = false;
  serialized_module = SerializeMlirModule(*mlir_module);
  EXPECT_FALSE(absl::StrContains(serialized_module, "loc("));
}

}  // namespace
}  // namespace tensorflow
