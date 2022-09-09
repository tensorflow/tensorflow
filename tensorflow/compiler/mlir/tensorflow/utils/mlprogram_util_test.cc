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

#include "tensorflow/compiler/mlir/tensorflow/utils/mlprogram_util.h"

#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

static void RegisterDialects(mlir::MLIRContext& context) {
  context.loadDialect<mlir::TF::TensorFlowDialect>();
}

TEST(ModuleToMlProgram, SmokeTest) {
  mlir::MLIRContext context;
  RegisterDialects(context);

  mlir::OwningOpRef<mlir::ModuleOp> module(
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context)));
  auto result = tensorflow::LowerToMlProgramAndHlo(*module);
  EXPECT_TRUE(succeeded(result));
}

}  // namespace
}  // namespace tensorflow
