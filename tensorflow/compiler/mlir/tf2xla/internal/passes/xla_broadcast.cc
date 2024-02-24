/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"

namespace tensorflow {
namespace tf2xla {
namespace internal {

namespace {
using mlir::OperationPass;
using mlir::func::FuncOp;

#define GEN_PASS_DEF_XLABROADCASTPASS
#include "tensorflow/compiler/mlir/tf2xla/internal/passes/clustering_passes.h.inc"

struct XlaBroadcast : public impl::XlaBroadcastPassBase<XlaBroadcast> {
  void runOnOperation() override;
};

void XlaBroadcast::runOnOperation() {}

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> CreateXlaBroadcastPass() {
  return std::make_unique<XlaBroadcast>();
}

}  // namespace internal
}  // namespace tf2xla
}  // namespace tensorflow
