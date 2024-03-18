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
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/debugger.h"

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/graph_def.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/exported_model.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/tf_quant_ops.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"

namespace stablehlo::quantization {

void DisableDebugging(mlir::ModuleOp module_op) {
  module_op.walk(
      [](mlir::TF::DumpTensorOp dump_op) { dump_op.setEnabled(false); });
}

void EnableDebugging(tensorflow::quantization::ExportedModel& exported_model) {
  MutateNodeDefs(*exported_model.mutable_graph_def(),
                 [](tensorflow::NodeDef& node_def) {
                   if (node_def.op() == "DumpTensor") {
                     (*node_def.mutable_attr())["enabled"].set_b(true);
                   }
                 });
}

void ChangeToQuantizedFilename(mlir::ModuleOp module_op) {
  module_op.walk([](mlir::TF::DumpTensorOp dump_op) {
    dump_op.setFileName("quantized_tensor_data.pb");
  });
}

}  // namespace stablehlo::quantization
