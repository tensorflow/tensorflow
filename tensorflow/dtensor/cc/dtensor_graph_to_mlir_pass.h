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

#ifndef TENSORFLOW_DTENSOR_CC_DTENSOR_GRAPH_TO_MLIR_PASS_H_
#define TENSORFLOW_DTENSOR_CC_DTENSOR_GRAPH_TO_MLIR_PASS_H_

#include "absl/container/flat_hash_set.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"

namespace tensorflow {

class DTensorMlirPassRunner {
 public:
  DTensorMlirPassRunner();
  // Translates `graph` and replaces it with the resulting rewritten graph.
  Status RunOnGraph(const DeviceSet& device_set, bool is_func,
                    FunctionLibraryDefinition* flib_def,
                    std::unique_ptr<Graph>* graph,
                    absl::flat_hash_set<Node*>& control_ret_nodes,
                    Fprint128 cache_key);

 private:
  // N.B. op_registration_ must be initialized before context/pass-manager to
  // ensure DTensor operations are available during optimization passes.
  bool op_registration_ = mlir::TF::RegisterDTensorTFOps();
  mlir::MLIRContext context_;
  mlir::PassManager pass_manager_;

  bool logging_enabled_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_DTENSOR_CC_DTENSOR_GRAPH_TO_MLIR_PASS_H_
