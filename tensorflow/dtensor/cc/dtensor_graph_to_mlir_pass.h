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

#include <memory>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
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

  // Imports Graph to MLIR module in tf_execute Dialect with DTensor attributes.
  absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ImportGraphToMlir(
      const DeviceSet& device_set, absl::string_view name, bool is_func,
      const dtensor::Mesh& default_mesh,
      const FunctionLibraryDefinition& flib_def, const Graph& graph,
      Fprint128 cache_key);

  // Transforms input MLIR module with DTensor Pass pipeline.
  Status Run(mlir::ModuleOp module);

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
