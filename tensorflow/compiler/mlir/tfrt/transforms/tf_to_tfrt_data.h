/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_TF_TO_TFRT_DATA_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_TF_TO_TFRT_DATA_H_

#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/platform/status.h"
#include "tfrt/bef/bef_buffer.h"  // from @tf_runtime

namespace tensorflow {

// Create a pass that converts MLIR TF dialect to MLIR TFRT CoreRT dialect.
std::unique_ptr<mlir::Pass> CreateTFToTFRTDataConversionPass();

Status TFDataGraphDefToHostBEF(const GraphDef& graph_def, tfrt::BefBuffer* bef);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_TF_TO_TFRT_DATA_H_
