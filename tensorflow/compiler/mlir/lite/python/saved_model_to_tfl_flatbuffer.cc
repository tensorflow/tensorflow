/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/lite/python/saved_model_to_tfl_flatbuffer.h"

#include <utility>

#include "llvm/Support/ToolOutputFile.h"
#include "mlir/IR/MLIRContext.h"  // TF:llvm-project
#include "mlir/IR/Module.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "mlir/Support/FileUtilities.h"  // TF:llvm-project
#include "mlir/Transforms/ViewOpGraph.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/lite/common/tfl_pass_config.h"
#include "tensorflow/compiler/mlir/lite/python/tf_tfl_flatbuffer_helpers.h"
#include "tensorflow/compiler/mlir/lite/tf_tfl_passes.h"
#include "tensorflow/compiler/mlir/lite/tf_to_tfl_flatbuffer.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/graph_debug_info.pb.h"
#include "tensorflow/lite/toco/model_flags.pb.h"
#include "tensorflow/lite/toco/toco_flags.pb.h"
#include "tensorflow/lite/toco/types.pb.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {

Status ConvertSavedModelToTFLiteFlatBuffer(
    const toco::ModelFlags& model_flags, const toco::TocoFlags& toco_flags,
    const string& saved_model_dir, bool saved_model_v1,
    const string& saved_model_tags, const string& saved_model_exported_names,
    string* result) {
  mlir::MLIRContext context;
  mlir::TFL::QuantizationSpecs quant_specs;

  // Parse input arrays.
  std::vector<string> node_names;
  std::vector<string> node_dtypes;
  std::vector<std::vector<int>> node_shapes;
  std::vector<double> node_mins;
  std::vector<double> node_maxs;

  // Populate quantization specs.
  TF_RETURN_IF_ERROR(internal::PopulateQuantizationSpecs(
      model_flags, toco_flags, &quant_specs, &node_names, &node_dtypes,
      &node_shapes, &node_mins, &node_maxs));

  internal::WarningUnusedFlags(model_flags, toco_flags);

  // Register all custom ops, including user-specified custom ops.
  TF_RETURN_IF_ERROR(internal::RegisterAllCustomOps(toco_flags));

  const bool import_saved_model = !saved_model_v1;
  TF_ASSIGN_OR_RETURN(
      auto module,
      ImportSavedModel(import_saved_model, saved_model_v1, saved_model_dir,
                       saved_model_tags, saved_model_exported_names, &context));
  return internal::ConvertMLIRToTFLiteFlatBuffer(toco_flags, std::move(module),
                                                 quant_specs, result);
}

}  // namespace tensorflow
