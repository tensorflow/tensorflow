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
#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PYTHON_PY_FUNCTION_LIB_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PYTHON_PY_FUNCTION_LIB_H_

#include <string>
#include <unordered_set>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/exported_model.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace tensorflow::quantization {

// Declares pure virtual member functions for a python-side derived class to
// override. This allows calling python implementations from the C++ layer.
// Member functions should be pure not stateful; they should not access or rely
// on member fields.
class PyFunctionLibrary {
 public:
  virtual ~PyFunctionLibrary() = default;

  // Assigns UUIDs to each CustomAggregator op found in each GraphDef in
  // `exported_model`. The UUIDs are set to the `id` attributes. The UUIDs will
  // be used during calibration step to identify the collected quantization
  // statistics for each CustsomAggregator op.
  virtual ExportedModel AssignIdsToCustomAggregatorOps(
      const ExportedModel& exported_model) const = 0;

  // Saves `exported_model` to `dst_saved_model_path` as SavedModel.
  // `src_saved_model_path` is the path to the source SavedModel from which the
  // exported model is produced. It is used to copy the asset files to
  // `dst_saved_model_path`. `tags` will be attached to the saved
  // `MetaGraphDef`. `signature_def_map` will be passed to the
  // `add_meta_graph_and_variables` function, which is internally used to add a
  // `MetaGraphDef` to save to the SavedModel.
  virtual void SaveExportedModel(
      absl::string_view dst_saved_model_path,
      const ExportedModel& exported_model,
      absl::string_view src_saved_model_path,
      const std::unordered_set<std::string>& tags,
      const absl::flat_hash_map<std::string, tensorflow::SignatureDef>&
          signature_def_map) const = 0;
};

}  // namespace tensorflow::quantization

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PYTHON_PY_FUNCTION_LIB_H_
