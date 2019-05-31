/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_TOCO_IMPORT_TENSORFLOW_H_
#define TENSORFLOW_LITE_TOCO_IMPORT_TENSORFLOW_H_

#include <memory>
#include <string>
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/model_flags.pb.h"

namespace toco {

struct TensorFlowImportFlags {
  // If true, control dependencies will be dropped immediately
  // during the import of the TensorFlow GraphDef.
  bool drop_control_dependency = false;

  // Do not recognize any op and import all ops as
  // `TensorFlowUnsupportedOperator`. This is used to populated with the
  // `force_select_tf_ops` flag.
  bool import_all_ops_as_unsupported = false;
};

// Converts TOCO model from TensorFlow GraphDef with given flags.
std::unique_ptr<Model> ImportTensorFlowGraphDef(
    const ModelFlags& model_flags, const TensorFlowImportFlags& tf_import_flags,
    const tensorflow::GraphDef& graph_def);

// Converts TOCO model from the file content of TensorFlow GraphDef with given
// flags.
std::unique_ptr<Model> ImportTensorFlowGraphDef(
    const ModelFlags& model_flags, const TensorFlowImportFlags& tf_import_flags,
    const string& input_file_contents);

// Gets a list of supported ops by their names.
std::vector<std::string> GetPotentiallySupportedOps();

}  // namespace toco

#endif  // TENSORFLOW_LITE_TOCO_IMPORT_TENSORFLOW_H_
