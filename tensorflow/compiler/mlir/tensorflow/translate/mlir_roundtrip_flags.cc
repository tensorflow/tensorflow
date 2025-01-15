/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"

#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow {

std::string GraphImportConfig::str() const {
  std::ostringstream ss;

  ss << "graph_func_name: " << graph_func_name;
  InputArrays inputs;
  ss << "\ninputs: ";
  for (auto& it : inputs) {
    ss << "\n\t" << it.first << " -> "
       << DataTypeString(it.second.imported_dtype) << " "
       << it.second.shape.DebugString();
  }
  ss << "\noutputs:";
  for (auto& output : outputs) ss << " " << output;
  ss << "\ncontrol_outputs:";
  for (auto& output : control_outputs) ss << " " << output;
  ss << "\nprune_unused_nodes: " << prune_unused_nodes;
  ss << "\nconvert_legacy_fed_inputs: " << convert_legacy_fed_inputs;
  ss << "\ngraph_as_function: " << graph_as_function;
  ss << "\nupgrade_legacy: " << upgrade_legacy;
  ss << "\nrestrict_functionalization_to_compiled_nodes: "
     << restrict_functionalization_to_compiled_nodes;
  ss << "\nenable_shape_inference: " << enable_shape_inference;
  ss << "\nunconditionally_use_set_output_shapes: "
     << unconditionally_use_set_output_shapes;
  ss << "\nxla_compile_device_type: " << xla_compile_device_type;

  return ss.str();
}

}  // namespace tensorflow
