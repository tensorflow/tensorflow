/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include <string>
#include <vector>

#include "absl/strings/numbers.h"
#include "tensorflow/contrib/lite/toco/model_cmdline_flags.h"
#include "tensorflow/contrib/lite/toco/toco_saved_model.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"

namespace toco {
namespace {

// Loads a SavedModel from the directory specified in parsed_toco_flags.
// Returns a SavedModelBundle with the requested MetaGraphDef.
const tensorflow::SavedModelBundle* LoadSavedModel(
    const ParsedTocoFlags& parsed_toco_flags) {
  const string model_path = parsed_toco_flags.savedmodel_directory.value();
  QCHECK(tensorflow::MaybeSavedModelDirectory(model_path))
      << "Model is not saved in the supported SavedModel format.\n";

  // Gets the tags identifying the MetaGraphDef from the command line arguments.
  string tags_str;
  if (parsed_toco_flags.savedmodel_tagset.specified()) {
    tags_str = parsed_toco_flags.savedmodel_tagset.value();
  } else {
    tags_str = parsed_toco_flags.savedmodel_tagset.default_value();
  }
  auto tags = absl::StrSplit(tags_str, ',');

  // Loads MetaGraphDef.
  auto* bundle = new tensorflow::SavedModelBundle;
  TF_CHECK_OK(tensorflow::LoadSavedModel(tensorflow::SessionOptions(),
                                         tensorflow::RunOptions(), model_path,
                                         tags, bundle))
      << "Failed to load exported model from " << model_path
      << ". Ensure the model contains the required tags '" << tags_str
      << "'.\n";
  return bundle;
}

// Returns the array name without the postfix.
//
// e.g. reduces "input:0" to "input".
string GetArrayName(const string& name) {
  const std::vector<string>& names = absl::StrSplit(name, ':');
  return names[0];
}

// Returns the list of array names without the postfix sorted alphabetically.
std::set<string> GetSortedNames(const std::unordered_set<string>& names) {
  std::vector<string> final_names;
  final_names.reserve(names.size());
  for (const auto& name : names) {
    final_names.push_back(GetArrayName(name));
  }
  return std::set<string>(final_names.begin(), final_names.end());
}

// Gets the final shape after replacing the first dimension with batch size, if
// it is undefined (containing the value -1). Returns whether the shape is
// valid.
bool ReplaceShapeBatchSize(const tensorflow::TensorShapeProto& shape,
                           int batch_size,
                           tensorflow::TensorShapeProto* final_shape) {
  for (int idx = 0; idx < shape.dim().size(); ++idx) {
    int64 final_dim = shape.dim()[idx].size();
    if (final_dim == -1) {
      if (idx > 0) return false;
      final_dim = batch_size;
    }
    final_shape->add_dim()->set_size(final_dim);
  }
  return true;
}

// Updates the input arrays in ModelFlags to contain the shape of the array.
void ProcessInputShapes(const tensorflow::GraphDef& graph_def, int batch_size,
                        ModelFlags* model_flags) {
  // Build map of input array names to input arrays.
  std::unordered_map<string, InputArray*> input_data_map;
  for (auto& input : *model_flags->mutable_input_arrays()) {
    input_data_map[input.name()] = &input;
  }

  // Adds shapes to the input arrays if the shape is valid.
  for (const tensorflow::NodeDef& node_def : graph_def.node()) {
    if (input_data_map.find(node_def.name()) != input_data_map.end()) {
      const auto shape_it = node_def.attr().find("shape");
      if (shape_it != node_def.attr().end()) {
        tensorflow::TensorShapeProto final_shape;
        bool is_valid = ReplaceShapeBatchSize(shape_it->second.shape(),
                                              batch_size, &final_shape);

        if (is_valid) {
          auto* shape = input_data_map.at(node_def.name())->mutable_shape();
          QCHECK_EQ(shape->dims_size(), 0)
              << "The shape for the input '" << node_def.name()
              << "' was previously defined. For clarity please define inputs "
              << "via --input_arrays and input_shapes flags.\n";
          for (const auto& dim : final_shape.dim()) {
            shape->add_dims(dim.size());
          }
        }
      }
    }
  }

  // Checks all input arrays have a shape.
  for (auto const& input : model_flags->input_arrays()) {
    QCHECK(input.shape().dims_size() > 0)
        << "A valid input shape was not found for input '" << input.name()
        << "'. Please define via --input_arrays and --input_shapes flags.\n";
  }
}

}  // namespace

void ParseMetaData(const tensorflow::GraphDef& graph_def,
                   const std::unordered_set<string>& inputs,
                   const std::unordered_set<string>& outputs,
                   const ParsedTocoFlags& parsed_toco_flags,
                   const ParsedModelFlags& parsed_model_flags,
                   TocoFlags* toco_flags, ModelFlags* model_flags) {
  if (!parsed_model_flags.input_arrays.specified()) {
    const std::set<string> sorted_inputs = GetSortedNames(inputs);
    for (const auto& input_name : sorted_inputs) {
      model_flags->add_input_arrays()->set_name(input_name);
    }
  }

  if (!parsed_model_flags.output_arrays.specified()) {
    const std::set<string> sorted_outputs = GetSortedNames(outputs);
    for (const auto& output_name : sorted_outputs) {
      model_flags->add_output_arrays(GetArrayName(output_name));
    }
  }

  if (!parsed_model_flags.input_shapes.specified()) {
    int batch_size = parsed_model_flags.batch_size.value();
    ProcessInputShapes(graph_def, batch_size, model_flags);
  }

  if (!parsed_toco_flags.inference_type.specified()) {
    toco_flags->set_inference_type(IODataType::FLOAT);
  }
}

// TODO(nupurgarg): Add top level tests.
void GetSavedModelContents(const ParsedTocoFlags& parsed_toco_flags,
                           const ParsedModelFlags& parsed_model_flags,
                           TocoFlags* toco_flags, ModelFlags* model_flags,
                           string* graph_def_contents) {
  // Loads the MetaGraphDef within a SavedModelBundle.
  auto bundle = LoadSavedModel(parsed_toco_flags);

  // Converts the MetaGraphDef to frozen GraphDef.
  tensorflow::GraphDef frozen_graph_def;
  std::unordered_set<string> inputs;
  std::unordered_set<string> outputs;
  TF_CHECK_OK(tensorflow::FreezeSavedModel(*bundle, &frozen_graph_def, &inputs,
                                           &outputs));

  // Reads the frozen GraphDef into a string.
  QCHECK(frozen_graph_def.SerializeToString(graph_def_contents))
      << "Unable to generate serialized GraphDef.\n";

  // Process inputs and outputs and metadata within GraphDef.
  const tensorflow::GraphDef graph_def = bundle->meta_graph_def.graph_def();
  ParseMetaData(graph_def, inputs, outputs, parsed_toco_flags,
                parsed_model_flags, toco_flags, model_flags);
}

}  // namespace toco
