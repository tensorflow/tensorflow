/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/tools/graph_transforms/transform_graph.h"

#include "tensorflow/core/lib/strings/scanner.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

using tensorflow::strings::Scanner;

Status ParseTransformParameters(const string& transforms_string,
                                TransformParameters* params_list) {
  params_list->clear();
  enum {
    TRANSFORM_NAME,
    TRANSFORM_PARAM_NAME,
    TRANSFORM_PARAM_VALUE,
  } state = TRANSFORM_NAME;
  StringPiece remaining(transforms_string);
  StringPiece match;
  StringPiece transform_name;
  StringPiece parameter_name;
  StringPiece parameter_value;
  TransformFuncParameters func_parameters;
  while (!remaining.empty()) {
    if (state == TRANSFORM_NAME) {
      // Reset the list of parameters.
      func_parameters.clear();
      // Eat up any leading spaces.
      Scanner(remaining).Any(Scanner::SPACE).GetResult(&remaining, &match);
      // See if we have a valid transform name.
      const bool found_transform_name =
          Scanner(remaining)
              .Any(Scanner::LETTER_DIGIT_UNDERSCORE)
              .GetResult(&remaining, &transform_name);
      if (!found_transform_name) {
        return errors::InvalidArgument("Looking for transform name, but found ",
                                       remaining.ToString().c_str());
      }
      if (Scanner(remaining).OneLiteral("(").GetResult(&remaining, &match)) {
        state = TRANSFORM_PARAM_NAME;
      } else {
        // Add a transform with no parameters.
        params_list->push_back({transform_name.ToString(), func_parameters});
        transform_name = "";
        state = TRANSFORM_NAME;
      }
    } else if (state == TRANSFORM_PARAM_NAME) {
      if (Scanner(remaining).OneLiteral(")").GetResult(&remaining, &match)) {
        params_list->push_back({transform_name.ToString(), func_parameters});
        transform_name = "";
        state = TRANSFORM_NAME;
      } else {
        // Eat up any leading spaces or commas.
        Scanner(remaining).ZeroOrOneLiteral(",").GetResult(&remaining, &match);
        Scanner(remaining).Any(Scanner::SPACE).GetResult(&remaining, &match);
        // See if we have a valid parameter name.
        const bool found_parameter_name =
            Scanner(remaining)
                .Any(Scanner::LETTER_DIGIT_UNDERSCORE)
                .GetResult(&remaining, &parameter_name);
        if (!found_parameter_name) {
          return errors::InvalidArgument(
              "Looking for parameter name, but found ",
              remaining.ToString().c_str());
        }
        if (Scanner(remaining).OneLiteral("=").GetResult(&remaining, &match)) {
          state = TRANSFORM_PARAM_VALUE;
        } else {
          return errors::InvalidArgument("Looking for =, but found ",
                                         remaining.ToString().c_str());
        }
      }
    } else if (state == TRANSFORM_PARAM_VALUE) {
      bool found_parameter_value;
      // Deal with quoted values.
      if (Scanner(remaining).OneLiteral("\"").GetResult(&remaining, &match)) {
        found_parameter_value =
            Scanner(remaining).ScanEscapedUntil('"').GetResult(
                &remaining, &parameter_value);
        if (found_parameter_value) {
          Scanner(remaining).OneLiteral("\"").GetResult(&remaining, &match);
        }
      } else {
        // See if we have a valid parameter name.
        found_parameter_value =
            Scanner(remaining)
                .Any(Scanner::LETTER_DIGIT_DASH_DOT_SLASH_UNDERSCORE)
                .GetResult(&remaining, &parameter_value);
      }
      if (!found_parameter_value) {
        return errors::InvalidArgument("Looking for parameter name, but found ",
                                       remaining.ToString().c_str());
      }
      func_parameters[parameter_name.ToString()].push_back(
          parameter_value.ToString());
      // Eat up any trailing quotes.
      Scanner(remaining).ZeroOrOneLiteral("\"").GetResult(&remaining, &match);
      Scanner(remaining).ZeroOrOneLiteral("'").GetResult(&remaining, &match);
      state = TRANSFORM_PARAM_NAME;
    }
  }
  return Status::OK();
}

int ParseFlagsAndTransformGraph(int argc, char* argv[], bool init_main) {
  string in_graph = "";
  string out_graph = "";
  string inputs_string = "";
  string outputs_string = "";
  string transforms_string = "";
  std::vector<Flag> flag_list = {
      Flag("in_graph", &in_graph, "input graph file name"),
      Flag("out_graph", &out_graph, "output graph file name"),
      Flag("inputs", &inputs_string, "inputs"),
      Flag("outputs", &outputs_string, "outputs"),
      Flag("transforms", &transforms_string, "list of transforms"),
  };
  string usage = Flags::Usage(argv[0], flag_list);
  usage += "\nTransforms are:\n";
  TransformRegistry* transform_registry = GetTransformRegistry();
  for (const auto& pair : *transform_registry) {
    usage += pair.first + "\n";
  }

  const bool parse_result = Flags::Parse(&argc, argv, flag_list);
  // We need to call this to set up global state for TensorFlow.
  if (init_main) {
    port::InitMain(argv[0], &argc, &argv);
  }
  if (!parse_result) {
    LOG(ERROR) << usage;
    return -1;
  }
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << ".\n" << usage;
    return -1;
  }
  if (in_graph.empty()) {
    LOG(ERROR) << "in_graph graph can't be empty.\n" << usage;
    return -1;
  }
  if (out_graph.empty()) {
    LOG(ERROR) << "out_graph graph can't be empty.\n" << usage;
    return -1;
  }
  if (transforms_string.empty()) {
    LOG(ERROR) << "You must specify at least one transform.\n" << usage;
    return -1;
  }

  std::vector<string> inputs = str_util::Split(inputs_string, ',');
  std::vector<string> outputs = str_util::Split(outputs_string, ',');
  TransformParameters transform_params;
  Status parse_status =
      ParseTransformParameters(transforms_string, &transform_params);
  if (!parse_status.ok()) {
    LOG(ERROR) << "Failed to parse --transform argument, error was "
               << parse_status.error_message();
    return -1;
  }
  if (transform_params.empty()) {
    LOG(ERROR) << "You must specify at least one transform.\n" << usage;
    return -1;
  }

  GraphDef graph_def;
  Status load_status = ReadBinaryProto(Env::Default(), in_graph, &graph_def);
  if (!load_status.ok()) {
    LOG(ERROR) << "Loading graph '" << in_graph << "' failed with "
               << load_status.error_message();
    LOG(ERROR) << usage;
    return -1;
  }

  Status transform_result =
      TransformGraph(inputs, outputs, transform_params, &graph_def);

  if (!transform_result.ok()) {
    LOG(ERROR) << transform_result.error_message();
    LOG(ERROR) << usage;
    return -1;
  }

  Status save_status = WriteBinaryProto(Env::Default(), out_graph, graph_def);
  if (!save_status.ok()) {
    LOG(ERROR) << "Saving graph '" << out_graph << "' failed with "
               << save_status.error_message();
    return -1;
  }

  return 0;
}

Status ShouldIgnoreErrors(const TransformFuncParameters& transform_params,
                          bool* ignore_errors) {
  *ignore_errors = false;
  if (transform_params.count("ignore_errors") &&
      (!transform_params.at("ignore_errors").empty())) {
    const string& ignore_errors_string =
        str_util::Lowercase(transform_params.at("ignore_errors").at(0));
    if (ignore_errors_string == "true") {
      *ignore_errors = true;
    } else if (ignore_errors_string == "false") {
      *ignore_errors = false;
    } else {
      return errors::InvalidArgument(
          "ignore_errors should be true or false, found ",
          ignore_errors_string);
    }
  }
  return Status::OK();
}

Status TransformGraph(const std::vector<string>& inputs,
                      const std::vector<string>& outputs,
                      const TransformParameters& transform_params,
                      GraphDef* graph_def) {
  TransformRegistry* transform_registry = GetTransformRegistry();
  for (const auto& transform_info : transform_params) {
    const string& transform_name = transform_info.first;
    if (transform_name == "") {
      continue;
    }
    if (!transform_registry->count(transform_name)) {
      return errors::InvalidArgument("Transform '", transform_name,
                                     "' not recognized.");
    }
    LOG(INFO) << "Applying " << transform_name;
    const TransformFunc& transform_func =
        transform_registry->at(transform_name);
    TransformFuncContext context;
    context.input_names = inputs;
    context.output_names = outputs;
    context.params = transform_info.second;
    bool ignore_errors;
    TF_RETURN_IF_ERROR(
        ShouldIgnoreErrors(transform_info.second, &ignore_errors));
    GraphDef transformed_graph_def;
    Status transform_result =
        transform_func(*graph_def, context, &transformed_graph_def);
    if (!transform_result.ok()) {
      if (ignore_errors) {
        LOG(ERROR) << transform_name << ": Ignoring error "
                   << transform_result.error_message();
        transformed_graph_def = *graph_def;
      } else {
        return transform_result;
      }
    }
    // Copy over the library from the original input graph.
    transformed_graph_def.mutable_library()->CopyFrom(graph_def->library());
    TF_RETURN_IF_ERROR(IsGraphValid(transformed_graph_def));

    *graph_def = transformed_graph_def;
  }
  return Status::OK();
}
}  // namespace graph_transforms
}  // namespace tensorflow
