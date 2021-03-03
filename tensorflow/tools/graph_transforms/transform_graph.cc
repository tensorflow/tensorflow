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

#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/lib/strings/scanner.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/tools/graph_transforms/file_utils.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"
#if !defined(PLATFORM_WINDOWS)
#include <pwd.h>
#include <unistd.h>
#endif

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
      Scanner(remaining).AnySpace().GetResult(&remaining, &match);
      if (remaining.empty()) {
        // Nothing remains after consuming trailing spaces.
        // Consumed all transform parameter string without errors.
        return Status::OK();
      }
      // See if we have a valid transform name.
      const bool found_transform_name =
          Scanner(remaining)
              .Many(Scanner::LETTER_DIGIT_UNDERSCORE)
              .GetResult(&remaining, &transform_name);
      if (!found_transform_name) {
        return errors::InvalidArgument("Looking for transform name, but found ",
                                       string(remaining).c_str());
      }
      if (Scanner(remaining).OneLiteral("(").GetResult(&remaining, &match)) {
        state = TRANSFORM_PARAM_NAME;
      } else {
        // Add a transform with no parameters.
        params_list->push_back({string(transform_name), func_parameters});
        transform_name = "";
        state = TRANSFORM_NAME;
      }
    } else if (state == TRANSFORM_PARAM_NAME) {
      if (Scanner(remaining).OneLiteral(")").GetResult(&remaining, &match)) {
        params_list->push_back({string(transform_name), func_parameters});
        transform_name = "";
        state = TRANSFORM_NAME;
      } else {
        // Eat up any leading spaces or commas.
        Scanner(remaining).ZeroOrOneLiteral(",").GetResult(&remaining, &match);
        Scanner(remaining).AnySpace().GetResult(&remaining, &match);
        // See if we have a valid parameter name.
        const bool found_parameter_name =
            Scanner(remaining)
                .Many(Scanner::LETTER_DIGIT_UNDERSCORE)
                .GetResult(&remaining, &parameter_name);
        if (!found_parameter_name) {
          return errors::InvalidArgument(
              "Looking for parameter name, but found ",
              string(remaining).c_str());
        }
        if (Scanner(remaining).OneLiteral("=").GetResult(&remaining, &match)) {
          state = TRANSFORM_PARAM_VALUE;
        } else {
          return errors::InvalidArgument("Looking for =, but found ",
                                         string(remaining).c_str());
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
                .Many(Scanner::LETTER_DIGIT_DASH_DOT_SLASH_UNDERSCORE)
                .GetResult(&remaining, &parameter_value);
      }
      if (!found_parameter_value) {
        return errors::InvalidArgument("Looking for parameter name, but found ",
                                       string(remaining).c_str());
      }
      func_parameters[string(parameter_name)].emplace_back(parameter_value);
      // Eat up any trailing quotes.
      Scanner(remaining).ZeroOrOneLiteral("\"").GetResult(&remaining, &match);
      Scanner(remaining).ZeroOrOneLiteral("'").GetResult(&remaining, &match);
      state = TRANSFORM_PARAM_NAME;
    }
  }
  return Status::OK();
}

std::string ExpandPath(const std::string& path_string) {
#if defined(PLATFORM_WINDOWS)
  return path_string;
#else
  if (path_string.empty() || path_string[0] != '~') {
    return path_string;
  }

  const char* home = nullptr;
  std::string::size_type prefix = path_string.find_first_of('/');
  if (path_string.length() == 1 || prefix == 1) {
    // The value of $HOME, e.g., ~/foo
    home = getenv("HOME");
    if (!home) {
      // If HOME is not available, get uid
      struct passwd* pw = getpwuid(getuid());
      if (pw) {
        home = pw->pw_dir;
      }
    }
  } else {
    // The value of ~user, e.g., ~user/foo
    std::string user(path_string, 1, (prefix == std::string::npos)
                                         ? std::string::npos
                                         : prefix - 1);
    struct passwd* pw = getpwnam(user.c_str());
    if (pw) {
      home = pw->pw_dir;
    }
  }

  if (!home) {
    return path_string;
  }

  string path(home);
  if (prefix == std::string::npos) {
    return path;
  }

  if (path.length() == 0 || path[path.length() - 1] != '/') {
    path += '/';
  }
  path += path_string.substr(prefix + 1);
  return path;
#endif
}

int ParseFlagsAndTransformGraph(int argc, char* argv[], bool init_main) {
  string in_graph_string = "";
  string out_graph_string = "";
  string inputs_string = "";
  string outputs_string = "";
  string transforms_string = "";
  bool output_as_text = false;
  std::vector<Flag> flag_list = {
      Flag("in_graph", &in_graph_string, "input graph file name"),
      Flag("out_graph", &out_graph_string, "output graph file name"),
      Flag("inputs", &inputs_string, "inputs"),
      Flag("outputs", &outputs_string, "outputs"),
      Flag("transforms", &transforms_string, "list of transforms"),
      Flag("output_as_text", &output_as_text,
           "whether to write the graph in text protobuf format"),
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
  if (in_graph_string.empty()) {
    LOG(ERROR) << "in_graph graph can't be empty.\n" << usage;
    return -1;
  }
  if (out_graph_string.empty()) {
    LOG(ERROR) << "out_graph graph can't be empty.\n" << usage;
    return -1;
  }
  if (transforms_string.empty()) {
    LOG(ERROR) << "You must specify at least one transform.\n" << usage;
    return -1;
  }

  string in_graph = ExpandPath(in_graph_string);
  string out_graph = ExpandPath(out_graph_string);

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
  Status load_status = LoadTextOrBinaryGraphFile(in_graph, &graph_def);
  if (!load_status.ok()) {
    LOG(ERROR) << "Loading graph '" << in_graph_string << "' failed with "
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

  Status save_status;
  if (output_as_text) {
    save_status = WriteTextProto(Env::Default(), out_graph, graph_def);
  } else {
    save_status = WriteBinaryProto(Env::Default(), out_graph, graph_def);
  }
  if (!save_status.ok()) {
    LOG(ERROR) << "Saving graph '" << out_graph_string << "' failed with "
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
        absl::AsciiStrToLower(transform_params.at("ignore_errors").at(0));
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
    if (transform_name.empty()) {
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
    *transformed_graph_def.mutable_library() = graph_def->library();
    TF_RETURN_IF_ERROR(IsGraphValid(transformed_graph_def));

    *graph_def = transformed_graph_def;
  }
  return Status::OK();
}
}  // namespace graph_transforms
}  // namespace tensorflow
