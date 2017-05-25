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

%include <std_string.i>
%include "tensorflow/python/lib/core/strings.i"
%include "tensorflow/python/platform/base.i"

%{
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/stat_summarizer.h"
#include "tensorflow/python/lib/core/py_func.h"

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/tools/graph_transforms/transform_graph.h"
%}

%ignoreall

%unignore tensorflow;
%unignore TransformGraphWithStringInputs;


%{
string TransformGraphWithStringInputs(string graph_def_string,
                                      string inputs_string,
                                      string outputs_string,
                                      string transforms_string,
                                      TF_Status* out_status) {
  tensorflow::GraphDef graph_def;
  if (!graph_def.ParseFromString(graph_def_string)) {
    Set_TF_Status_from_Status(out_status, tensorflow::errors::InvalidArgument(
        "Couldn't interpret input as a GraphDef"));
    return "";
  }

  tensorflow::graph_transforms::TransformParameters params_list;
  tensorflow::Status parse_status =
      tensorflow::graph_transforms::ParseTransformParameters(
          transforms_string, &params_list);
  if (!parse_status.ok()) {
    tensorflow::Set_TF_Status_from_Status(out_status, parse_status);
    return "";
  }
  std::vector<string> inputs = tensorflow::str_util::Split(inputs_string, ',');
  std::vector<string> outputs =
      tensorflow::str_util::Split(outputs_string, ',');

  tensorflow::Status transform_status =
      tensorflow::graph_transforms::TransformGraph(
          inputs, outputs, params_list, &graph_def);
  if (!transform_status.ok()) {
    tensorflow::Set_TF_Status_from_Status(out_status, transform_status);
    return "";
  }
  string result;
  if (!graph_def.SerializeToString(&result)) {
    Set_TF_Status_from_Status(out_status, tensorflow::errors::InvalidArgument(
        "Couldn't serialize output as a GraphDef"));
    return "";
  }
  Set_TF_Status_from_Status(out_status, tensorflow::Status::OK());
  return result;
}
%}


string TransformGraphWithStringInputs(string graph_def_string,
                                      string inputs_string,
                                      string outputs_string,
                                      string transforms_string,
                                      TF_Status* out_status);

%unignoreall
