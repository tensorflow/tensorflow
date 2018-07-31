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
#ifndef TENSORFLOW_CONTRIB_LITE_TESTING_TFLITE_DIFF_FLAGS_H_
#define TENSORFLOW_CONTRIB_LITE_TESTING_TFLITE_DIFF_FLAGS_H_

#include <cstring>

#include "tensorflow/contrib/lite/testing/split.h"
#include "tensorflow/contrib/lite/testing/tflite_diff_util.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace tflite {
namespace testing {

DiffOptions ParseTfliteDiffFlags(int* argc, char** argv) {
  struct {
    string tensorflow_model;
    string tflite_model;
    string input_layer;
    string input_layer_type;
    string input_layer_shape;
    string output_layer;
    int32_t num_runs_per_pass = 100;
  } values;

  std::vector<tensorflow::Flag> flags = {
      tensorflow::Flag("tensorflow_model", &values.tensorflow_model,
                       "Path of tensorflow model."),
      tensorflow::Flag("tflite_model", &values.tflite_model,
                       "Path of tensorflow lite model."),
      tensorflow::Flag("input_layer", &values.input_layer,
                       "Names of input tensors, separated by comma. Example: "
                       "input_1,input_2"),
      tensorflow::Flag("input_layer_type", &values.input_layer_type,
                       "Data types of input tensors, separated by comma. "
                       "Example: float,int"),
      tensorflow::Flag(
          "input_layer_shape", &values.input_layer_shape,
          "Shapes of input tensors, separated by colon. Example: 1,3,4,1:2"),
      tensorflow::Flag("output_layer", &values.output_layer,
                       "Names of output tensors, separated by comma. Example "
                       "output_1,output_2"),
      tensorflow::Flag("num_runs_per_pass", &values.num_runs_per_pass,
                       "Number of full runs in each pass."),
  };

  bool no_inputs = *argc == 1;
  bool success = tensorflow::Flags::Parse(argc, argv, flags);
  if (!success || no_inputs || (*argc == 2 && !strcmp(argv[1], "--helpfull"))) {
    fprintf(stderr, "%s", tensorflow::Flags::Usage(argv[0], flags).c_str());
    return {};
  }

  return {values.tensorflow_model,
          values.tflite_model,
          Split<string>(values.input_layer, ","),
          Split<string>(values.input_layer_type, ","),
          Split<string>(values.input_layer_shape, ":"),
          Split<string>(values.output_layer, ","),
          values.num_runs_per_pass};
}

}  // namespace testing
}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_TESTING_TFLITE_DIFF_FLAGS_H_
