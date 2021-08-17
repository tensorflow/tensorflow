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
#ifndef TENSORFLOW_LITE_TESTING_TFLITE_DIFF_FLAGS_H_
#define TENSORFLOW_LITE_TESTING_TFLITE_DIFF_FLAGS_H_

#include <cstring>

#include "absl/strings/match.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/lite/testing/split.h"
#include "tensorflow/lite/testing/tflite_diff_util.h"
#include "tensorflow/lite/testing/tflite_driver.h"

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
    string delegate_name;
    string reference_tflite_model;
  } values;

  std::string delegate_name;
  std::vector<tensorflow::Flag> flags = {
      tensorflow::Flag("tensorflow_model", &values.tensorflow_model,
                       "Path of tensorflow model."),
      tensorflow::Flag("tflite_model", &values.tflite_model,
                       "Path of tensorflow lite model."),
      tensorflow::Flag("input_layer", &values.input_layer,
                       "Names of input tensors, separated by comma. Example: "
                       "input_1,input_2."),
      tensorflow::Flag("input_layer_type", &values.input_layer_type,
                       "Data types of input tensors, separated by comma. "
                       "Example: float,int."),
      tensorflow::Flag(
          "input_layer_shape", &values.input_layer_shape,
          "Shapes of input tensors, separated by colon. Example: 1,3,4,1:2."),
      tensorflow::Flag("output_layer", &values.output_layer,
                       "Names of output tensors, separated by comma. Example: "
                       "output_1,output_2."),
      tensorflow::Flag("num_runs_per_pass", &values.num_runs_per_pass,
                       "[optional] Number of full runs in each pass."),
      tensorflow::Flag("delegate", &values.delegate_name,
                       "[optional] Delegate to use for executing ops. Must be "
                       "`{\"\", NNAPI, GPU, FLEX}`"),
      tensorflow::Flag("reference_tflite_model", &values.reference_tflite_model,
                       "[optional] Path of the TensorFlow Lite model to "
                       "compare inference results against the model given in "
                       "`tflite_model`."),
  };

  bool no_inputs = *argc == 1;
  bool success = tensorflow::Flags::Parse(argc, argv, flags);
  if (!success || no_inputs || (*argc == 2 && !strcmp(argv[1], "--helpfull"))) {
    fprintf(stderr, "%s", tensorflow::Flags::Usage(argv[0], flags).c_str());
    return {};
  } else if (values.tensorflow_model.empty() || values.tflite_model.empty() ||
             values.input_layer.empty() || values.input_layer_type.empty() ||
             values.input_layer_shape.empty() || values.output_layer.empty()) {
    fprintf(stderr, "%s", tensorflow::Flags::Usage(argv[0], flags).c_str());
    return {};
  }

  TfLiteDriver::DelegateType delegate = TfLiteDriver::DelegateType::kNone;
  if (!values.delegate_name.empty()) {
    if (absl::EqualsIgnoreCase(values.delegate_name, "nnapi")) {
      delegate = TfLiteDriver::DelegateType::kNnapi;
    } else if (absl::EqualsIgnoreCase(values.delegate_name, "gpu")) {
      delegate = TfLiteDriver::DelegateType::kGpu;
    } else if (absl::EqualsIgnoreCase(values.delegate_name, "flex")) {
      delegate = TfLiteDriver::DelegateType::kFlex;
    } else {
      fprintf(stderr, "%s", tensorflow::Flags::Usage(argv[0], flags).c_str());
      return {};
    }
  }

  return {values.tensorflow_model,
          values.tflite_model,
          Split<string>(values.input_layer, ","),
          Split<string>(values.input_layer_type, ","),
          Split<string>(values.input_layer_shape, ":"),
          Split<string>(values.output_layer, ","),
          values.num_runs_per_pass,
          delegate,
          values.reference_tflite_model};
}

}  // namespace testing
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TESTING_TFLITE_DIFF_FLAGS_H_
