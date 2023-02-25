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

#include <memory>

#include "fuzztest/fuzztest.h"
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow::fuzzing {
namespace {

// Fuzzer that loads an arbitrary model and performs inference using a fixed
// input.
void FuzzEndToEnd(const SavedModel& model) {
  SavedModelBundle bundle;
  SessionOptions session_options;
  RunOptions run_options;
  string export_dir = "ram://";
  TF_CHECK_OK(tsl::WriteBinaryProto(tensorflow::Env::Default(),
                                    export_dir + kSavedModelFilenamePb, model));

  Status status = LoadSavedModel(session_options, run_options, export_dir,
                                 {kSavedModelTagServe}, &bundle);
  if (!status.ok()) {
    return;
  }

  // Create a simple inference using the model.
  tensorflow::Tensor input_1(tensorflow::DT_FLOAT,
                             tensorflow::TensorShape({1, 100}));
  tensorflow::Tensor input_2(tensorflow::DT_FLOAT,
                             tensorflow::TensorShape({1, 100}));

  auto input_mat_1 = input_1.matrix<float>();
  auto input_mat_2 = input_2.matrix<float>();
  for (unsigned i = 0; i < 100; ++i) {
    input_mat_1(0, i) = 1.0;
    input_mat_2(0, i) = 2.0;
  }

  typedef std::vector<std::pair<std::string, tensorflow::Tensor>> tensor_dict;
  tensor_dict input_dict = {{input_tensor_1_name, "fuzz_arg0:0"},
                           {input_tensor_2_name, "fuzz_arg1:0"}};

  // Create output placeholder tensors for results
  std::vector<tensorflow::Tensor> outputs;
  std::vector<std::string> output_names = {"fuzz_out:0", "fuzz_out:1"};
  tensorflow::Status status_run =
      bundle.session->Run(input_dict, output_names, {}, &outputs);
}

FUZZ_TEST(End2EndFuzz, FuzzEndToEnd);

}  // namespace
}  // namespace tensorflow::fuzzing
