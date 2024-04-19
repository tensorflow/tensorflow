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
#include <string>
#include <utility>
#include <vector>

#include "fuzztest/fuzztest.h"
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/security/fuzzing/cc/core/framework/datatype_domains.h"
#include "tensorflow/security/fuzzing/cc/core/framework/tensor_domains.h"
#include "tensorflow/security/fuzzing/cc/core/framework/tensor_shape_domains.h"
#include "tsl/platform/env.h"
#include "tsl/platform/status.h"

namespace tensorflow::fuzzing {
namespace {

// Fuzzer that loads an arbitrary model and performs inference using a fixed
// input.
void FuzzEndToEnd(
    const SavedModel& model,
    const std::vector<std::pair<std::string, tensorflow::Tensor>>& input_dict) {
  SavedModelBundle bundle;
  const SessionOptions session_options;
  const RunOptions run_options;
  const std::string export_dir = "ram://";
  TF_CHECK_OK(tsl::WriteBinaryProto(tensorflow::Env::Default(),
                                    export_dir + kSavedModelFilenamePb, model));

  Status status = LoadSavedModel(session_options, run_options, export_dir,
                                 {kSavedModelTagServe}, &bundle);
  if (!status.ok()) {
    return;
  }

  // Create output placeholder tensors for results
  std::vector<tensorflow::Tensor> outputs;
  std::vector<std::string> output_names = {"fuzz_out:0", "fuzz_out:1"};
  tensorflow::Status status_run =
      bundle.session->Run(input_dict, output_names, {}, &outputs);
}

FUZZ_TEST(End2EndFuzz, FuzzEndToEnd)
    .WithDomains(
        fuzztest::Arbitrary<SavedModel>(),
        fuzztest::VectorOf(fuzztest::PairOf(fuzztest::Arbitrary<std::string>(),
                                            fuzzing::AnyValidNumericTensor(
                                                fuzzing::AnyValidTensorShape(
                                                    /*max_rank=*/3,
                                                    /*dim_lower_bound=*/0,
                                                    /*dim_upper_bound=*/20),
                                                fuzzing::AnyValidDataType())))
            .WithMaxSize(6));

}  // namespace
}  // namespace tensorflow::fuzzing
