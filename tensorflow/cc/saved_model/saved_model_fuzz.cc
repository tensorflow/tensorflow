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

void FuzzLoadSavedModel(const SavedModel& model) {
  SavedModelBundleLite bundle;
  SessionOptions session_options;
  RunOptions run_options;

  string export_dir = "ram://";
  TF_CHECK_OK(tsl::WriteBinaryProto(tensorflow::Env::Default(),
                                    export_dir + kSavedModelFilenamePb, model));

  LoadSavedModel(session_options, run_options, export_dir,
                 {kSavedModelTagServe}, &bundle)
      .IgnoreError();
}
FUZZ_TEST(SavedModelFuzz, FuzzLoadSavedModel);

}  // namespace
}  // namespace tensorflow::fuzzing
