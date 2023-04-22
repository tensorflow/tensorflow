/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/cc/experimental/libexport/constants.h"

namespace tensorflow {
namespace libexport {

const char* const kAssetsDirectory = "assets";
const char* const kExtraAssetsDirectory = "assets.extra";
const char* const kAssetsKey = "saved_model_assets";
const char* const kLegacyInitOpKey = "legacy_init_op";
const char* const kMainOpKey = "saved_model_main_op";
const char* const kTrainOpKey = "saved_model_train_op";
const int kSavedModelSchemaVersion = 1;
const char* const kSavedModelFilenamePb = "saved_model.pb";
const char* const kSavedModelFilenamePbtxt = "saved_model.pbtxt";
const char* const kDebugDirectory = "debug";
const char* const kDebugInfoFilenamePb = "saved_model_debug_info.pb";
const char* const kVariablesDirectory = "variables";
const char* const kVariablesFilename = "variables";
const char* const kInitOpSignatureKey = "__saved_model_init_op";
const char* const kTrainOpSignatureKey = "__saved_model_train_op";

}  // namespace libexport
}  // namespace tensorflow
