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

#ifndef THIRD_PARTY_TENSORFLOW_CC_SAVED_MODEL_CONSTANTS_H_
#define THIRD_PARTY_TENSORFLOW_CC_SAVED_MODEL_CONSTANTS_H_

namespace tensorflow {

// SavedModel assets directory.
constexpr char kSavedModelAssetsDirectory[] = "assets";

// SavedModel assets key for graph collection-def.
constexpr char kSavedModelAssetsKey[] = "saved_model_assets";

// SavedModel proto filename.
constexpr char kSavedModelFilenamePb[] = "saved_model.pb";

// SavedModel text format proto filename.
constexpr char kSavedModelFilenamePbTxt[] = "saved_model.pbtxt";

// SavedModel legacy init op key.
constexpr char kSavedModelLegacyInitOpKey[] = "legacy_init_op";

// Directory in which to save the SavedModel variables.
constexpr char kSavedModelVariablesDirectory[] = "variables";

// SavedModel variables filename.
constexpr char kSavedModelVariablesFilename[] = "variables";

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CC_SAVED_MODEL_CONSTANTS_H_
