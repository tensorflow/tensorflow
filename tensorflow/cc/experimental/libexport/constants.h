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
#ifndef TENSORFLOW_CC_EXPERIMENTAL_LIBEXPORT_CONSTANTS_H_
#define TENSORFLOW_CC_EXPERIMENTAL_LIBEXPORT_CONSTANTS_H_

#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace libexport {

// Subdirectory name containing the asset files.
TF_EXPORT extern const char* const kAssetsDirectory;

// Subdirectory name containing unmanaged files from higher-level APIs.
TF_EXPORT extern const char* const kExtraAssetsDirectory;

// CollectionDef key containing SavedModel assets.
TF_EXPORT extern const char* const kAssetsKey;

// CollectionDef key for the legacy init op.
TF_EXPORT extern const char* const kLegacyInitOpKey;

// CollectionDef key for the SavedModel main op.
TF_EXPORT extern const char* const kMainOpKey;

// CollectionDef key for the SavedModel train op.
// Not exported while export_all_saved_models is experimental.
TF_EXPORT extern const char* const kTrainOpKey;

// Schema version for SavedModel.
TF_EXPORT extern const int kSavedModelSchemaVersion;

// File name for SavedModel protocol buffer.
TF_EXPORT extern const char* const kSavedModelFilenamePb;

// File name for text version of SavedModel protocol buffer.
TF_EXPORT extern const char* const kSavedModelFilenamePbtxt;

// Subdirectory where debugging related files are written.
TF_EXPORT extern const char* const kDebugDirectory;

// File name for GraphDebugInfo protocol buffer which corresponds to the
// SavedModel.
TF_EXPORT extern const char* const kDebugInfoFilenamePb;

// Subdirectory name containing the variables/checkpoint files.
TF_EXPORT extern const char* const kVariablesDirectory;

// File name used for variables.
TF_EXPORT extern const char* const kVariablesFilename;

// The initialization and train ops for a MetaGraph are stored in the
// signature def map. The ops are added to the map with the following keys.
TF_EXPORT extern const char* const kInitOpSignatureKey;
TF_EXPORT extern const char* const kTrainOpSignatureKey;

}  // namespace libexport
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_EXPERIMENTAL_EXPORT_CONSTANTS_H_
