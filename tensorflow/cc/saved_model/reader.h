/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

/// Functions to read the SavedModel proto, or parts of it.

#ifndef TENSORFLOW_CC_SAVED_MODEL_READER_H_
#define TENSORFLOW_CC_SAVED_MODEL_READER_H_

#include <memory>
#include <unordered_set>

#include "absl/status/statusor.h"
#include "tensorflow/core/framework/graph_debug_info.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"

namespace tensorflow {
Status ReadSavedModel(absl::string_view export_dir,
                      SavedModel* saved_model_proto);

// Finds and returns the MetaGraphDef (within the provided SavedModel) that
// matches the given set of tags. The lifetime of the returned MetaGraphDef is
// the same as the lifetime of `saved_model_proto`.
//
// FindMetaGraphDef returns a failure status when no MetaGraphDef matches the
// provided tags.
absl::StatusOr<MetaGraphDef*> FindMetaGraphDef(
    const std::unordered_set<string>& tags, SavedModel* saved_model_proto);

// Reads the SavedModel proto from saved_model.pb(txt) in the given directory,
// finds the MetaGraphDef that matches the given set of tags and writes it to
// the `meta_graph_def` parameter. Returns a failure status when the SavedModel
// file does not exist or no MetaGraphDef matches the tags.
Status ReadMetaGraphDefFromSavedModel(absl::string_view export_dir,
                                      const std::unordered_set<string>& tags,
                                      MetaGraphDef* meta_graph_def);

// Store debug info from the SavedModel export dir.
Status ReadSavedModelDebugInfoIfPresent(
    absl::string_view export_dir,
    std::unique_ptr<GraphDebugInfo>* debug_info_proto);

}  // namespace tensorflow

#endif  // TENSORFLOW_CC_SAVED_MODEL_READER_H_
