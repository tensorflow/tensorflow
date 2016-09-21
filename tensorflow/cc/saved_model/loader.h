/* Copyright 2016 Google Inc. All Rights Reserved.

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

// SavedModel loading functions and SavedModelBundle struct.

#ifndef THIRD_PARTY_TENSORFLOW_CC_SAVED_MODEL_LOADER_H_
#define THIRD_PARTY_TENSORFLOW_CC_SAVED_MODEL_LOADER_H_

#include <string>
#include <unordered_set>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {

// SavedModel representation once the SavedModel is loaded from storage.
struct SavedModelBundle {
  std::unique_ptr<Session> session;
  MetaGraphDef meta_graph_def;
};

// Loads a SavedModel from the specified export directory. The meta graph def to
// be loaded is identified by the supplied tags, corresponding exactly to the
// set of tags used at SavedModel build time. Returns a SavedModel bundle with a
// session and the requested meta graph def, if found.
Status LoadSavedModel(const string& export_dir,
                      const std::unordered_set<string>& tags,
                      const SessionOptions& session_options,
                      const RunOptions& run_options,
                      SavedModelBundle* const bundle);

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CC_SAVED_MODEL_LOADER_H_
