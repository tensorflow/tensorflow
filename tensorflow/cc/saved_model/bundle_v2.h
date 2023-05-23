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

// Helpers for loading the persistent representation of a SavedModelV2.
// Please note that this is depended on by code that does not make use of
// the full runtime and its dependencies should be restricted.

#ifndef TENSORFLOW_CC_SAVED_MODEL_BUNDLE_V2_H_
#define TENSORFLOW_CC_SAVED_MODEL_BUNDLE_V2_H_

#include <functional>
#include <memory>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/graph_debug_info.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saved_object_graph.pb.h"
#include "tensorflow/core/protobuf/trackable_object_graph.pb.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"

namespace tensorflow {

/// Represents a version 2 SavedModel that is loaded from storage (but not yet
/// loaded into an executable in-memory representation).
class SavedModelV2Bundle {
 public:
  using RestoreObjectsCallback =
      std::function<Status(int, const TrackableObjectGraph::TrackableObject&)>;

  /// Loads persistent representations for a SavedModelV2 from the specified
  /// export directory.
  static Status Load(const std::string& export_dir, SavedModelV2Bundle* bundle);

  /// MetaGraphDef from the loaded SavedModel.
  MetaGraphDef& meta_graph_def() { return meta_graph_def_; }

  /// SavedObjectGraph from the MetaGraphDef.
  const SavedObjectGraph& saved_object_graph() {
    return meta_graph_def().object_graph_def();
  }

  /// TrackableObjectGraph loaded from the variable_reader() checkpoint.
  TrackableObjectGraph& trackable_object_graph() {
    return trackable_object_graph_;
  }

  /// BundleReader for accessing the variables bundle.
  BundleReader* variable_reader() { return variable_reader_.get(); }

  /// The GraphDebugInfo (or nullptr if none).
  GraphDebugInfo* debug_info() { return debug_info_.get(); }

  /// Restores objects, invoking the callback with the node id in the
  /// saved_object_graph() and the corresponding TrackableObject from the
  /// trackable_object_graph(). The callback may use the variable_reader() but
  /// must not modify the underlying saved_object_graph().
  Status VisitObjectsToRestore(RestoreObjectsCallback callback);

 private:
  Status RecurseObjectsToRestore(
      const SavedObject* saved_object, int saved_object_node_id,
      const TrackableObjectGraph::TrackableObject* trackable_object,
      std::string object_name,
      absl::flat_hash_set<int>* seen_trackable_node_ids,
      RestoreObjectsCallback callback);

  MetaGraphDef meta_graph_def_;
  TrackableObjectGraph trackable_object_graph_;
  std::unique_ptr<BundleReader> variable_reader_;
  std::unique_ptr<GraphDebugInfo> debug_info_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CC_SAVED_MODEL_BUNDLE_V2_H_
