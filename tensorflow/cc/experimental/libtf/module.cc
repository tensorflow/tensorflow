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
#include "tensorflow/cc/experimental/libtf/module.h"

#include <string>

#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/protobuf/saved_object_graph.pb.h"
namespace tf {
namespace libtf {
namespace impl {

using tensorflow::libexport::TFPackage;
using tf::libtf::runtime::Runtime;

// TODO(danielellis): Fill in with implementations.

// Builds a vector of runtime representations of `SavedObject`s from a
// SavedModel. These are returned as a flat list.  The full hierarchy building
// and initialization should be done in a later pass.
tensorflow::StatusOr<std::vector<Handle>> BuildObjects(TFPackage tf_package) {
  std::vector<Handle> objects;
  const tensorflow::SavedObjectGraph object_graph = tf_package.GetObjectGraph();
  for (auto& node : object_graph.nodes()) {
    if (node.kind_case() == tensorflow::SavedObject::kUserObject) {
      tensorflow::StatusOr<Handle> result = BuildSavedUserObject(node);
      if (result.ok()) {
        objects.push_back(*result);
      } else {
        return result.status();
      }
    }
  }
  return objects;
}

tensorflow::StatusOr<Handle> BuildSavedUserObject(
    tensorflow::SavedObject saved_object_proto) {
  if (saved_object_proto.kind_case() != tensorflow::SavedObject::kUserObject) {
    return tensorflow::errors::InvalidArgument("Not a UserObject.");
  }

  std::string identifier = saved_object_proto.user_object().identifier();
  if (identifier == "trackable_list_wrapper") {
    tf::libtf::List user_list;
    // TODO(b/191267013): Populate with values.
    return user_list;
  }
  if (identifier == "trackable_dict_wrapper") {
    tf::libtf::Dictionary user_dict;
    // TODO(b/191267013): Populate with values.
    return user_dict;
  }
  if (identifier == "signature_map") {
    tf::libtf::Dictionary signature_map;
    // TODO(b/191267013): Populate with values.
    return signature_map;
  }
  if (identifier == "_generic_user_object") {
    tf::libtf::Dictionary user_object;
    // TODO(b/191267013): Populate with values.
    return user_object;
  }
  return tensorflow::errors::Unimplemented(absl::StrCat(
      "UserObject with identifier '", identifier, "' not implemented."));
}

// Register all available concrete functions from a SavedModel into a runtime.
tensorflow::Status RegisterConcreteFunctions(Runtime runtime,
                                             TFPackage tf_package) {
  return tensorflow::errors::Unimplemented("Not implemented.");
}

// Initialize any variables found in the SavedModel and attach them to the
// appropriate object representation in the runtime.
tensorflow::Status InitializeVariables(Runtime runtime, TFPackage tf_package,
                                       std::vector<Handle> objects) {
  return tensorflow::errors::Unimplemented("Not implemented.");
}

// Register concrete functions with their associated polymorphic functions.
tensorflow::Status SetupPolymorphicFunctions(Runtime runtime,
                                             TFPackage tf_package,
                                             std::vector<Handle> objects) {
  return tensorflow::errors::Unimplemented("Not implemented.");
}

// Register any captures with their associated higher-level functions.
tensorflow::Status SetupFunctionCaptures(Runtime runtime, TFPackage tf_package,
                                         std::vector<Handle> objects) {
  return tensorflow::errors::Unimplemented("Not implemented.");
}

// Takes a flat list of Handles and builds them into the hierarchical
// representation defined by the SavedModel.
tensorflow::StatusOr<Handle> BuildObjectHierarchy(TFPackage tf_package,
                                                  std::vector<Handle> objects) {
  return tensorflow::errors::Unimplemented("Not implemented.");
}

tensorflow::StatusOr<Handle> BuildProgram(Runtime runtime,
                                          TFPackage tf_package) {
  return tensorflow::errors::Unimplemented("Not implemented.");
}

}  // namespace impl
}  // namespace libtf
}  // namespace tf
