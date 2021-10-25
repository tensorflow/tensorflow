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
#ifndef TENSORFLOW_CC_EXPERIMENTAL_LIBTF_MODULE_H_
#define TENSORFLOW_CC_EXPERIMENTAL_LIBTF_MODULE_H_

#include "tensorflow/cc/experimental/libexport/load.h"
#include "tensorflow/cc/experimental/libtf/runtime/runtime.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/protobuf/saved_object_graph.pb.h"

namespace tf {
namespace libtf {
namespace impl {

// The main interface for taking a serialized saved model and getting back a
// fully-built model.
//
// Implementation steps:
//
//   1) For each function def in the SavedModel, register it with the runtime.
//   2) For each object in the object graph def, build it.
//   3) For each variable stored in the checkpoint in the SavedModel,
//      restore it, and attach it to the associated variable object.
//   4) For each polymorphic function, associate it with the appropriate
//      concrete function(s).
//   5) For each function with captures, bind the appropriate objects as
//      captured inputs.
//   6) Take the fully-prepared objects, and build them into a hierarchy.
//   7) Return the prepared model.

// Converts a SavedUserObject into its corresponding data structure.
// TODO(b/185579152): This method returns empty data structures currently.
tensorflow::StatusOr<Handle> BuildSavedUserObject(
    tensorflow::SavedObject saved_object_proto);

// "Build" all SavedObjects, ie convert from proto to their runtime
// representation, in the tf_package.
tensorflow::StatusOr<std::vector<Handle>> BuildObjects(
    tensorflow::libexport::TFPackage tf_package);

// Convert tf_package to a program in the runtime.
tensorflow::StatusOr<Handle> BuildProgram(
    runtime::Runtime runtime, tensorflow::libexport::TFPackage tf_package);

}  // namespace impl
}  // namespace libtf
}  // namespace tf

#endif  // TENSORFLOW_CC_EXPERIMENTAL_LIBTF_MODULE_H_
