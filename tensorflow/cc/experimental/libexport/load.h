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
#ifndef TENSORFLOW_CC_EXPERIMENTAL_LIBEXPORT_LOAD_H_
#define TENSORFLOW_CC_EXPERIMENTAL_LIBEXPORT_LOAD_H_

#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"
#include "tensorflow/core/protobuf/saved_object_graph.pb.h"

namespace tensorflow {
namespace libexport {

// A low-level representation of a SavedModel.
//
// This class should only ever be a thin wrapper around disk (or other storage)
// access for a SavedModel.  Higher level functionality should be layered on top
// by other functions and classes.
//
// In the future, this class can also provide a mechanism for automatic version
// migration.  This will allow the calling code to always work against the most
// recent version of SavedModel.
class TFPackage {
 public:
  // Load a SavedModel, parsing the associated protobuf for later access.
  static tensorflow::StatusOr<TFPackage> Load(const std::string& path);

  // Reads and returns a list of variable checkpoint keys found in the
  // SavedModel.
  //
  // RestoreV2 is the operation that will ultimately be responsible for reading
  // and restoring the variable(s)' values.  Variable values are indexed in the
  // checkpoint files by "checkpoint keys".  These keys along with dtype and
  // shape / slice information allow RestoreV2 to look up a variable's value in
  // the SavedModel and restore it into a tensor.
  //
  // In an ideal world, we wouldn't need this extra layer of indirection; this
  // class would be responsible for reading the values and providing them to the
  // caller for registration in the runtime.  We should explore whether that is
  // feasible and migrate to it if possible.
  //
  // Regardless of what we decide to do, we should eventually split this out
  // into its own checkpoint abstraction.
  struct CheckpointKey {
    std::string key;
    DataType dtype;
    // Use an empty string for a non-partitioned variable.
    //
    // TODO(danielellis): Create a better description around what valid values
    // look like for this.
    std::string shape_and_slice;
  };
  tensorflow::StatusOr<std::vector<CheckpointKey>> GetVariableCheckpointKeys();

  // Retrieves the object graph from the SavedModel.
  //
  // For now, we're returning the object graph directly (i.e. the parsed proto)
  // rather than adding abstraction on top.  We may later find we would like an
  // intermediate abstraction layer to make traversal easier, but for now the
  // extra complexity doesn't seem justified.  Regardless of what we choose,
  // that logic should live outside this class; this class should continue to
  // have the clearly-defined, singular responsibility of reading and parsing
  // the low-level, serialized format.
  const SavedObjectGraph& GetObjectGraph();

  // Returns a list of function defs in the SavedModel.
  const protobuf::RepeatedPtrField<FunctionDef>& GetFunctionDefs();

 private:
  SavedModel saved_model_proto_;
};

}  // namespace libexport
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_EXPERIMENTAL_LIBEXPORT_LOAD_H_
