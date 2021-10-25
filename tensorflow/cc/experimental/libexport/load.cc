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
#include "tensorflow/cc/experimental/libexport/load.h"

#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

#define RETURN_IF_ERROR(s) \
  {                        \
    auto c = (s);          \
    if (!c.ok()) return c; \
  }

namespace tensorflow {
namespace libexport {

using protobuf::RepeatedPtrField;
using CheckpointKey = TFPackage::CheckpointKey;

tensorflow::StatusOr<TFPackage> TFPackage::Load(const std::string& path) {
  // Load the proto
  TFPackage tf_package;
  const string saved_model_pb_path = io::JoinPath(path, kSavedModelFilenamePb);
  const string saved_model_pbtxt_path =
      io::JoinPath(path, kSavedModelFilenamePbTxt);
  if (Env::Default()->FileExists(saved_model_pb_path).ok()) {
    RETURN_IF_ERROR(ReadBinaryProto(Env::Default(), saved_model_pb_path,
                                    &tf_package.saved_model_proto_));
  } else if (Env::Default()->FileExists(saved_model_pbtxt_path).ok()) {
    RETURN_IF_ERROR(ReadTextProto(Env::Default(), saved_model_pbtxt_path,
                                  &tf_package.saved_model_proto_));
  } else {
    return Status(error::Code::NOT_FOUND,
                  "Could not find SavedModel .pb or .pbtxt at supplied export "
                  "directory path: " +
                      path);
  }
  return tf_package;
}

tensorflow::StatusOr<std::vector<CheckpointKey>>
TFPackage::GetVariableCheckpointKeys() {
  return errors::Unimplemented("GetVariableCheckpointKeys not implemented.");
}

const SavedObjectGraph& TFPackage::GetObjectGraph() {
  return saved_model_proto_.mutable_meta_graphs(0)->object_graph_def();
}

const RepeatedPtrField<FunctionDef>& TFPackage::GetFunctionDefs() {
  auto& function_library =
      saved_model_proto_.mutable_meta_graphs(0)->graph_def().library();
  return function_library.function();
}

}  // namespace libexport
}  // namespace tensorflow
