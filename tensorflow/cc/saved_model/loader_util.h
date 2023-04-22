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

#ifndef TENSORFLOW_CC_SAVED_MODEL_LOADER_UTIL_H_
#define TENSORFLOW_CC_SAVED_MODEL_LOADER_UTIL_H_

#include <string>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace tensorflow {
namespace internal {

// A SavedModel may store the name of the initialization op to run in the
// in the SignatureDef (v2) or a collection (v1). If an init_op collection
// exists, then the collection must contain exactly one op.
Status GetInitOp(const string& export_dir, const MetaGraphDef& meta_graph_def,
                 string* init_op_name);

Status GetAssetFileDefs(const MetaGraphDef& meta_graph_def,
                        std::vector<AssetFileDef>* asset_file_defs);

}  // namespace internal
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_SAVED_MODEL_LOADER_UTIL_H_
