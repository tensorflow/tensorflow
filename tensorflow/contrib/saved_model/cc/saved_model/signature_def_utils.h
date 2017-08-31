/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// Helpers for working with the SignatureDefs of TensorFlow SavedModels.

#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_SAVED_MODEL_CC_SAVED_MODEL_SIGNATURE_DEF_UTILS_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_SAVED_MODEL_CC_SAVED_MODEL_SIGNATURE_DEF_UTILS_H_

#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace tensorflow {

// Finds the entry in meta_graph_def.signature_def with the given key, or
// returns NotFound and leaves *signature_def unchanged. NOTE: The output
// SignatureDef* points into meta_graph_def and may be invalidated by changes
// to that protocol buffer, as usual.
Status FindSignatureDefByKey(const MetaGraphDef& meta_graph_def,
                             const string& signature_def_key,
                             const SignatureDef** signature_def);

// Finds the entry in signature_def.inputs with the given key, or returns
// NotFound and leaves *tensor_info unchanged. NOTE: The output TensorInfo*
// points into signature_def and may be invalidated by changes to that protocol
// buffer, as usual.
Status FindInputTensorInfoByKey(const SignatureDef& signature_def,
                                const string& tensor_info_key,
                                const TensorInfo** tensor_info);

// Finds the entry in signature_def.outputs with the given key, or returns
// NotFound and leaves *tensor_info unchanged. NOTE: The output TensorInfo*
// points into signature_def and may be invalidated by changes to that protocol
// buffer, as usual.
Status FindOutputTensorInfoByKey(const SignatureDef& signature_def,
                                 const string& tensor_info_key,
                                 const TensorInfo** tensor_info);

// Finds the entry in signature_def.inputs with the given key and copies out
// the name of this Tensor in the graph, or returns NotFound and leaves *name
// unchanged.
Status FindInputTensorNameByKey(const SignatureDef& signature_def,
                                const string& tensor_info_key, string* name);

// Finds the entry in signature_def.outputs with the given key and copies out
// the name of this Tensor in the graph, or returns NotFound and leaves *name
// unchanged.
Status FindOutputTensorNameByKey(const SignatureDef& signature_def,
                                 const string& tensor_info_key, string* name);

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_SAVED_MODEL_CC_SAVED_MODEL_SIGNATURE_DEF_UTILS_H_
