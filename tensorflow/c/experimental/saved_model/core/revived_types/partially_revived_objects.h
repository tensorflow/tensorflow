/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_REVIVED_TYPES_PARTIALLY_REVIVED_OBJECTS_H_
#define TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_REVIVED_TYPES_PARTIALLY_REVIVED_OBJECTS_H_

#include <memory>

#include "tensorflow/c/eager/immediate_execution_context.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/asset.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/constant.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/restored_resource_revival_state.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/revived_objects.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/tf_concrete_function_revival_state.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/tf_signature_def_function_revival_state.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/variable.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/saved_object_graph.pb.h"

namespace tensorflow {

// Container for objects during the revival step in SavedModel's loading.
// Notably, resources and functions can be in a state where they reference
// other resources/functions that have not been constructed yet. We collect
// *all* objects in a partially valid state here, then properly initialize
// resources and functions. Implementation-wise, PartiallyRevivedObjects
// contains maps keyed by the node number of the SavedObjectGraph, and map to an
// object of the corresponding type. So, if node 2 in the object graph is a
// variable, PartiallyRevivedObjects.variables[2] exists, and corresponds to a
// tensorflow::Variable object. The only exception to this is the
// "signatures_map", which is keyed by the "signature" key
// (https://github.com/tensorflow/tensorflow/blob/372918decee7f558b3c194b04f77c20dcc679a31/tensorflow/core/protobuf/meta_graph.proto#L89),
// and maps to the SignatureDefFunction node in the SavedObjectGraph.
struct PartiallyRevivedObjects {
  gtl::FlatMap<int, std::unique_ptr<Variable>> variables;
  gtl::FlatMap<int, std::unique_ptr<Asset>> assets;
  gtl::FlatMap<int, std::unique_ptr<Constant>> constants;
  gtl::FlatMap<int, TFConcreteFunctionRevivalState> concrete_functions;
  gtl::FlatMap<int, TFSignatureDefFunctionRevivalState> signature_def_functions;
  gtl::FlatMap<int, RestoredResourceRevivalState> restored_resources;
  gtl::FlatMap<std::string, int> signatures_map;

  Status Build(ImmediateExecutionContext* ctx,
               const SavedObjectGraph& obj_graph, RevivedObjects* revived);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_REVIVED_TYPES_PARTIALLY_REVIVED_OBJECTS_H_
