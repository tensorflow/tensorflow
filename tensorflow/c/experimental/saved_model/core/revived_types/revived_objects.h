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

#ifndef TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_REVIVED_TYPES_REVIVED_OBJECTS_H_
#define TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_REVIVED_TYPES_REVIVED_OBJECTS_H_

#include <memory>
#include <unordered_map>

#include "tensorflow/c/experimental/saved_model/core/revived_types/asset.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/constant.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/restored_resource.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/tf_concrete_function.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/tf_signature_def_function.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/variable.h"
#include "tensorflow/core/lib/gtl/flatmap.h"

namespace tensorflow {

// RevivedObjects is mainly used as a container for all the "state" owned by
// SavedModel. It stores all non-"user object" nodes from a SavedModel
// (https://github.com/tensorflow/tensorflow/blob/568e2bef00f24af1159a0846abf67c099ca78a21/tensorflow/core/protobuf/saved_object_graph.proto#L57-L62)
// in a "fully constructed" state. It is effectively a strongly typed map, where
// each member is a map from the node id in the SavedObjectGraph's nodes
// (https://github.com/tensorflow/tensorflow/blob/568e2bef00f24af1159a0846abf67c099ca78a21/tensorflow/core/protobuf/saved_object_graph.proto#L25-L29)
// to the revived object of the corresponding type.
struct RevivedObjects {
  gtl::FlatMap<int, std::unique_ptr<Variable>> variables;
  gtl::FlatMap<int, std::unique_ptr<Asset>> assets;
  gtl::FlatMap<int, std::unique_ptr<Constant>> constants;
  gtl::FlatMap<int, std::unique_ptr<TFConcreteFunction>> concrete_functions;
  gtl::FlatMap<int, std::unique_ptr<TFSignatureDefFunction>>
      signature_def_functions;
  gtl::FlatMap<int, RestoredResource> restored_resources;
  gtl::FlatMap<std::string, int> signatures_map;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_REVIVED_TYPES_REVIVED_OBJECTS_H_
