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

#include "absl/container/flat_hash_map.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/asset.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/constant.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/restored_resource.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/tf_concrete_function.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/tf_signature_def_function.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/variable.h"
#include "tensorflow/core/lib/gtl/flatmap.h"

namespace tensorflow {

// A container for revived saved model objects.
//
// Most of the objects will be revived from nodes in the object graph, and for
// those objects this container provides a map from node id to the revived
// objects.
//
// For objects that have to be revived but are not part of the object graph,
// this container provides a place where the objects can be stored so they are
// available to the runtime.
template <typename T>
class RevivedObjectContainer {
 public:
  // Insert an object that is not related to a node id. This usually means the
  // object was not referenced by the object_graph, but is needed by other
  // objects.
  void Insert(std::unique_ptr<T> object) {
    objects_.push_back(std::move(object));
  }

  // Insert an object that is tied to the given object graph node id.
  void Insert(std::unique_ptr<T> object, int node_id) {
    objects_by_id_[node_id] = object.get();
    Insert(std::move(object));
  }

  // Find an object by the object graph node id.
  // Returns nullptr if there is no such object.
  T* Find(int node_id) {
    auto it = objects_by_id_.find(node_id);
    return it == objects_by_id_.end() ? nullptr : it->second;
  }

 private:
  std::vector<std::unique_ptr<T>> objects_;
  absl::flat_hash_map<int, T*> objects_by_id_;
};

// RevivedObjects is mainly used as a container for all the "state" owned by
// SavedModel. It stores all non-"user object" nodes from a SavedModel
// (https://github.com/tensorflow/tensorflow/blob/568e2bef00f24af1159a0846abf67c099ca78a21/tensorflow/core/protobuf/saved_object_graph.proto#L57-L62)
// in a "fully constructed" state. It is effectively a strongly typed map, where
// each member is a map from the node id in the SavedObjectGraph's nodes
// (https://github.com/tensorflow/tensorflow/blob/568e2bef00f24af1159a0846abf67c099ca78a21/tensorflow/core/protobuf/saved_object_graph.proto#L25-L29)
// to the revived object of the corresponding type.
struct RevivedObjects {
  // Order of declaration is important here: we want the RestoredResources to be
  // freed after TFConcreteFunctions, for example.
  gtl::FlatMap<int, std::unique_ptr<Variable>> variables;
  gtl::FlatMap<int, std::unique_ptr<Asset>> assets;
  gtl::FlatMap<int, std::unique_ptr<Constant>> constants;
  gtl::FlatMap<int, std::unique_ptr<TFSignatureDefFunction>>
      signature_def_functions;
  RevivedObjectContainer<TFConcreteFunction> concrete_functions;
  gtl::FlatMap<int, RestoredResource> restored_resources;
  gtl::FlatMap<std::string, int> signatures_map;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_REVIVED_TYPES_REVIVED_OBJECTS_H_
