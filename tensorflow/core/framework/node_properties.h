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

#ifndef TENSORFLOW_CORE_FRAMEWORK_NODE_PROPERTIES_H_
#define TENSORFLOW_CORE_FRAMEWORK_NODE_PROPERTIES_H_

#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

class OpRegistryInterface;

struct NodeProperties {
 public:
  NodeProperties(const OpDef* op_def, NodeDef node_def,
                 const DataTypeSlice inputs, const DataTypeSlice outputs)
      : NodeProperties(op_def, std::move(node_def),
                       DataTypeVector(inputs.begin(), inputs.end()),
                       DataTypeVector(outputs.begin(), outputs.end())) {}

  NodeProperties(const OpDef* _op_def, NodeDef&& _node_def,
                 DataTypeVector inputs, DataTypeVector outputs)
      : op_def(_op_def),
        node_def(std::move(_node_def)),
        input_types(std::move(inputs)),
        input_types_slice(input_types),
        output_types(std::move(outputs)),
        output_types_slice(output_types) {}

  // Resets the 'props' shared pointer to point to a new NodeProperties created
  // from the given NodeDef. 'op_registry' is used to look up the OpDef
  // corresponding to node_def.op(). Returns an error if OpDef lookup or
  // creation failed.
  static Status CreateFromNodeDef(NodeDef node_def,
                                  const OpRegistryInterface* op_registry,
                                  std::shared_ptr<const NodeProperties>* props);

  const OpDef* op_def;  // not owned.
  NodeDef node_def;
  DataTypeVector input_types;
  DataTypeSlice input_types_slice;
  DataTypeVector output_types;
  DataTypeSlice output_types_slice;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_NODE_PROPERTIES_H_
