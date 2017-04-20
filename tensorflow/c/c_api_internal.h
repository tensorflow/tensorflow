/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/c/c_api.h"

#include <vector>
#include <unordered_map>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/common_runtime/shape_refiner.h"


// Internal structures used by the C API. These are likely to change and should
// not be depended on.

struct TF_Status {
  tensorflow::Status status;
};

struct TF_Tensor {
  TF_DataType dtype;
  tensorflow::TensorShape shape;
  tensorflow::TensorBuffer* buffer;
};

struct TF_SessionOptions {
  tensorflow::SessionOptions options;
};

struct TF_DeprecatedSession {
  tensorflow::Session* session;
};

struct TF_Library {
  void* lib_handle;
  TF_Buffer op_list;
};

struct TF_Graph {
  TF_Graph()
      : graph(tensorflow::OpRegistry::Global()),
        refiner(graph.versions().producer(), graph.op_registry()),
        num_sessions(0),
        delete_requested(false),
        parent(nullptr),
        parent_inputs(nullptr) {}
  tensorflow::mutex mu;
  tensorflow::Graph graph GUARDED_BY(mu);

  // Runs shape inference.
  tensorflow::ShapeRefiner refiner GUARDED_BY(mu);

  // Maps from name of an operation to the Node* in 'graph'.
  std::unordered_map<tensorflow::string, tensorflow::Node*> name_map
      GUARDED_BY(mu);

  // TF_Graph may only / must be deleted when
  //   num_sessions == 0 && delete_requested == true

  // num_sessions incremented by TF_NewSession, and decremented by
  // TF_DeleteSession.
  int num_sessions GUARDED_BY(mu);
  bool delete_requested GUARDED_BY(mu);  // set true by TF_DeleteGraph

  // Used to link graphs contained in TF_WhileParams to the parent graph that
  // will eventually contain the full while loop.
  TF_Graph* parent;
  TF_Output* parent_inputs;
};

struct TF_OperationDescription {
  TF_OperationDescription(TF_Graph* g, const char* op_type,
                          const char* node_name)
      : node_builder(node_name, op_type, g->graph.op_registry()), graph(g) {}

  tensorflow::NodeBuilder node_builder;
  TF_Graph* graph;
  std::vector<tensorflow::string> colocation_constraints;
};

struct TF_Operation {
  tensorflow::Node node;
};

struct TF_Session {
  TF_Session(tensorflow::Session* s, TF_Graph* g)
      : session(s), graph(g), last_num_graph_nodes(0) {}
  tensorflow::Session* session;
  TF_Graph* graph;
  tensorflow::mutex mu;
  int last_num_graph_nodes;
};

struct TF_ImportGraphDefOptions {
  tensorflow::ImportGraphDefOptions opts;
};
