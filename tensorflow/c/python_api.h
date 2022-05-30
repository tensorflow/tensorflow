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

#ifndef TENSORFLOW_C_PYTHON_API_H_
#define TENSORFLOW_C_PYTHON_API_H_

#include <string>

#include "tensorflow/c/c_api.h"
#include "tensorflow/core/framework/full_type.pb.h"

// These functions can be removed without notice. They exist to facilitate some
// refactoring of graph construction code in the Python API.

namespace tensorflow {

void AddControlInput(TF_Graph* graph, TF_Operation* op, TF_Operation* input);

// Changes an attr value in the node_def Protocol Buffer and sets a status upon
// completion.
void SetAttr(TF_Graph* graph, TF_Operation* op, const char* attr_name,
             TF_Buffer* attr_value_proto, TF_Status* status);

// Clears the attr in the node_def Protocol Buffer and sets a status upon
// completion.
void ClearAttr(TF_Graph* graph, TF_Operation* op, const char* attr_name,
               TF_Status* status);

// Sets the experimental_type` field in the node_def Protocol Buffer.
void SetFullType(TF_Graph* graph, TF_Operation* op,
                 const FullTypeDef& full_type);

void SetRequestedDevice(TF_Graph* graph, TF_Operation* op, const char* device);

// Updates 'dst' to consume 'new_src'.
void UpdateEdge(TF_Graph* graph, TF_Output new_src, TF_Input dst,
                TF_Status* status);

void RemoveAllControlInputs(TF_Graph* graph, TF_Operation* op);

// Sets whether ops missing a shape inference function should trigger an
// error. The default is true.
void SetRequireShapeInferenceFns(TF_Graph* graph, bool require);

// Extends `session` with any new operations added to its associated graph.
// Usually this happens automatically in TF_SessionRun. After this is called,
// TF_SessionRun will no longer extend the session on every call.
//
// We expose this here to allow fine-grained synchronization in multi-threaded
// workloads, which is required since the Python implementation depends on the
// above mutation methods. This allows us to prevent modifications to nodes in
// the graph after the session has been made aware of them.
void ExtendSession(TF_Session* session, TF_Status* status);

// Returns the serialized CppShapeInferenceResult::HandleData proto for
// `output` if its a resource or variant tensor, or otherwise returns the empty
// string.
std::string GetHandleShapeAndType(TF_Graph* graph, TF_Output output);

// Sets `output` based on `proto`, which should be a serialized
// CppShapeInferenceResult::HandleData proto. `output` should be a resource
// or variant tensor.
// NOTE(skyewm): `proto` is passed a void*/size_t pair instead of a std::string
// because I couldn't get SWIG to work otherwise.
void SetHandleShapeAndType(TF_Graph* graph, TF_Output output, const void* proto,
                           size_t proto_len, TF_Status* status);

// This method is used to add a new input edge to 'dst', which must be a While
// op. The While op's "T" attribute must have already been updated to include
// the new edge. This is used to construct tf.while_loop gradients.
void AddWhileInputHack(TF_Graph* graph, TF_Output new_src, TF_Operation* dst,
                       TF_Status* status);

}  // namespace tensorflow

#endif  // TENSORFLOW_C_PYTHON_API_H_
