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

#include "tensorflow/c/python_api.h"

#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/core/framework/full_type.pb.h"
#include "tensorflow/python/framework/cpp_shape_inference.pb.h"

namespace tensorflow {

void AddControlInput(TF_Graph* graph, TF_Operation* op, TF_Operation* input) {
  TF_AddOperationControlInput(graph, op, input);
}

void SetAttr(TF_Graph* graph, TF_Operation* op, const char* attr_name,
             TF_Buffer* attr_value_proto, TF_Status* status) {
  TF_SetAttr(graph, op, attr_name, attr_value_proto, status);
}

void ClearAttr(TF_Graph* graph, TF_Operation* op, const char* attr_name,
               TF_Status* status) {
  TF_ClearAttr(graph, op, attr_name, status);
}

void SetFullType(TF_Graph* graph, TF_Operation* op,
                 const FullTypeDef& full_type) {
  TF_SetFullType(graph, op, full_type);
}

void SetRequestedDevice(TF_Graph* graph, TF_Operation* op, const char* device) {
  TF_SetRequestedDevice(graph, op, device);
}

void UpdateEdge(TF_Graph* graph, TF_Output new_src, TF_Input dst,
                TF_Status* status) {
  TF_UpdateEdge(graph, new_src, dst, status);
}

void ExtendSession(TF_Session* session, TF_Status* status) {
  TF_ExtendSession(session, status)
}

std::string GetHandleShapeAndType(TF_Graph* graph, TF_Output output) {
  return TF_GetHandleShapeAndType(graph, output);
}

void SetHandleShapeAndType(TF_Graph* graph, TF_Output output, const void* proto,
                           size_t proto_len, TF_Status* status) {
  TF_SetHandleShapeAndType(graph, output, proto, proto_len, status);
}

void AddWhileInputHack(TF_Graph* graph, TF_Output new_src, TF_Operation* dst,
                       TF_Status* status) {
  TF_AddWhileInputHack(graph, new_src, dst, status);
}

}  // namespace tensorflow
