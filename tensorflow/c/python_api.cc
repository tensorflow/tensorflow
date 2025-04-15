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

#include <string>

#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/core/framework/cpp_shape_inference.pb.h"
#include "tensorflow/core/framework/full_type.pb.h"

namespace tensorflow {

// Hack to export the tensorflow::RecordMutation symbol for windows.
// Do not delete. Do not use.
void ExportRecordMutation(  // NOLINT: Intentionally unused function.
    TF_Graph* graph, const TF_Operation& op, const char* mutation_type) {
  mutex_lock l(graph->mu);
  RecordMutation(graph, op, mutation_type);
}

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
                 const TF_Buffer* full_type_proto) {
  TF_SetFullType(graph, op, full_type_proto);
}

void SetRequestedDevice(TF_Graph* graph, TF_Operation* op, const char* device) {
  TF_SetRequestedDevice(graph, op, device);
}

void UpdateEdge(TF_Graph* graph, TF_Output new_src, TF_Input dst,
                TF_Status* status) {
  TF_UpdateEdge(graph, new_src, dst, status);
}

void ExtendSession(TF_Session* session, TF_Status* status) {
  TF_ExtendSession(session, status);
}

std::string GetHandleShapeAndType(TF_Graph* graph, TF_Output output) {
  Node* node = &output.oper->node;
  tensorflow::core::CppShapeInferenceResult::HandleData handle_data;
  handle_data.set_is_set(true);
  {
    mutex_lock l(graph->mu);
    tensorflow::shape_inference::InferenceContext* ic =
        graph->refiner.GetContext(node);
    CHECK(ic != nullptr);
    CHECK_LT(output.index, ic->num_outputs());
    const auto* shapes_and_types =
        ic->output_handle_shapes_and_types(output.index);
    if (shapes_and_types == nullptr) return "";

    for (const auto& p : *shapes_and_types) {
      auto* out_shape_and_type = handle_data.add_shape_and_type();
      ic->ShapeHandleToProto(p.shape, out_shape_and_type->mutable_shape());
      out_shape_and_type->set_dtype(p.dtype);
      *out_shape_and_type->mutable_type() = p.type;
    }
  }
  string result;
  handle_data.SerializeToString(&result);
  return result;
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
