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

#include "tensorflow/c/c_test_util.h"

#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/platform/logging.h"

using tensorflow::GraphDef;
using tensorflow::NodeDef;

static void Int32Deallocator(void* data, size_t, void* arg) {
  delete[] static_cast<int32_t*>(data);
}

TF_Tensor* Int8Tensor(const int64_t* dims, int num_dims, const char* values) {
  int64_t num_values = 1;
  for (int i = 0; i < num_dims; ++i) {
    num_values *= dims[i];
  }
  TF_Tensor* t =
      TF_AllocateTensor(TF_INT8, dims, num_dims, sizeof(char) * num_values);
  memcpy(TF_TensorData(t), values, sizeof(char) * num_values);
  return t;
}

TF_Tensor* Int32Tensor(int32_t v) {
  const int num_bytes = sizeof(int32_t);
  int32_t* values = new int32_t[1];
  values[0] = v;
  return TF_NewTensor(TF_INT32, nullptr, 0, values, num_bytes,
                      &Int32Deallocator, nullptr);
}

TF_Operation* Placeholder(TF_Graph* graph, TF_Status* s, const char* name) {
  TF_OperationDescription* desc = TF_NewOperation(graph, "Placeholder", name);
  TF_SetAttrType(desc, "dtype", TF_INT32);
  return TF_FinishOperation(desc, s);
}

TF_Operation* Const(TF_Tensor* t, TF_Graph* graph, TF_Status* s,
                    const char* name) {
  TF_OperationDescription* desc = TF_NewOperation(graph, "Const", name);
  TF_SetAttrTensor(desc, "value", t, s);
  if (TF_GetCode(s) != TF_OK) return nullptr;
  TF_SetAttrType(desc, "dtype", TF_TensorType(t));
  return TF_FinishOperation(desc, s);
}

TF_Operation* ScalarConst(int32_t v, TF_Graph* graph, TF_Status* s,
                          const char* name) {
  unique_tensor_ptr tensor(Int32Tensor(v), TF_DeleteTensor);
  return Const(tensor.get(), graph, s, name);
}

TF_Operation* Add(TF_Operation* l, TF_Operation* r, TF_Graph* graph,
                  TF_Status* s, const char* name) {
  TF_OperationDescription* desc = TF_NewOperation(graph, "AddN", name);
  TF_Output add_inputs[2] = {{l, 0}, {r, 0}};
  TF_AddInputList(desc, add_inputs, 2);
  return TF_FinishOperation(desc, s);
}

TF_Operation* Add(TF_Output l, TF_Output r, TF_Graph* graph, TF_Status* s,
                  const char* name) {
  TF_OperationDescription* desc = TF_NewOperation(graph, "AddN", name);
  TF_Output inputs[2] = {l, r};
  TF_AddInputList(desc, inputs, 2);
  return TF_FinishOperation(desc, s);
}

TF_Operation* Neg(TF_Operation* n, TF_Graph* graph, TF_Status* s) {
  TF_OperationDescription* desc = TF_NewOperation(graph, "Neg", "neg");
  TF_Output neg_input = {n, 0};
  TF_AddInput(desc, neg_input);
  return TF_FinishOperation(desc, s);
}

TF_Operation* LessThan(TF_Output l, TF_Output r, TF_Graph* graph,
                       TF_Status* s) {
  TF_OperationDescription* desc = TF_NewOperation(graph, "Less", "less_than");
  TF_AddInput(desc, l);
  TF_AddInput(desc, r);
  return TF_FinishOperation(desc, s);
}

bool IsPlaceholder(const tensorflow::NodeDef& node_def) {
  if (node_def.op() != "Placeholder" || node_def.name() != "feed") {
    return false;
  }
  bool found_dtype = false;
  bool found_shape = false;
  for (const auto& attr : node_def.attr()) {
    if (attr.first == "dtype") {
      if (attr.second.type() == tensorflow::DT_INT32) {
        found_dtype = true;
      } else {
        return false;
      }
    } else if (attr.first == "shape") {
      found_shape = true;
    }
  }
  return found_dtype && found_shape;
}

bool IsScalarConst(const tensorflow::NodeDef& node_def, int v) {
  if (node_def.op() != "Const" || node_def.name() != "scalar") {
    return false;
  }
  bool found_dtype = false;
  bool found_value = false;
  for (const auto& attr : node_def.attr()) {
    if (attr.first == "dtype") {
      if (attr.second.type() == tensorflow::DT_INT32) {
        found_dtype = true;
      } else {
        return false;
      }
    } else if (attr.first == "value") {
      if (attr.second.has_tensor() &&
          attr.second.tensor().int_val_size() == 1 &&
          attr.second.tensor().int_val(0) == v) {
        found_value = true;
      } else {
        return false;
      }
    }
  }
  return found_dtype && found_value;
}

bool IsAddN(const tensorflow::NodeDef& node_def, int n) {
  if (node_def.op() != "AddN" || node_def.name() != "add" ||
      node_def.input_size() != n) {
    return false;
  }
  bool found_t = false;
  bool found_n = false;
  for (const auto& attr : node_def.attr()) {
    if (attr.first == "T") {
      if (attr.second.type() == tensorflow::DT_INT32) {
        found_t = true;
      } else {
        return false;
      }
    } else if (attr.first == "N") {
      if (attr.second.i() == n) {
        found_n = true;
      } else {
        return false;
      }
    }
  }
  return found_t && found_n;
}

bool IsNeg(const tensorflow::NodeDef& node_def, const string& input) {
  return node_def.op() == "Neg" && node_def.name() == "neg" &&
         node_def.input_size() == 1 && node_def.input(0) == input;
}

bool GetGraphDef(TF_Graph* graph, tensorflow::GraphDef* graph_def) {
  TF_Status* s = TF_NewStatus();
  TF_Buffer* buffer = TF_NewBuffer();
  TF_GraphToGraphDef(graph, buffer, s);
  bool ret = TF_GetCode(s) == TF_OK;
  EXPECT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  if (ret) ret = graph_def->ParseFromArray(buffer->data, buffer->length);
  TF_DeleteBuffer(buffer);
  TF_DeleteStatus(s);
  return ret;
}

bool GetNodeDef(TF_Operation* oper, tensorflow::NodeDef* node_def) {
  TF_Status* s = TF_NewStatus();
  TF_Buffer* buffer = TF_NewBuffer();
  TF_OperationToNodeDef(oper, buffer, s);
  bool ret = TF_GetCode(s) == TF_OK;
  EXPECT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  if (ret) ret = node_def->ParseFromArray(buffer->data, buffer->length);
  TF_DeleteBuffer(buffer);
  TF_DeleteStatus(s);
  return ret;
}

bool GetAttrValue(TF_Operation* oper, const char* attr_name,
                  tensorflow::AttrValue* attr_value, TF_Status* s) {
  TF_Buffer* buffer = TF_NewBuffer();
  TF_OperationGetAttrValueProto(oper, attr_name, buffer, s);
  bool ret = TF_GetCode(s) == TF_OK;
  if (ret) ret = attr_value->ParseFromArray(buffer->data, buffer->length);
  TF_DeleteBuffer(buffer);
  return ret;
}

CSession::CSession(TF_Graph* graph, TF_Status* s) {
  TF_SessionOptions* opts = TF_NewSessionOptions();
  session_ = TF_NewSession(graph, opts, s);
  TF_DeleteSessionOptions(opts);
}

CSession::CSession(TF_Session* session) : session_(session) {}

CSession::~CSession() {
  TF_Status* s = TF_NewStatus();
  CloseAndDelete(s);
  EXPECT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  TF_DeleteStatus(s);
}

void CSession::SetInputs(
    std::vector<std::pair<TF_Operation*, TF_Tensor*>> inputs) {
  DeleteInputValues();
  inputs_.clear();
  for (const auto& p : inputs) {
    inputs_.emplace_back(TF_Output{p.first, 0});
    input_values_.emplace_back(p.second);
  }
}

void CSession::SetOutputs(std::initializer_list<TF_Operation*> outputs) {
  ResetOutputValues();
  outputs_.clear();
  for (TF_Operation* o : outputs) {
    outputs_.emplace_back(TF_Output{o, 0});
  }
  output_values_.resize(outputs_.size());
}

void CSession::SetOutputs(const std::vector<TF_Output>& outputs) {
  ResetOutputValues();
  outputs_ = outputs;
  output_values_.resize(outputs_.size());
}

void CSession::SetTargets(std::initializer_list<TF_Operation*> targets) {
  targets_.clear();
  for (TF_Operation* t : targets) {
    targets_.emplace_back(t);
  }
}

void CSession::Run(TF_Status* s) {
  if (inputs_.size() != input_values_.size()) {
    ADD_FAILURE() << "Call SetInputs() before Run()";
    return;
  }
  ResetOutputValues();
  output_values_.resize(outputs_.size(), nullptr);

  const TF_Output* inputs_ptr = inputs_.empty() ? nullptr : &inputs_[0];
  TF_Tensor* const* input_values_ptr =
      input_values_.empty() ? nullptr : &input_values_[0];

  const TF_Output* outputs_ptr = outputs_.empty() ? nullptr : &outputs_[0];
  TF_Tensor** output_values_ptr =
      output_values_.empty() ? nullptr : &output_values_[0];

  TF_Operation* const* targets_ptr = targets_.empty() ? nullptr : &targets_[0];

  TF_SessionRun(session_, nullptr, inputs_ptr, input_values_ptr, inputs_.size(),
                outputs_ptr, output_values_ptr, outputs_.size(), targets_ptr,
                targets_.size(), nullptr, s);

  DeleteInputValues();
}

void CSession::CloseAndDelete(TF_Status* s) {
  DeleteInputValues();
  ResetOutputValues();
  if (session_ != nullptr) {
    TF_CloseSession(session_, s);
    EXPECT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
    TF_DeleteSession(session_, s);
    session_ = nullptr;
  }
}

void CSession::DeleteInputValues() {
  for (size_t i = 0; i < input_values_.size(); ++i) {
    TF_DeleteTensor(input_values_[i]);
  }
  input_values_.clear();
}

void CSession::ResetOutputValues() {
  for (size_t i = 0; i < output_values_.size(); ++i) {
    if (output_values_[i] != nullptr) TF_DeleteTensor(output_values_[i]);
  }
  output_values_.clear();
}
