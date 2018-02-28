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

#include "tensorflow/c/c_api_experimental.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/session_options.h"

using tensorflow::GraphDef;
using tensorflow::NodeDef;

static void Int32Deallocator(void* data, size_t, void* arg) {
  delete[] static_cast<int32_t*>(data);
}

static void DoubleDeallocator(void* data, size_t, void* arg) {
  delete[] static_cast<double*>(data);
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

TF_Tensor* Int32Tensor(const int64_t* dims, int num_dims,
                       const int32_t* values) {
  int64_t num_values = 1;
  for (int i = 0; i < num_dims; ++i) {
    num_values *= dims[i];
  }
  TF_Tensor* t =
      TF_AllocateTensor(TF_INT32, dims, num_dims, sizeof(int32_t) * num_values);
  memcpy(TF_TensorData(t), values, sizeof(int32_t) * num_values);
  return t;
}

TF_Tensor* Int32Tensor(const std::vector<int32_t>& values) {
  int64_t dims = values.size();
  return Int32Tensor(&dims, 1, values.data());
}

TF_Tensor* Int32Tensor(int32_t v) {
  const int num_bytes = sizeof(int32_t);
  int32_t* values = new int32_t[1];
  values[0] = v;
  return TF_NewTensor(TF_INT32, nullptr, 0, values, num_bytes,
                      &Int32Deallocator, nullptr);
}

TF_Tensor* DoubleTensor(double v) {
  const int num_bytes = sizeof(double);
  double* values = new double[1];
  values[0] = v;
  return TF_NewTensor(TF_DOUBLE, nullptr, 0, values, num_bytes,
                      &DoubleDeallocator, nullptr);
}

// All the *Helper methods are used as a workaround for the restrictions that
// one cannot call ASSERT_* methods in non-void-returning functions (when
// exceptions are disabled during compilation)
void PlaceholderHelper(TF_Graph* graph, TF_Status* s, const char* name,
                       TF_Operation** op) {
  TF_OperationDescription* desc = TF_NewOperation(graph, "Placeholder", name);
  TF_SetAttrType(desc, "dtype", TF_INT32);
  *op = TF_FinishOperation(desc, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  ASSERT_NE(*op, nullptr);
}

TF_Operation* Placeholder(TF_Graph* graph, TF_Status* s, const char* name) {
  TF_Operation* op;
  PlaceholderHelper(graph, s, name, &op);
  return op;
}

void ConstHelper(TF_Tensor* t, TF_Graph* graph, TF_Status* s, const char* name,
                 TF_Operation** op) {
  TF_OperationDescription* desc = TF_NewOperation(graph, "Const", name);
  TF_SetAttrTensor(desc, "value", t, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  TF_SetAttrType(desc, "dtype", TF_TensorType(t));
  *op = TF_FinishOperation(desc, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  ASSERT_NE(*op, nullptr);
}

TF_Operation* Const(TF_Tensor* t, TF_Graph* graph, TF_Status* s,
                    const char* name) {
  TF_Operation* op;
  ConstHelper(t, graph, s, name, &op);
  return op;
}

TF_Operation* ScalarConst(int32_t v, TF_Graph* graph, TF_Status* s,
                          const char* name) {
  unique_tensor_ptr tensor(Int32Tensor(v), TF_DeleteTensor);
  return Const(tensor.get(), graph, s, name);
}

TF_Operation* ScalarConst(double v, TF_Graph* graph, TF_Status* s,
                          const char* name) {
  unique_tensor_ptr tensor(DoubleTensor(v), TF_DeleteTensor);
  return Const(tensor.get(), graph, s, name);
}

void AddOpHelper(TF_Operation* l, TF_Operation* r, TF_Graph* graph,
                 TF_Status* s, const char* name, TF_Operation** op,
                 bool check) {
  TF_OperationDescription* desc = TF_NewOperation(graph, "AddN", name);
  TF_Output add_inputs[2] = {{l, 0}, {r, 0}};
  TF_AddInputList(desc, add_inputs, 2);
  *op = TF_FinishOperation(desc, s);
  if (check) {
    ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
    ASSERT_NE(*op, nullptr);
  }
}

TF_Operation* Add(TF_Operation* l, TF_Operation* r, TF_Graph* graph,
                  TF_Status* s, const char* name) {
  TF_Operation* op;
  AddOpHelper(l, r, graph, s, name, &op, true);
  return op;
}

TF_Operation* AddNoCheck(TF_Operation* l, TF_Operation* r, TF_Graph* graph,
                         TF_Status* s, const char* name) {
  TF_Operation* op;
  AddOpHelper(l, r, graph, s, name, &op, false);
  return op;
}

TF_Operation* AddWithCtrlDependency(TF_Operation* l, TF_Operation* r,
                                    TF_Graph* graph, TF_Operation* ctrl_op,
                                    TF_Status* s, const char* name) {
  TF_OperationDescription* desc = TF_NewOperation(graph, "AddN", name);
  TF_Output add_inputs[2] = {{l, 0}, {r, 0}};
  TF_AddInputList(desc, add_inputs, 2);
  TF_AddControlInput(desc, ctrl_op);
  return TF_FinishOperation(desc, s);
}

// If `op_device` is non-empty, set the created op on that device.
void BinaryOpHelper(const char* op_name, TF_Operation* l, TF_Operation* r,
                    TF_Graph* graph, TF_Status* s, const char* name,
                    TF_Operation** op, const string& op_device, bool check) {
  TF_OperationDescription* desc = TF_NewOperation(graph, op_name, name);
  if (!op_device.empty()) {
    TF_SetDevice(desc, op_device.c_str());
  }
  TF_AddInput(desc, {l, 0});
  TF_AddInput(desc, {r, 0});
  *op = TF_FinishOperation(desc, s);
  if (check) {
    ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
    ASSERT_NE(*op, nullptr);
  }
}

TF_Operation* MinWithDevice(TF_Operation* l, TF_Operation* r, TF_Graph* graph,
                            const string& op_device, TF_Status* s,
                            const char* name) {
  TF_Operation* op;
  BinaryOpHelper("Min", l, r, graph, s, name, &op, op_device, true);
  return op;
}

TF_Operation* Min(TF_Operation* l, TF_Operation* r, TF_Graph* graph,
                  TF_Status* s, const char* name) {
  return MinWithDevice(l, r, graph, /*op_device=*/"", s, name);
}

TF_Operation* Add(TF_Output l, TF_Output r, TF_Graph* graph, TF_Status* s,
                  const char* name) {
  TF_OperationDescription* desc = TF_NewOperation(graph, "AddN", name);
  TF_Output inputs[2] = {l, r};
  TF_AddInputList(desc, inputs, 2);
  return TF_FinishOperation(desc, s);
}

void NegHelper(TF_Operation* n, TF_Graph* graph, TF_Status* s, const char* name,
               TF_Operation** op) {
  TF_OperationDescription* desc = TF_NewOperation(graph, "Neg", name);
  TF_Output neg_input = {n, 0};
  TF_AddInput(desc, neg_input);
  *op = TF_FinishOperation(desc, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  ASSERT_NE(*op, nullptr);
}

TF_Operation* Neg(TF_Operation* n, TF_Graph* graph, TF_Status* s,
                  const char* name) {
  TF_Operation* op;
  NegHelper(n, graph, s, name, &op);
  return op;
}

TF_Operation* LessThan(TF_Output l, TF_Output r, TF_Graph* graph,
                       TF_Status* s) {
  TF_OperationDescription* desc = TF_NewOperation(graph, "Less", "less_than");
  TF_AddInput(desc, l);
  TF_AddInput(desc, r);
  return TF_FinishOperation(desc, s);
}

TF_Operation* RandomUniform(TF_Operation* shape, TF_DataType dtype,
                            TF_Graph* graph, TF_Status* s) {
  TF_OperationDescription* desc =
      TF_NewOperation(graph, "RandomUniform", "random_uniform");
  TF_AddInput(desc, {shape, 0});
  TF_SetAttrType(desc, "dtype", dtype);
  return TF_FinishOperation(desc, s);
}

void Split3Helper(TF_Operation* input, TF_Graph* graph, TF_Status* s,
                  const char* name, TF_Operation** op) {
  TF_Operation* zero = ScalarConst(
      0, graph, s, ::tensorflow::strings::StrCat(name, "_const0").c_str());
  TF_OperationDescription* desc = TF_NewOperation(graph, "Split", name);
  TF_AddInput(desc, {zero, 0});
  TF_AddInput(desc, {input, 0});
  TF_SetAttrInt(desc, "num_split", 3);
  TF_SetAttrType(desc, "T", TF_INT32);
  // Set device to CPU since there is no version of split for int32 on GPU
  // TODO(iga): Convert all these helpers and tests to use floats because
  // they are usually available on GPUs. After doing this, remove TF_SetDevice
  // call in c_api_function_test.cc
  TF_SetDevice(desc, "/cpu:0");
  *op = TF_FinishOperation(desc, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  ASSERT_NE(*op, nullptr);
}

TF_Operation* Split3(TF_Operation* input, TF_Graph* graph, TF_Status* s,
                     const char* name) {
  TF_Operation* op;
  Split3Helper(input, graph, s, name, &op);
  return op;
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

bool GetFunctionDef(TF_Function* func, tensorflow::FunctionDef* func_def) {
  TF_Status* s = TF_NewStatus();
  TF_Buffer* buffer = TF_NewBuffer();
  TF_FunctionToFunctionDef(func, buffer, s);
  bool ret = TF_GetCode(s) == TF_OK;
  EXPECT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  if (ret) ret = func_def->ParseFromArray(buffer->data, buffer->length);
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

std::vector<std::pair<string, string>> GetGradDefs(
    const tensorflow::GraphDef& graph_def) {
  std::vector<std::pair<string, string>> grads;
  for (const tensorflow::GradientDef& grad : graph_def.library().gradient()) {
    grads.emplace_back(grad.function_name(), grad.gradient_func());
  }
  std::sort(grads.begin(), grads.end());
  return grads;
}

std::vector<string> GetFuncNames(const tensorflow::GraphDef& graph_def) {
  std::vector<string> names;
  for (const tensorflow::FunctionDef& func : graph_def.library().function()) {
    names.push_back(func.signature().name());
  }
  std::sort(names.begin(), names.end());
  return names;
}

CSession::CSession(TF_Graph* graph, TF_Status* s, bool use_XLA) {
  TF_SessionOptions* opts = TF_NewSessionOptions();
  TF_EnableXLACompilation(opts, use_XLA);
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
