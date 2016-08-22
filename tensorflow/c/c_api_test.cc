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
#include "tensorflow/core/framework/graph.pb_text.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/platform/test.h"

using tensorflow::int32;
using tensorflow::string;
using tensorflow::GraphDef;
using tensorflow::NodeDef;
using tensorflow::Tensor;
using tensorflow::TensorShape;

namespace tensorflow {
bool TF_Tensor_DecodeStrings(TF_Tensor* src, Tensor* dst, TF_Status* status);
TF_Tensor* TF_Tensor_EncodeStrings(const Tensor& src);
}  // namespace tensorflow

namespace {

TEST(CApi, Status) {
  TF_Status* s = TF_NewStatus();
  EXPECT_EQ(TF_OK, TF_GetCode(s));
  EXPECT_EQ(tensorflow::string(), TF_Message(s));
  TF_SetStatus(s, TF_CANCELLED, "cancel");
  EXPECT_EQ(TF_CANCELLED, TF_GetCode(s));
  EXPECT_EQ(tensorflow::string("cancel"), TF_Message(s));
  TF_DeleteStatus(s);
}

static void Deallocator(void* data, size_t, void* arg) {
  tensorflow::cpu_allocator()->DeallocateRaw(data);
  *reinterpret_cast<bool*>(arg) = true;
}

TEST(CApi, Tensor) {
  const int num_bytes = 6 * sizeof(float);
  float* values =
      reinterpret_cast<float*>(tensorflow::cpu_allocator()->AllocateRaw(
          EIGEN_MAX_ALIGN_BYTES, num_bytes));
  int64_t dims[] = {2, 3};
  bool deallocator_called = false;
  TF_Tensor* t = TF_NewTensor(TF_FLOAT, dims, 2, values, num_bytes,
                              &Deallocator, &deallocator_called);
  EXPECT_FALSE(deallocator_called);
  EXPECT_EQ(TF_FLOAT, TF_TensorType(t));
  EXPECT_EQ(2, TF_NumDims(t));
  EXPECT_EQ(dims[0], TF_Dim(t, 0));
  EXPECT_EQ(dims[1], TF_Dim(t, 1));
  EXPECT_EQ(num_bytes, TF_TensorByteSize(t));
  EXPECT_EQ(static_cast<void*>(values), TF_TensorData(t));
  TF_DeleteTensor(t);
  EXPECT_TRUE(deallocator_called);
}

static void TestEncodeDecode(int line,
                             const std::vector<tensorflow::string>& data) {
  const tensorflow::int64 n = data.size();
  for (std::vector<tensorflow::int64> dims :
       std::vector<std::vector<tensorflow::int64>>{
           {n}, {1, n}, {n, 1}, {n / 2, 2}}) {
    // Create C++ Tensor
    Tensor src(tensorflow::DT_STRING, TensorShape(dims));
    for (tensorflow::int64 i = 0; i < src.NumElements(); i++) {
      src.flat<tensorflow::string>()(i) = data[i];
    }
    TF_Tensor* dst = TF_Tensor_EncodeStrings(src);

    // Convert back to a C++ Tensor and ensure we get expected output.
    TF_Status* status = TF_NewStatus();
    Tensor output;
    ASSERT_TRUE(TF_Tensor_DecodeStrings(dst, &output, status)) << line;
    ASSERT_EQ(TF_OK, TF_GetCode(status)) << line;
    ASSERT_EQ(src.NumElements(), output.NumElements()) << line;
    for (tensorflow::int64 i = 0; i < src.NumElements(); i++) {
      ASSERT_EQ(data[i], output.flat<tensorflow::string>()(i)) << line;
    }

    TF_DeleteStatus(status);
    TF_DeleteTensor(dst);
  }
}

TEST(CApi, TensorEncodeDecodeStrings) {
  TestEncodeDecode(__LINE__, {});
  TestEncodeDecode(__LINE__, {"hello"});
  TestEncodeDecode(__LINE__,
                   {"the", "quick", "brown", "fox", "jumped", "over"});

  tensorflow::string big(1000, 'a');
  TestEncodeDecode(__LINE__, {"small", big, "small2"});
}

TEST(CApi, SessionOptions) {
  TF_SessionOptions* opt = TF_NewSessionOptions();
  TF_DeleteSessionOptions(opt);
}

TEST(CApi, SessionWithRunMetadata) {
  TF_Status* s = TF_NewStatus();
  TF_SessionOptions* opt = TF_NewSessionOptions();
  TF_Session* session = TF_NewSession(opt, s);
  TF_DeleteSessionOptions(opt);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  TF_Buffer* run_options = TF_NewBufferFromString("", 0);
  TF_Buffer* run_metadata = TF_NewBuffer();
  TF_Run(session, run_options, nullptr, nullptr, 0, nullptr, nullptr, 0,
         nullptr, 0, run_metadata, s);
  EXPECT_EQ(TF_INVALID_ARGUMENT, TF_GetCode(s)) << TF_Message(s);
  EXPECT_EQ(std::string("Session was not created with a graph before Run()!"),
            std::string(TF_Message(s)));
  TF_DeleteBuffer(run_metadata);
  TF_DeleteBuffer(run_options);

  TF_DeleteSession(session, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  TF_DeleteStatus(s);
}

TEST(CAPI, DataTypeEnum) {
  EXPECT_EQ(TF_FLOAT, static_cast<TF_DataType>(tensorflow::DT_FLOAT));
  EXPECT_EQ(TF_DOUBLE, static_cast<TF_DataType>(tensorflow::DT_DOUBLE));
  EXPECT_EQ(TF_INT32, static_cast<TF_DataType>(tensorflow::DT_INT32));
  EXPECT_EQ(TF_UINT8, static_cast<TF_DataType>(tensorflow::DT_UINT8));
  EXPECT_EQ(TF_INT16, static_cast<TF_DataType>(tensorflow::DT_INT16));
  EXPECT_EQ(TF_INT8, static_cast<TF_DataType>(tensorflow::DT_INT8));
  EXPECT_EQ(TF_STRING, static_cast<TF_DataType>(tensorflow::DT_STRING));
  EXPECT_EQ(TF_COMPLEX64, static_cast<TF_DataType>(tensorflow::DT_COMPLEX64));
  EXPECT_EQ(TF_COMPLEX, TF_COMPLEX64);
  EXPECT_EQ(TF_INT64, static_cast<TF_DataType>(tensorflow::DT_INT64));
  EXPECT_EQ(TF_BOOL, static_cast<TF_DataType>(tensorflow::DT_BOOL));
  EXPECT_EQ(TF_QINT8, static_cast<TF_DataType>(tensorflow::DT_QINT8));
  EXPECT_EQ(TF_QUINT8, static_cast<TF_DataType>(tensorflow::DT_QUINT8));
  EXPECT_EQ(TF_QINT32, static_cast<TF_DataType>(tensorflow::DT_QINT32));
  EXPECT_EQ(TF_BFLOAT16, static_cast<TF_DataType>(tensorflow::DT_BFLOAT16));
  EXPECT_EQ(TF_QINT16, static_cast<TF_DataType>(tensorflow::DT_QINT16));
  EXPECT_EQ(TF_QUINT16, static_cast<TF_DataType>(tensorflow::DT_QUINT16));
  EXPECT_EQ(TF_UINT16, static_cast<TF_DataType>(tensorflow::DT_UINT16));
  EXPECT_EQ(TF_COMPLEX128, static_cast<TF_DataType>(tensorflow::DT_COMPLEX128));
  EXPECT_EQ(TF_HALF, static_cast<TF_DataType>(tensorflow::DT_HALF));
}

TEST(CAPI, StatusEnum) {
  EXPECT_EQ(TF_OK, static_cast<TF_Code>(tensorflow::error::OK));
  EXPECT_EQ(TF_CANCELLED, static_cast<TF_Code>(tensorflow::error::CANCELLED));
  EXPECT_EQ(TF_UNKNOWN, static_cast<TF_Code>(tensorflow::error::UNKNOWN));
  EXPECT_EQ(TF_INVALID_ARGUMENT,
            static_cast<TF_Code>(tensorflow::error::INVALID_ARGUMENT));
  EXPECT_EQ(TF_DEADLINE_EXCEEDED,
            static_cast<TF_Code>(tensorflow::error::DEADLINE_EXCEEDED));
  EXPECT_EQ(TF_NOT_FOUND, static_cast<TF_Code>(tensorflow::error::NOT_FOUND));
  EXPECT_EQ(TF_ALREADY_EXISTS,
            static_cast<TF_Code>(tensorflow::error::ALREADY_EXISTS));
  EXPECT_EQ(TF_PERMISSION_DENIED,
            static_cast<TF_Code>(tensorflow::error::PERMISSION_DENIED));
  EXPECT_EQ(TF_UNAUTHENTICATED,
            static_cast<TF_Code>(tensorflow::error::UNAUTHENTICATED));
  EXPECT_EQ(TF_RESOURCE_EXHAUSTED,
            static_cast<TF_Code>(tensorflow::error::RESOURCE_EXHAUSTED));
  EXPECT_EQ(TF_FAILED_PRECONDITION,
            static_cast<TF_Code>(tensorflow::error::FAILED_PRECONDITION));
  EXPECT_EQ(TF_ABORTED, static_cast<TF_Code>(tensorflow::error::ABORTED));
  EXPECT_EQ(TF_OUT_OF_RANGE,
            static_cast<TF_Code>(tensorflow::error::OUT_OF_RANGE));
  EXPECT_EQ(TF_UNIMPLEMENTED,
            static_cast<TF_Code>(tensorflow::error::UNIMPLEMENTED));
  EXPECT_EQ(TF_INTERNAL, static_cast<TF_Code>(tensorflow::error::INTERNAL));
  EXPECT_EQ(TF_UNAVAILABLE,
            static_cast<TF_Code>(tensorflow::error::UNAVAILABLE));
  EXPECT_EQ(TF_DATA_LOSS, static_cast<TF_Code>(tensorflow::error::DATA_LOSS));
}

static void Int32Deallocator(void* data, size_t, void* arg) {
  delete[] static_cast<tensorflow::int32*>(data);
}

static TF_Tensor* Int32Tensor(int32 v) {
  const int num_bytes = sizeof(tensorflow::int32);
  tensorflow::int32* values = new tensorflow::int32[1];
  values[0] = v;
  return TF_NewTensor(TF_INT32, nullptr, 0, values, num_bytes,
                      &Int32Deallocator, nullptr);
}

TF_Operation* Placeholder(TF_Graph* graph, TF_Status* s) {
  TF_OperationDescription* desc = TF_NewOperation(graph, "Placeholder", "feed");
  TF_SetAttrType(desc, "dtype", TF_INT32);
  return TF_FinishOperation(desc, s);
}

TF_Operation* ScalarConst(int32 v, TF_Graph* graph, TF_Status* s) {
  TF_OperationDescription* desc = TF_NewOperation(graph, "Const", "scalar");
  TF_SetAttrTensor(desc, "value", Int32Tensor(v), s);
  if (TF_GetCode(s) != TF_OK) return nullptr;
  TF_SetAttrType(desc, "dtype", TF_INT32);
  return TF_FinishOperation(desc, s);
}

TF_Operation* Add(TF_Operation* l, TF_Operation* r, TF_Graph* graph,
                  TF_Status* s) {
  TF_OperationDescription* desc = TF_NewOperation(graph, "AddN", "add");
  TF_Port add_inputs[2] = {{l, 0}, {r, 0}};
  TF_AddInputList(desc, add_inputs, 2);
  return TF_FinishOperation(desc, s);
}

TF_Operation* Neg(TF_Operation* n, TF_Graph* graph, TF_Status* s) {
  TF_OperationDescription* desc = TF_NewOperation(graph, "Neg", "neg");
  TF_Port neg_input = {n, 0};
  TF_AddInput(desc, neg_input);
  return TF_FinishOperation(desc, s);
}

bool IsPlaceholder(const NodeDef& node_def) {
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

bool IsScalarConst(const NodeDef& node_def, int v) {
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

bool IsAddN(const NodeDef& node_def, int n) {
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

bool IsNeg(const NodeDef& node_def, const string& input) {
  return node_def.op() == "Neg" && node_def.name() == "neg" &&
         node_def.input_size() == 1 && node_def.input(0) == input;
}

bool GetGraphDef(TF_Graph* graph, GraphDef* graph_def) {
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

bool GetNodeDef(TF_Operation* oper, NodeDef* node_def) {
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

TEST(CAPI, Graph) {
  TF_Status* s = TF_NewStatus();
  TF_Graph* graph = TF_NewGraph();

  // Make a placeholder oper.
  TF_Operation* feed = Placeholder(graph, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  // Test TF_Operation*() query functions.
  EXPECT_EQ(string("feed"), string(TF_OperationName(feed)));
  EXPECT_EQ(string("Placeholder"), string(TF_OperationOpType(feed)));
  EXPECT_EQ(string(""), string(TF_OperationDevice(feed)));
  EXPECT_EQ(1, TF_OperationNumOutputs(feed));
  EXPECT_EQ(TF_INT32, TF_OperationOutputType(TF_Port{feed, 0}));
  EXPECT_EQ(1, TF_OperationOutputListLength(feed, "output", s));
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  EXPECT_EQ(0, TF_OperationNumInputs(feed));
  EXPECT_EQ(0, TF_OperationOutputNumConsumers(TF_Port{feed, 0}));
  EXPECT_EQ(0, TF_OperationNumControlInputs(feed));
  EXPECT_EQ(0, TF_OperationNumControlOutputs(feed));

  tensorflow::AttrValue attr_value;
  ASSERT_TRUE(GetAttrValue(feed, "dtype", &attr_value, s)) << TF_Message(s);
  EXPECT_EQ(attr_value.type(), tensorflow::DT_INT32);

  // Test not found errors in TF_Operation*() query functions.
  EXPECT_EQ(-1, TF_OperationOutputListLength(feed, "bogus", s));
  EXPECT_EQ(TF_INVALID_ARGUMENT, TF_GetCode(s));

  ASSERT_FALSE(GetAttrValue(feed, "missing", &attr_value, s));
  EXPECT_EQ(string("Operation has no attr named 'missing'."),
            string(TF_Message(s)));

  // Make a constant oper with the scalar "3".
  TF_Operation* three = ScalarConst(3, graph, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  // Add oper.
  TF_Operation* add = Add(feed, three, graph, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  // Test TF_Operation*() query functions.
  EXPECT_EQ(string("add"), string(TF_OperationName(add)));
  EXPECT_EQ(string("AddN"), string(TF_OperationOpType(add)));
  EXPECT_EQ(string(""), string(TF_OperationDevice(add)));
  EXPECT_EQ(1, TF_OperationNumOutputs(add));
  EXPECT_EQ(TF_INT32, TF_OperationOutputType(TF_Port{add, 0}));
  EXPECT_EQ(1, TF_OperationOutputListLength(add, "sum", s));
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  EXPECT_EQ(2, TF_OperationNumInputs(add));
  EXPECT_EQ(2, TF_OperationInputListLength(add, "inputs", s));
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  EXPECT_EQ(TF_INT32, TF_OperationInputType(TF_Port{add, 0}));
  EXPECT_EQ(TF_INT32, TF_OperationInputType(TF_Port{add, 1}));
  TF_Port add_in_0 = TF_OperationInput(TF_Port{add, 0});
  EXPECT_EQ(feed, add_in_0.oper);
  EXPECT_EQ(0, add_in_0.index);
  TF_Port add_in_1 = TF_OperationInput(TF_Port{add, 1});
  EXPECT_EQ(three, add_in_1.oper);
  EXPECT_EQ(0, add_in_1.index);
  EXPECT_EQ(0, TF_OperationOutputNumConsumers(TF_Port{add, 0}));
  EXPECT_EQ(0, TF_OperationNumControlInputs(add));
  EXPECT_EQ(0, TF_OperationNumControlOutputs(add));

  ASSERT_TRUE(GetAttrValue(add, "T", &attr_value, s)) << TF_Message(s);
  EXPECT_EQ(attr_value.type(), tensorflow::DT_INT32);
  ASSERT_TRUE(GetAttrValue(add, "N", &attr_value, s)) << TF_Message(s);
  EXPECT_EQ(attr_value.i(), 2);

  // Placeholder oper now has a consumer.
  ASSERT_EQ(1, TF_OperationOutputNumConsumers(TF_Port{feed, 0}));
  TF_Port feed_port;
  EXPECT_EQ(1, TF_OperationOutputConsumers(TF_Port{feed, 0}, &feed_port, 1));
  EXPECT_EQ(add, feed_port.oper);
  EXPECT_EQ(0, feed_port.index);

  // The scalar const oper also has a consumer.
  ASSERT_EQ(1, TF_OperationOutputNumConsumers(TF_Port{three, 0}));
  TF_Port three_port;
  EXPECT_EQ(1, TF_OperationOutputConsumers(TF_Port{three, 0}, &three_port, 1));
  EXPECT_EQ(add, three_port.oper);
  EXPECT_EQ(1, three_port.index);

  // Serialize to GraphDef.
  GraphDef graph_def;
  ASSERT_TRUE(GetGraphDef(graph, &graph_def));

  // Validate GraphDef is what we expect.
  bool found_placeholder = false;
  bool found_scalar_const = false;
  bool found_add = false;
  for (const auto& n : graph_def.node()) {
    if (IsPlaceholder(n)) {
      EXPECT_FALSE(found_placeholder);
      found_placeholder = true;
    } else if (IsScalarConst(n, 3)) {
      EXPECT_FALSE(found_scalar_const);
      found_scalar_const = true;
    } else if (IsAddN(n, 2)) {
      EXPECT_FALSE(found_add);
      found_add = true;
    } else {
      ADD_FAILURE() << "Unexpected NodeDef: " << ProtoDebugString(n);
    }
  }
  EXPECT_TRUE(found_placeholder);
  EXPECT_TRUE(found_scalar_const);
  EXPECT_TRUE(found_add);

  // Add another oper to the graph.
  TF_Operation* neg = Neg(add, graph, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  // Serialize to NodeDef.
  NodeDef node_def;
  ASSERT_TRUE(GetNodeDef(neg, &node_def));

  // Validate NodeDef is what we expect.
  EXPECT_TRUE(IsNeg(node_def, "add"));

  // Serialize to GraphDef.
  GraphDef graph_def2;
  ASSERT_TRUE(GetGraphDef(graph, &graph_def2));

  // Compare with first GraphDef + added NodeDef.
  NodeDef* added_node = graph_def.add_node();
  *added_node = node_def;
  EXPECT_EQ(ProtoDebugString(graph_def), ProtoDebugString(graph_def2));

  // Look up some nodes by name.
  TF_Operation* neg2 = TF_GraphOperationByName(graph, "neg");
  EXPECT_TRUE(neg == neg2);
  NodeDef node_def2;
  ASSERT_TRUE(GetNodeDef(neg2, &node_def2));
  EXPECT_EQ(ProtoDebugString(node_def), ProtoDebugString(node_def2));

  TF_Operation* feed2 = TF_GraphOperationByName(graph, "feed");
  EXPECT_TRUE(feed == feed2);
  ASSERT_TRUE(GetNodeDef(feed, &node_def));
  ASSERT_TRUE(GetNodeDef(feed2, &node_def2));
  EXPECT_EQ(ProtoDebugString(node_def), ProtoDebugString(node_def2));

  // Test iterating through the nodes of a graph.
  found_placeholder = false;
  found_scalar_const = false;
  found_add = false;
  bool found_neg = false;
  size_t pos = 0;
  TF_Operation* oper;
  while ((oper = TF_GraphNextOperation(graph, &pos)) != nullptr) {
    if (oper == feed) {
      EXPECT_FALSE(found_placeholder);
      found_placeholder = true;
    } else if (oper == three) {
      EXPECT_FALSE(found_scalar_const);
      found_scalar_const = true;
    } else if (oper == add) {
      EXPECT_FALSE(found_add);
      found_add = true;
    } else if (oper == neg) {
      EXPECT_FALSE(found_neg);
      found_neg = true;
    } else {
      ASSERT_TRUE(GetNodeDef(oper, &node_def));
      ADD_FAILURE() << "Unexpected Node: " << ProtoDebugString(node_def);
    }
  }
  EXPECT_TRUE(found_placeholder);
  EXPECT_TRUE(found_scalar_const);
  EXPECT_TRUE(found_add);
  EXPECT_TRUE(found_neg);

  // Clean up
  TF_DeleteGraph(graph);
  TF_DeleteStatus(s);
}

class CSessionWithGraph {
 public:
  CSessionWithGraph(TF_Graph* graph, TF_Status* s) {
    TF_SessionOptions* opts = TF_NewSessionOptions();
    session_ = TF_NewSessionWithGraph(graph, opts, s);
    TF_DeleteSessionOptions(opts);
  }

  ~CSessionWithGraph() {
    TF_Status* s = TF_NewStatus();
    CloseAndDelete(s);
    EXPECT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
    TF_DeleteStatus(s);
  }

  void SetInputs(
      std::initializer_list<std::pair<TF_Operation*, TF_Tensor*>> inputs) {
    DeleteInputValues();
    inputs_.clear();
    for (const auto& p : inputs) {
      inputs_.emplace_back(TF_Port{p.first, 0});
      input_values_.emplace_back(p.second);
    }
  }

  void SetOutputs(std::initializer_list<TF_Operation*> outputs) {
    ResetOutputValues();
    outputs_.clear();
    for (TF_Operation* o : outputs) {
      outputs_.emplace_back(TF_Port{o, 0});
    }
  }

  void SetTargets(std::initializer_list<TF_Operation*> targets) {
    targets_.clear();
    for (TF_Operation* t : targets) {
      targets_.emplace_back(t);
    }
  }

  void Run(TF_Status* s) {
    if (inputs_.size() != input_values_.size()) {
      ADD_FAILURE() << "Call SetInputs() before Run()";
      return;
    }
    ResetOutputValues();
    output_values_.resize(outputs_.size(), nullptr);

    const TF_Port* inputs_ptr = inputs_.empty() ? nullptr : &inputs_[0];
    TF_Tensor* const* input_values_ptr =
        input_values_.empty() ? nullptr : &input_values_[0];

    const TF_Port* outputs_ptr = outputs_.empty() ? nullptr : &outputs_[0];
    TF_Tensor** output_values_ptr =
        output_values_.empty() ? nullptr : &output_values_[0];

    TF_Operation* const* targets_ptr =
        targets_.empty() ? nullptr : &targets_[0];

    TF_SessionRun(session_, nullptr, inputs_ptr, input_values_ptr,
                  inputs_.size(), outputs_ptr, output_values_ptr,
                  outputs_.size(), targets_ptr, targets_.size(), nullptr, s);

    // TF_SessionRun() takes ownership of the tensors in input_values_.
    input_values_.clear();
  }

  void CloseAndDelete(TF_Status* s) {
    DeleteInputValues();
    ResetOutputValues();
    if (session_ != nullptr) {
      TF_CloseSessionWithGraph(session_, s);
      EXPECT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
      TF_DeleteSessionWithGraph(session_, s);
      session_ = nullptr;
    }
  }

  TF_Tensor* output_tensor(int i) { return output_values_[i]; }

 private:
  void DeleteInputValues() {
    for (int i = 0; i < input_values_.size(); ++i) {
      TF_DeleteTensor(input_values_[i]);
    }
    input_values_.clear();
  }

  void ResetOutputValues() {
    for (int i = 0; i < output_values_.size(); ++i) {
      if (output_values_[i] != nullptr) TF_DeleteTensor(output_values_[i]);
    }
    output_values_.clear();
  }

  TF_SessionWithGraph* session_;
  std::vector<TF_Port> inputs_;
  std::vector<TF_Tensor*> input_values_;
  std::vector<TF_Port> outputs_;
  std::vector<TF_Tensor*> output_values_;
  std::vector<TF_Operation*> targets_;
};

TEST(CAPI, SessionWithGraph) {
  TF_Status* s = TF_NewStatus();
  TF_Graph* graph = TF_NewGraph();

  // Make a placeholder operation.
  TF_Operation* feed = Placeholder(graph, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  // Make a constant operation with the scalar "2".
  TF_Operation* two = ScalarConst(2, graph, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  // Add operation.
  TF_Operation* add = Add(feed, two, graph, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  // Create a session for this graph.
  CSessionWithGraph csession(graph, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  // Run the graph.
  csession.SetInputs({{feed, Int32Tensor(3)}});
  csession.SetOutputs({add});
  csession.Run(s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  TF_Tensor* out = csession.output_tensor(0);
  ASSERT_TRUE(out != nullptr);
  EXPECT_EQ(TF_INT32, TF_TensorType(out));
  EXPECT_EQ(0, TF_NumDims(out));  // scalar
  ASSERT_EQ(sizeof(tensorflow::int32), TF_TensorByteSize(out));
  tensorflow::int32* output_contents =
      static_cast<tensorflow::int32*>(TF_TensorData(out));
  EXPECT_EQ(3 + 2, *output_contents);

  // Add another operation to the graph.
  TF_Operation* neg = Neg(add, graph, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  // Run up to the new operation.
  csession.SetInputs({{feed, Int32Tensor(7)}});
  csession.SetOutputs({neg});
  csession.Run(s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  out = csession.output_tensor(0);
  ASSERT_TRUE(out != nullptr);
  EXPECT_EQ(TF_INT32, TF_TensorType(out));
  EXPECT_EQ(0, TF_NumDims(out));  // scalar
  ASSERT_EQ(sizeof(tensorflow::int32), TF_TensorByteSize(out));
  output_contents = static_cast<tensorflow::int32*>(TF_TensorData(out));
  EXPECT_EQ(-(7 + 2), *output_contents);

  // Clean up
  csession.CloseAndDelete(s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  TF_DeleteGraph(graph);
  TF_DeleteStatus(s);
}

// TODO(josh11b): Test:
// * TF_SetDevice(desc, "/job:worker");
// * control inputs / outputs
// * targets
// * TF_DeleteGraph() before TF_DeleteSessionWithGraph()

}  // namespace
