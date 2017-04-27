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

#include <algorithm>
#include <iterator>
#include <memory>
#include <vector>
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#include "tensorflow/core/framework/graph.pb_text.h"
#include "tensorflow/core/framework/node_def.pb_text.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/util/equal_graph_def.h"

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

typedef std::unique_ptr<TF_Tensor, decltype(&TF_DeleteTensor)>
    unique_tensor_ptr;

TEST(CAPI, Version) { EXPECT_NE("", string(TF_Version())); }

TEST(CAPI, Status) {
  TF_Status* s = TF_NewStatus();
  EXPECT_EQ(TF_OK, TF_GetCode(s));
  EXPECT_EQ(string(), TF_Message(s));
  TF_SetStatus(s, TF_CANCELLED, "cancel");
  EXPECT_EQ(TF_CANCELLED, TF_GetCode(s));
  EXPECT_EQ(string("cancel"), TF_Message(s));
  TF_DeleteStatus(s);
}

static void Deallocator(void* data, size_t, void* arg) {
  tensorflow::cpu_allocator()->DeallocateRaw(data);
  *reinterpret_cast<bool*>(arg) = true;
}

TEST(CAPI, Tensor) {
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

TEST(CAPI, AllocateTensor) {
  const int num_bytes = 6 * sizeof(float);
  int64_t dims[] = {2, 3};
  TF_Tensor* t = TF_AllocateTensor(TF_FLOAT, dims, 2, num_bytes);
  EXPECT_EQ(TF_FLOAT, TF_TensorType(t));
  EXPECT_EQ(2, TF_NumDims(t));
  EXPECT_EQ(dims[0], TF_Dim(t, 0));
  EXPECT_EQ(dims[1], TF_Dim(t, 1));
  EXPECT_EQ(num_bytes, TF_TensorByteSize(t));
  TF_DeleteTensor(t);
}

TEST(CAPI, MaybeMove) {
  const int num_bytes = 6 * sizeof(float);
  float* values =
      reinterpret_cast<float*>(tensorflow::cpu_allocator()->AllocateRaw(
          EIGEN_MAX_ALIGN_BYTES, num_bytes));
  int64_t dims[] = {2, 3};
  bool deallocator_called = false;
  TF_Tensor* t = TF_NewTensor(TF_FLOAT, dims, 2, values, num_bytes,
                              &Deallocator, &deallocator_called);

  TF_Tensor* o = TF_TensorMaybeMove(t);
  ASSERT_TRUE(o == nullptr);  // It is unsafe to move memory TF might not own.
  TF_DeleteTensor(t);
  EXPECT_TRUE(deallocator_called);
}

TEST(CAPI, LibraryLoadFunctions) {
  // Load the library.
  TF_Status* status = TF_NewStatus();
  TF_Library* lib =
      TF_LoadLibrary("tensorflow/c/test_op.so", status);
  TF_Code code = TF_GetCode(status);
  string status_msg(TF_Message(status));
  TF_DeleteStatus(status);
  ASSERT_EQ(TF_OK, code) << status_msg;

  // Test op list.
  TF_Buffer op_list_buf = TF_GetOpList(lib);
  tensorflow::OpList op_list;
  EXPECT_TRUE(op_list.ParseFromArray(op_list_buf.data, op_list_buf.length));
  ASSERT_EQ(op_list.op_size(), 1);
  EXPECT_EQ("TestCApi", op_list.op(0).name());

  TF_DeleteLibraryHandle(lib);
}

static void TestEncodeDecode(int line, const std::vector<string>& data) {
  const tensorflow::int64 n = data.size();
  for (const std::vector<tensorflow::int64>& dims :
       std::vector<std::vector<tensorflow::int64>>{
           {n}, {1, n}, {n, 1}, {n / 2, 2}}) {
    // Create C++ Tensor
    Tensor src(tensorflow::DT_STRING, TensorShape(dims));
    for (tensorflow::int64 i = 0; i < src.NumElements(); ++i) {
      src.flat<string>()(i) = data[i];
    }
    TF_Tensor* dst = TF_Tensor_EncodeStrings(src);

    // Convert back to a C++ Tensor and ensure we get expected output.
    TF_Status* status = TF_NewStatus();
    Tensor output;
    ASSERT_TRUE(TF_Tensor_DecodeStrings(dst, &output, status)) << line;
    ASSERT_EQ(TF_OK, TF_GetCode(status)) << line;
    ASSERT_EQ(src.NumElements(), output.NumElements()) << line;
    for (tensorflow::int64 i = 0; i < src.NumElements(); ++i) {
      ASSERT_EQ(data[i], output.flat<string>()(i)) << line;
    }

    TF_DeleteStatus(status);
    TF_DeleteTensor(dst);
  }
}

TEST(CAPI, TensorEncodeDecodeStrings) {
  TestEncodeDecode(__LINE__, {});
  TestEncodeDecode(__LINE__, {"hello"});
  TestEncodeDecode(__LINE__,
                   {"the", "quick", "brown", "fox", "jumped", "over"});

  string big(1000, 'a');
  TestEncodeDecode(__LINE__, {"small", big, "small2"});
}

TEST(CAPI, SessionOptions) {
  TF_SessionOptions* opt = TF_NewSessionOptions();
  TF_DeleteSessionOptions(opt);
}

TEST(CAPI, DeprecatedSession) {
  TF_Status* s = TF_NewStatus();
  TF_SessionOptions* opt = TF_NewSessionOptions();
  TF_DeprecatedSession* session = TF_NewDeprecatedSession(opt, s);
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

  TF_DeleteDeprecatedSession(session, s);
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
  EXPECT_EQ(TF_DataTypeSize(TF_DOUBLE),
            tensorflow::DataTypeSize(tensorflow::DT_DOUBLE));
  EXPECT_EQ(TF_DataTypeSize(TF_STRING),
            tensorflow::DataTypeSize(tensorflow::DT_STRING));
  // Test with invalid type; should always return 0 as documented
  EXPECT_EQ(TF_DataTypeSize(static_cast<TF_DataType>(0)), 0);
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

TEST(CAPI, GetAllOpList) {
  TF_Buffer* buf = TF_GetAllOpList();
  tensorflow::OpList op_list;
  EXPECT_TRUE(op_list.ParseFromArray(buf->data, buf->length));
  EXPECT_GT(op_list.op_size(), 0);
  TF_DeleteBuffer(buf);
}

static void Int32Deallocator(void* data, size_t, void* arg) {
  delete[] static_cast<int32*>(data);
}

static TF_Tensor* Int32Tensor(int32 v) {
  const int num_bytes = sizeof(int32);
  int32* values = new int32[1];
  values[0] = v;
  return TF_NewTensor(TF_INT32, nullptr, 0, values, num_bytes,
                      &Int32Deallocator, nullptr);
}

TF_Operation* Placeholder(TF_Graph* graph, TF_Status* s,
                          const char* name = "feed") {
  TF_OperationDescription* desc = TF_NewOperation(graph, "Placeholder", name);
  TF_SetAttrType(desc, "dtype", TF_INT32);
  return TF_FinishOperation(desc, s);
}

TF_Operation* ScalarConst(int32 v, TF_Graph* graph, TF_Status* s,
                          const char* name = "scalar") {
  unique_tensor_ptr tensor(Int32Tensor(v), TF_DeleteTensor);
  TF_OperationDescription* desc = TF_NewOperation(graph, "Const", name);
  TF_SetAttrTensor(desc, "value", tensor.get(), s);
  if (TF_GetCode(s) != TF_OK) return nullptr;
  TF_SetAttrType(desc, "dtype", TF_INT32);
  return TF_FinishOperation(desc, s);
}

TF_Operation* Add(TF_Operation* l, TF_Operation* r, TF_Graph* graph,
                  TF_Status* s, const char* name = "add") {
  TF_OperationDescription* desc = TF_NewOperation(graph, "AddN", name);
  TF_Output add_inputs[2] = {{l, 0}, {r, 0}};
  TF_AddInputList(desc, add_inputs, 2);
  return TF_FinishOperation(desc, s);
}

TF_Operation* Add(TF_Output l, TF_Output r, TF_Graph* graph, TF_Status* s,
                  const char* name = "add") {
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

TEST(CAPI, SetShape) {
  TF_Status* s = TF_NewStatus();
  TF_Graph* graph = TF_NewGraph();

  TF_Operation* feed = Placeholder(graph, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  TF_Output feed_out_0 = TF_Output{feed, 0};
  int num_dims;

  // Fetch the shape, it should be completely unknown.
  num_dims = TF_GraphGetTensorNumDims(graph, feed_out_0, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  EXPECT_EQ(-1, num_dims);

  // Set the shape to be 2 x Unknown
  int64_t dims[] = {2, -1};
  TF_GraphSetTensorShape(graph, feed_out_0, dims, 2, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  // Fetch the shape and validate it is 2 by -1.
  num_dims = TF_GraphGetTensorNumDims(graph, feed_out_0, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  EXPECT_EQ(2, num_dims);

  // Resize the dimension vector appropriately.
  int64_t returned_dims[2];
  TF_GraphGetTensorShape(graph, feed_out_0, returned_dims, num_dims, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  EXPECT_EQ(dims[0], returned_dims[0]);
  EXPECT_EQ(dims[1], returned_dims[1]);

  // Set to a new valid shape: [2, 3]
  dims[1] = 3;
  TF_GraphSetTensorShape(graph, feed_out_0, dims, 2, s);
  EXPECT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  // Fetch and see that the new value is returned.
  TF_GraphGetTensorShape(graph, feed_out_0, returned_dims, num_dims, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  EXPECT_EQ(dims[0], returned_dims[0]);
  EXPECT_EQ(dims[1], returned_dims[1]);

  // Try to set 'unknown' on the shape and see that
  // it doesn't change.
  dims[0] = -1;
  dims[1] = -1;
  TF_GraphSetTensorShape(graph, feed_out_0, dims, 2, s);
  EXPECT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  // Fetch and see that the new value is returned.
  TF_GraphGetTensorShape(graph, feed_out_0, returned_dims, num_dims, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  EXPECT_EQ(2, num_dims);
  EXPECT_EQ(2, returned_dims[0]);
  EXPECT_EQ(3, returned_dims[1]);

  // Try to fetch a shape with the wrong num_dims
  TF_GraphGetTensorShape(graph, feed_out_0, returned_dims, 5, s);
  EXPECT_EQ(TF_INVALID_ARGUMENT, TF_GetCode(s)) << TF_Message(s);

  // Try to set an invalid shape (cannot change 2x3 to a 2x5).
  dims[1] = 5;
  TF_GraphSetTensorShape(graph, feed_out_0, dims, 2, s);
  EXPECT_EQ(TF_INVALID_ARGUMENT, TF_GetCode(s)) << TF_Message(s);

  // Test for a scalar.
  TF_Operation* three = ScalarConst(3, graph, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  TF_Output three_out_0 = TF_Output{three, 0};

  num_dims = TF_GraphGetTensorNumDims(graph, three_out_0, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  EXPECT_EQ(0, num_dims);
  TF_GraphGetTensorShape(graph, three_out_0, returned_dims, num_dims, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  // Clean up
  TF_DeleteGraph(graph);
  TF_DeleteStatus(s);
}

TEST(CAPI, Graph) {
  TF_Status* s = TF_NewStatus();
  TF_Graph* graph = TF_NewGraph();

  // Make a placeholder operation.
  TF_Operation* feed = Placeholder(graph, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  // Test TF_Operation*() query functions.
  EXPECT_EQ(string("feed"), string(TF_OperationName(feed)));
  EXPECT_EQ(string("Placeholder"), string(TF_OperationOpType(feed)));
  EXPECT_EQ(string(""), string(TF_OperationDevice(feed)));
  EXPECT_EQ(1, TF_OperationNumOutputs(feed));
  EXPECT_EQ(TF_INT32, TF_OperationOutputType(TF_Output{feed, 0}));
  EXPECT_EQ(1, TF_OperationOutputListLength(feed, "output", s));
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  EXPECT_EQ(0, TF_OperationNumInputs(feed));
  EXPECT_EQ(0, TF_OperationOutputNumConsumers(TF_Output{feed, 0}));
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
  EXPECT_EQ(TF_INT32, TF_OperationOutputType(TF_Output{add, 0}));
  EXPECT_EQ(1, TF_OperationOutputListLength(add, "sum", s));
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  EXPECT_EQ(2, TF_OperationNumInputs(add));
  EXPECT_EQ(2, TF_OperationInputListLength(add, "inputs", s));
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  EXPECT_EQ(TF_INT32, TF_OperationInputType(TF_Input{add, 0}));
  EXPECT_EQ(TF_INT32, TF_OperationInputType(TF_Input{add, 1}));
  TF_Output add_in_0 = TF_OperationInput(TF_Input{add, 0});
  EXPECT_EQ(feed, add_in_0.oper);
  EXPECT_EQ(0, add_in_0.index);
  TF_Output add_in_1 = TF_OperationInput(TF_Input{add, 1});
  EXPECT_EQ(three, add_in_1.oper);
  EXPECT_EQ(0, add_in_1.index);
  EXPECT_EQ(0, TF_OperationOutputNumConsumers(TF_Output{add, 0}));
  EXPECT_EQ(0, TF_OperationNumControlInputs(add));
  EXPECT_EQ(0, TF_OperationNumControlOutputs(add));

  ASSERT_TRUE(GetAttrValue(add, "T", &attr_value, s)) << TF_Message(s);
  EXPECT_EQ(attr_value.type(), tensorflow::DT_INT32);
  ASSERT_TRUE(GetAttrValue(add, "N", &attr_value, s)) << TF_Message(s);
  EXPECT_EQ(attr_value.i(), 2);

  // Placeholder oper now has a consumer.
  ASSERT_EQ(1, TF_OperationOutputNumConsumers(TF_Output{feed, 0}));
  TF_Input feed_port;
  EXPECT_EQ(1, TF_OperationOutputConsumers(TF_Output{feed, 0}, &feed_port, 1));
  EXPECT_EQ(add, feed_port.oper);
  EXPECT_EQ(0, feed_port.index);

  // The scalar const oper also has a consumer.
  ASSERT_EQ(1, TF_OperationOutputNumConsumers(TF_Output{three, 0}));
  TF_Input three_port;
  EXPECT_EQ(1,
            TF_OperationOutputConsumers(TF_Output{three, 0}, &three_port, 1));
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

/*
TODO(skyewm): this test currently DCHECKs, change to bad status

TEST(CAPI, InputFromDifferentGraphError) {
  TF_Status* s = TF_NewStatus();
  TF_Graph* g1 = TF_NewGraph();
  TF_Graph* g2 = TF_NewGraph();

  TF_Operation* feed = Placeholder(g1, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  // Attempt to create node in g2 with input from g1
  Neg(feed, g2, s);
  EXPECT_EQ(TF_INVALID_ARGUMENT, TF_GetCode(s));
  EXPECT_STREQ("foo", TF_Message(s));

  TF_DeleteGraph(g1);
  TF_DeleteGraph(g2);
  TF_DeleteStatus(s);
}
*/

TEST(CAPI, ImportGraphDef) {
  TF_Status* s = TF_NewStatus();
  TF_Graph* graph = TF_NewGraph();

  // Create a graph with two nodes: x and 3
  Placeholder(graph, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  ASSERT_TRUE(TF_GraphOperationByName(graph, "feed") != nullptr);
  TF_Operation* oper = ScalarConst(3, graph, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  ASSERT_TRUE(TF_GraphOperationByName(graph, "scalar") != nullptr);
  Neg(oper, graph, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  ASSERT_TRUE(TF_GraphOperationByName(graph, "neg") != nullptr);

  // Export to a GraphDef
  TF_Buffer* graph_def = TF_NewBuffer();
  TF_GraphToGraphDef(graph, graph_def, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  // Import it again, with a prefix, in a fresh graph.
  TF_DeleteGraph(graph);
  graph = TF_NewGraph();
  TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();
  TF_ImportGraphDefOptionsSetPrefix(opts, "imported");
  TF_GraphImportGraphDef(graph, graph_def, opts, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  TF_Operation* scalar = TF_GraphOperationByName(graph, "imported/scalar");
  TF_Operation* feed = TF_GraphOperationByName(graph, "imported/feed");
  TF_Operation* neg = TF_GraphOperationByName(graph, "imported/neg");
  ASSERT_TRUE(scalar != nullptr);
  ASSERT_TRUE(feed != nullptr);
  ASSERT_TRUE(neg != nullptr);

  // Import it again, with an input mapping and return outputs, into the same
  // graph.
  TF_DeleteImportGraphDefOptions(opts);
  opts = TF_NewImportGraphDefOptions();
  TF_ImportGraphDefOptionsSetPrefix(opts, "imported2");
  TF_ImportGraphDefOptionsAddInputMapping(opts, "scalar", 0, {scalar, 0});
  TF_ImportGraphDefOptionsAddReturnOutput(opts, "feed", 0);
  TF_ImportGraphDefOptionsAddReturnOutput(opts, "scalar", 0);
  EXPECT_EQ(2, TF_ImportGraphDefOptionsNumReturnOutputs(opts));
  TF_Output return_outputs[2];
  TF_GraphImportGraphDefWithReturnOutputs(graph, graph_def, opts,
                                          return_outputs, 2, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  TF_Operation* scalar2 = TF_GraphOperationByName(graph, "imported2/scalar");
  TF_Operation* feed2 = TF_GraphOperationByName(graph, "imported2/feed");
  TF_Operation* neg2 = TF_GraphOperationByName(graph, "imported2/neg");
  ASSERT_TRUE(scalar2 != nullptr);
  ASSERT_TRUE(feed2 != nullptr);
  ASSERT_TRUE(neg2 != nullptr);

  // Check input mapping
  TF_Output neg_input = TF_OperationInput({neg, 0});
  EXPECT_EQ(scalar, neg_input.oper);
  EXPECT_EQ(0, neg_input.index);

  // Check return outputs
  EXPECT_EQ(feed2, return_outputs[0].oper);
  EXPECT_EQ(0, return_outputs[0].index);
  EXPECT_EQ(scalar, return_outputs[1].oper);  // remapped
  EXPECT_EQ(0, return_outputs[1].index);

  // Import again, with control dependencies, into the same graph.
  TF_DeleteImportGraphDefOptions(opts);
  opts = TF_NewImportGraphDefOptions();
  TF_ImportGraphDefOptionsSetPrefix(opts, "imported3");
  TF_ImportGraphDefOptionsAddControlDependency(opts, feed);
  TF_ImportGraphDefOptionsAddControlDependency(opts, feed2);
  TF_GraphImportGraphDef(graph, graph_def, opts, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  TF_Operation* scalar3 = TF_GraphOperationByName(graph, "imported3/scalar");
  TF_Operation* feed3 = TF_GraphOperationByName(graph, "imported3/feed");
  TF_Operation* neg3 = TF_GraphOperationByName(graph, "imported3/neg");
  ASSERT_TRUE(scalar3 != nullptr);
  ASSERT_TRUE(feed3 != nullptr);
  ASSERT_TRUE(neg3 != nullptr);

  // Check that newly-imported scalar and feed have control deps (neg3 will
  // inherit them from input)
  TF_Operation* control_inputs[100];
  int num_control_inputs = TF_OperationGetControlInputs(
      scalar3, control_inputs, TF_OperationNumControlInputs(scalar3));
  ASSERT_EQ(2, num_control_inputs);
  EXPECT_EQ(feed, control_inputs[0]);
  EXPECT_EQ(feed2, control_inputs[1]);

  num_control_inputs = TF_OperationGetControlInputs(
      feed3, control_inputs, TF_OperationNumControlInputs(feed3));
  ASSERT_EQ(2, num_control_inputs);
  EXPECT_EQ(feed, control_inputs[0]);
  EXPECT_EQ(feed2, control_inputs[1]);

  // Export to a graph def so we can import a graph with control dependencies
  TF_DeleteBuffer(graph_def);
  graph_def = TF_NewBuffer();
  TF_GraphToGraphDef(graph, graph_def, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  // Import again, with remapped control dependency, into the same graph
  TF_DeleteImportGraphDefOptions(opts);
  opts = TF_NewImportGraphDefOptions();
  TF_ImportGraphDefOptionsSetPrefix(opts, "imported4");
  TF_ImportGraphDefOptionsRemapControlDependency(opts, "imported/feed", feed);
  TF_GraphImportGraphDef(graph, graph_def, opts, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  TF_Operation* scalar4 =
      TF_GraphOperationByName(graph, "imported4/imported3/scalar");
  TF_Operation* feed4 =
      TF_GraphOperationByName(graph, "imported4/imported2/feed");

  // Check that imported `imported3/scalar` has remapped control dep from
  // original graph and imported control dep
  num_control_inputs = TF_OperationGetControlInputs(
      scalar4, control_inputs, TF_OperationNumControlInputs(scalar4));
  ASSERT_EQ(2, num_control_inputs);
  EXPECT_EQ(feed, control_inputs[0]);
  EXPECT_EQ(feed4, control_inputs[1]);

  TF_DeleteImportGraphDefOptions(opts);
  TF_DeleteBuffer(graph_def);

  // Can add nodes to the imported graph without trouble.
  Add(feed, scalar, graph, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  TF_DeleteGraph(graph);
  TF_DeleteStatus(s);
}

class CSession {
 public:
  CSession(TF_Graph* graph, TF_Status* s) {
    TF_SessionOptions* opts = TF_NewSessionOptions();
    session_ = TF_NewSession(graph, opts, s);
    TF_DeleteSessionOptions(opts);
  }

  CSession(TF_Session* session) { session_ = session; }

  ~CSession() {
    TF_Status* s = TF_NewStatus();
    CloseAndDelete(s);
    EXPECT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
    TF_DeleteStatus(s);
  }

  void SetInputs(std::vector<std::pair<TF_Operation*, TF_Tensor*>> inputs) {
    DeleteInputValues();
    inputs_.clear();
    for (const auto& p : inputs) {
      inputs_.emplace_back(TF_Output{p.first, 0});
      input_values_.emplace_back(p.second);
    }
  }

  void SetOutputs(std::initializer_list<TF_Operation*> outputs) {
    ResetOutputValues();
    outputs_.clear();
    for (TF_Operation* o : outputs) {
      outputs_.emplace_back(TF_Output{o, 0});
    }
  }

  void SetOutputs(const std::vector<TF_Output>& outputs) {
    ResetOutputValues();
    outputs_ = outputs;
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

    const TF_Output* inputs_ptr = inputs_.empty() ? nullptr : &inputs_[0];
    TF_Tensor* const* input_values_ptr =
        input_values_.empty() ? nullptr : &input_values_[0];

    const TF_Output* outputs_ptr = outputs_.empty() ? nullptr : &outputs_[0];
    TF_Tensor** output_values_ptr =
        output_values_.empty() ? nullptr : &output_values_[0];

    TF_Operation* const* targets_ptr =
        targets_.empty() ? nullptr : &targets_[0];

    TF_SessionRun(session_, nullptr, inputs_ptr, input_values_ptr,
                  inputs_.size(), outputs_ptr, output_values_ptr,
                  outputs_.size(), targets_ptr, targets_.size(), nullptr, s);

    DeleteInputValues();
  }

  void CloseAndDelete(TF_Status* s) {
    DeleteInputValues();
    ResetOutputValues();
    if (session_ != nullptr) {
      TF_CloseSession(session_, s);
      EXPECT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
      TF_DeleteSession(session_, s);
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

  TF_Session* session_;
  std::vector<TF_Output> inputs_;
  std::vector<TF_Tensor*> input_values_;
  std::vector<TF_Output> outputs_;
  std::vector<TF_Tensor*> output_values_;
  std::vector<TF_Operation*> targets_;
};

TEST(CAPI, Session) {
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
  CSession csession(graph, s);
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
  ASSERT_EQ(sizeof(int32), TF_TensorByteSize(out));
  int32* output_contents = static_cast<int32*>(TF_TensorData(out));
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
  ASSERT_EQ(sizeof(int32), TF_TensorByteSize(out));
  output_contents = static_cast<int32*>(TF_TensorData(out));
  EXPECT_EQ(-(7 + 2), *output_contents);

  // Clean up
  csession.CloseAndDelete(s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  TF_DeleteGraph(graph);
  TF_DeleteStatus(s);
}

TEST(CAPI, SessionPRun) {
  TF_Status* s = TF_NewStatus();
  TF_Graph* graph = TF_NewGraph();

  // Construct the graph: A + 2 + B
  TF_Operation* a = Placeholder(graph, s, "A");
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  TF_Operation* b = Placeholder(graph, s, "B");
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  TF_Operation* two = ScalarConst(2, graph, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  TF_Operation* plus2 = Add(a, two, graph, s, "plus2");
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  TF_Operation* plusB = Add(plus2, b, graph, s, "plusB");
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  // Setup a session and a partial run handle.  The partial run will allow
  // computation of A + 2 + B in two phases (calls to TF_SessionPRun):
  // 1. Feed A and get (A+2)
  // 2. Feed B and get (A+2)+B
  TF_SessionOptions* opts = TF_NewSessionOptions();
  TF_Session* sess = TF_NewSession(graph, opts, s);
  TF_DeleteSessionOptions(opts);

  TF_Output feeds[] = {TF_Output{a, 0}, TF_Output{b, 0}};
  TF_Output fetches[] = {TF_Output{plus2, 0}, TF_Output{plusB, 0}};

  const char* handle = nullptr;
  TF_SessionPRunSetup(sess, feeds, TF_ARRAYSIZE(feeds), fetches,
                      TF_ARRAYSIZE(fetches), nullptr, 0, &handle, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  // Feed A and fetch A + 2.
  TF_Output feeds1[] = {TF_Output{a, 0}};
  TF_Output fetches1[] = {TF_Output{plus2, 0}};
  TF_Tensor* feedValues1[] = {Int32Tensor(1)};
  TF_Tensor* fetchValues1[1];
  TF_SessionPRun(sess, handle, feeds1, feedValues1, 1, fetches1, fetchValues1,
                 1, nullptr, 0, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  EXPECT_EQ(3, *(static_cast<int32*>(TF_TensorData(fetchValues1[0]))));
  TF_DeleteTensor(feedValues1[0]);
  TF_DeleteTensor(fetchValues1[0]);

  // Feed B and fetch (A + 2) + B.
  TF_Output feeds2[] = {TF_Output{b, 0}};
  TF_Output fetches2[] = {TF_Output{plusB, 0}};
  TF_Tensor* feedValues2[] = {Int32Tensor(4)};
  TF_Tensor* fetchValues2[1];
  TF_SessionPRun(sess, handle, feeds2, feedValues2, 1, fetches2, fetchValues2,
                 1, nullptr, 0, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  EXPECT_EQ(7, *(static_cast<int32*>(TF_TensorData(fetchValues2[0]))));
  TF_DeleteTensor(feedValues2[0]);
  TF_DeleteTensor(fetchValues2[0]);

  // Clean up.
  TF_DeletePRunHandle(handle);
  TF_DeleteSession(sess, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  TF_DeleteGraph(graph);
  TF_DeleteStatus(s);
}

TEST(CAPI, ColocateWith) {
  TF_Status* s = TF_NewStatus();
  TF_Graph* graph = TF_NewGraph();

  TF_Operation* feed = Placeholder(graph, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  TF_Operation* constant = ScalarConst(10, graph, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  TF_OperationDescription* desc = TF_NewOperation(graph, "AddN", "add");
  TF_Output inputs[] = {{feed, 0}, {constant, 0}};
  TF_AddInputList(desc, inputs, TF_ARRAYSIZE(inputs));
  TF_ColocateWith(desc, feed);
  TF_Operation* add = TF_FinishOperation(desc, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  TF_AttrMetadata m =
      TF_OperationGetAttrMetadata(add, tensorflow::kColocationAttrName, s);
  EXPECT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  EXPECT_EQ(1, m.is_list);
  EXPECT_EQ(1, m.list_size);
  EXPECT_EQ(TF_ATTR_STRING, m.type);
  void* values[1];
  size_t lens[1];
  std::unique_ptr<char[]> storage(new char[m.total_size]);
  TF_OperationGetAttrStringList(add, tensorflow::kColocationAttrName, values,
                                lens, 1, storage.get(), m.total_size, s);
  EXPECT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  EXPECT_EQ("loc:@feed", string(static_cast<const char*>(values[0]), lens[0]));

  TF_DeleteGraph(graph);
  TF_DeleteStatus(s);
}

TEST(CAPI, SavedModel) {
  // Load the saved model.
  const char kSavedModel[] = "cc/saved_model/testdata/half_plus_two/00000123";
  const string saved_model_dir = tensorflow::io::JoinPath(
      tensorflow::testing::TensorFlowSrcRoot(), kSavedModel);
  TF_SessionOptions* opt = TF_NewSessionOptions();
  TF_Buffer* run_options = TF_NewBufferFromString("", 0);
  TF_Buffer* metagraph = TF_NewBuffer();
  TF_Status* s = TF_NewStatus();
  const char* tags[] = {tensorflow::kSavedModelTagServe};
  TF_Graph* graph = TF_NewGraph();
  TF_Session* session = TF_LoadSessionFromSavedModel(
      opt, run_options, saved_model_dir.c_str(), tags, 1, graph, metagraph, s);
  TF_DeleteBuffer(run_options);
  TF_DeleteSessionOptions(opt);
  tensorflow::MetaGraphDef metagraph_def;
  metagraph_def.ParseFromArray(metagraph->data, metagraph->length);
  TF_DeleteBuffer(metagraph);

  EXPECT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  CSession csession(session);

  // Retrieve the regression signature from meta graph def.
  const auto signature_def_map = metagraph_def.signature_def();
  const auto signature_def = signature_def_map.at("regress_x_to_y");

  const string input_name =
      signature_def.inputs().at(tensorflow::kRegressInputs).name();
  const string output_name =
      signature_def.outputs().at(tensorflow::kRegressOutputs).name();

  // Write {0, 1, 2, 3} as tensorflow::Example inputs.
  Tensor input(tensorflow::DT_STRING, TensorShape({4}));
  for (tensorflow::int64 i = 0; i < input.NumElements(); ++i) {
    tensorflow::Example example;
    auto* feature_map = example.mutable_features()->mutable_feature();
    (*feature_map)["x"].mutable_float_list()->add_value(i);
    input.flat<string>()(i) = example.SerializeAsString();
  }

  const tensorflow::string input_op_name =
      tensorflow::ParseTensorName(input_name).first.ToString();
  TF_Operation* input_op =
      TF_GraphOperationByName(graph, input_op_name.c_str());
  ASSERT_TRUE(input_op != nullptr);
  csession.SetInputs({{input_op, TF_Tensor_EncodeStrings(input)}});

  const tensorflow::string output_op_name =
      tensorflow::ParseTensorName(output_name).first.ToString();
  TF_Operation* output_op =
      TF_GraphOperationByName(graph, output_op_name.c_str());
  ASSERT_TRUE(output_op != nullptr);
  csession.SetOutputs({output_op});
  csession.Run(s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  TF_Tensor* out = csession.output_tensor(0);
  ASSERT_TRUE(out != nullptr);
  EXPECT_EQ(TF_FLOAT, TF_TensorType(out));
  EXPECT_EQ(2, TF_NumDims(out));
  EXPECT_EQ(4, TF_Dim(out, 0));
  EXPECT_EQ(1, TF_Dim(out, 1));
  float* values = static_cast<float*>(TF_TensorData(out));
  // These values are defined to be (input / 2) + 2.
  EXPECT_EQ(2, values[0]);
  EXPECT_EQ(2.5, values[1]);
  EXPECT_EQ(3, values[2]);
  EXPECT_EQ(3.5, values[3]);

  csession.CloseAndDelete(s);
  EXPECT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  TF_DeleteGraph(graph);
  TF_DeleteStatus(s);
}

TEST(CAPI, SavedModelNullArgsAreValid) {
  const char kSavedModel[] = "cc/saved_model/testdata/half_plus_two/00000123";
  const string saved_model_dir = tensorflow::io::JoinPath(
      tensorflow::testing::TensorFlowSrcRoot(), kSavedModel);
  TF_SessionOptions* opt = TF_NewSessionOptions();
  TF_Status* s = TF_NewStatus();
  const char* tags[] = {tensorflow::kSavedModelTagServe};
  TF_Graph* graph = TF_NewGraph();
  // NULL run_options and meta_graph_def should work.
  TF_Session* session = TF_LoadSessionFromSavedModel(
      opt, nullptr, saved_model_dir.c_str(), tags, 1, graph, nullptr, s);
  EXPECT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  TF_DeleteSessionOptions(opt);
  TF_CloseSession(session, s);
  EXPECT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  TF_DeleteSession(session, s);
  EXPECT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  TF_DeleteGraph(graph);
  TF_DeleteStatus(s);
}

class CApiWhileLoopTest : public ::testing::Test {
 protected:
  CApiWhileLoopTest() : s_(TF_NewStatus()), graph_(TF_NewGraph()) {}

  ~CApiWhileLoopTest() override {
    TF_DeleteGraph(graph_);
    TF_DeleteStatus(s_);
  }

  void Init(int ninputs) {
    DCHECK(inputs_.empty());
    DCHECK_GT(ninputs, 0);

    for (int i = 0; i < ninputs; ++i) {
      TF_Operation* placeholder = Placeholder(
          graph_, s_, ::tensorflow::strings::StrCat("p", i).c_str());
      DCHECK_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
      inputs_.push_back({placeholder, 0});
    }

    original_graph_description_ = GraphDebugString();

    params_.reset(new TF_WhileParams(
        TF_NewWhile(graph_, &inputs_[0], inputs_.size(), s_)));
    ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
    ASSERT_EQ(original_graph_description_, GraphDebugString())
        << "TF_NewWhile() altered graph";

    params_->name = "test_loop";

    // Initialize outputs_ so we can easily detect errors/bugs
    outputs_.resize(ninputs, {nullptr, -1});
  }

  void ExpectOK() {
    TF_FinishWhile(params_.get(), s_, &outputs_[0]);
    EXPECT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  }

  void ExpectError(TF_Code expected_code, const string& expected_msg) {
    TF_FinishWhile(params_.get(), s_, &outputs_[0]);
    EXPECT_EQ(expected_code, TF_GetCode(s_));
    EXPECT_EQ(expected_msg, TF_Message(s_));
    // TODO(skyewm): this assert is currently broken. Fix or remove guarantee.
    // ASSERT_EQ(original_graph_description_, GraphDebugString()) <<
    //     "TF_FinishWhile() altered graph on error";
  }

  void Run(std::initializer_list<int> input_values) {
    DCHECK_EQ(inputs_.size(), input_values.size());
    std::vector<std::pair<TF_Operation*, TF_Tensor*>> inputs(inputs_.size());
    int i = 0;
    for (int v : input_values) {
      inputs[i] = {inputs_[i].oper, Int32Tensor(v)};
      ++i;
    }
    csession_.reset(new CSession(graph_, s_));
    csession_->SetInputs(inputs);
    csession_->SetOutputs(outputs_);
    csession_->Run(s_);
    ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  }

  void ExpectOutputValue(int idx, int expected_value) {
    TF_Tensor* out = csession_->output_tensor(idx);
    ASSERT_TRUE(out != nullptr);
    EXPECT_EQ(TF_INT32, TF_TensorType(out));
    EXPECT_EQ(0, TF_NumDims(out));
    ASSERT_EQ(sizeof(int32), TF_TensorByteSize(out));
    int32* data = static_cast<int32*>(TF_TensorData(out));
    EXPECT_EQ(expected_value, *data);
  }

  // Create a valid conditonal graph. Useful for testing unrelated errors.
  void CreateCondGraph() {
    TF_Operation* one = ScalarConst(1, params_->cond_graph, s_);
    TF_Operation* less_than =
        LessThan(params_->cond_inputs[0], {one, 0}, params_->cond_graph, s_);
    DCHECK_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
    params_->cond_output = {less_than, 0};
  }

  string GraphDebugString() const {
    TF_Buffer* buf = TF_NewBuffer();
    TF_GraphToGraphDef(graph_, buf, s_);
    DCHECK_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
    GraphDef def;
    bool success = def.ParseFromArray(buf->data, buf->length);
    DCHECK(success);
    TF_DeleteBuffer(buf);
    return def.DebugString();
  }

  TF_Status* s_;
  TF_Graph* graph_;
  std::vector<TF_Output> inputs_;   // The inputs to the while loop
  std::vector<TF_Output> outputs_;  // The final outputs of the while loop
  std::unique_ptr<TF_WhileParams> params_;
  std::unique_ptr<CSession> csession_;

 private:
  // Used to verify that errors don't change graph_
  string original_graph_description_;
};

TEST_F(CApiWhileLoopTest, BasicLoop) {
  Init(2);

  // Validate TF_WhileParams returned by TF_NewWhile()
  EXPECT_TRUE(params_->body_graph != nullptr);
  EXPECT_TRUE(params_->cond_graph != nullptr);

  EXPECT_EQ(params_->ninputs, 2);

  ASSERT_TRUE(params_->cond_inputs != nullptr);
  ASSERT_TRUE(params_->cond_inputs[0].oper != nullptr);
  EXPECT_TRUE(params_->cond_inputs[1].oper != nullptr);

  ASSERT_TRUE(params_->body_inputs != nullptr);
  EXPECT_TRUE(params_->body_inputs[0].oper != nullptr);
  EXPECT_TRUE(params_->body_inputs[1].oper != nullptr);

  ASSERT_TRUE(params_->body_outputs != nullptr);

  // Create loop: while (input1 < input2) input1 += input2 + 1
  TF_Operation* less_than =
      LessThan(params_->cond_inputs[0], params_->cond_inputs[1],
               params_->cond_graph, s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  params_->cond_output = {less_than, 0};

  TF_Operation* add1 = Add(params_->body_inputs[0], params_->body_inputs[1],
                           params_->body_graph, s_, "add1");
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  TF_Operation* one = ScalarConst(1, params_->body_graph, s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  TF_Operation* add2 = Add(add1, one, params_->body_graph, s_, "add2");
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  params_->body_outputs[0] = {add2, 0};
  params_->body_outputs[1] = params_->body_inputs[1];

  // Finalize while loop
  ExpectOK();

  // Validate while loop outputs returned by TF_FinishWhile()
  EXPECT_TRUE(outputs_[0].oper != nullptr);
  EXPECT_GE(outputs_[0].index, 0);
  EXPECT_TRUE(outputs_[1].oper != nullptr);
  EXPECT_GE(outputs_[1].index, 0);

  // Run the graph
  Run({-9, 2});
  ExpectOutputValue(0, 3);
  ExpectOutputValue(1, 2);
}

TEST_F(CApiWhileLoopTest, NestedLoop) {
  Init(2);
  // Create nested loop:
  //  while (input1 < 6) {
  //    inner_input1 = input1
  //    while (inner_input1 < 3) {
  //      input2 += 1
  //      inner_input1 += 2
  //    }
  //    input1 += input2
  //  }
  //
  // Expected execution with initial values input1 = input2 = 0:
  //
  // outer inner               inner_
  // step# step# input1 input2 input1
  // ------------------------------------
  //   0     0     0      0      0
  //   0     1     0      1      2
  //   0     2     0      2      4
  //   0     -     2      2      -
  //   1     0     2      2      2
  //   1     1     2      3      4
  //   1     -     5      3      -
  //   2     0     5      3      5
  //   2     -     8      3      -

  // Create outer cond graph
  TF_Operation* six = ScalarConst(6, params_->cond_graph, s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  TF_Operation* less_than =
      LessThan(params_->cond_inputs[0], {six, 0}, params_->cond_graph, s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  params_->cond_output = {less_than, 0};

  // Create outer body graph
  // Init inner graph
  TF_Output inner_inputs[] = {params_->body_inputs[0], params_->body_inputs[1]};
  TF_WhileParams inner_params =
      TF_NewWhile(params_->body_graph, inner_inputs, 2, s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  inner_params.name = "inner_loop";

  // Create inner cond graph
  TF_Operation* three = ScalarConst(3, inner_params.cond_graph, s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  TF_Operation* inner_less_than = LessThan(
      inner_params.cond_inputs[0], {three, 0}, inner_params.cond_graph, s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  inner_params.cond_output = {inner_less_than, 0};

  // Create inner body graph
  TF_Operation* one = ScalarConst(1, inner_params.body_graph, s_, "one");
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  TF_Operation* two = ScalarConst(2, inner_params.body_graph, s_, "two");
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);

  TF_Operation* input2_add =
      Add(inner_params.body_inputs[1].oper, one, inner_params.body_graph, s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  inner_params.body_outputs[1] = {input2_add, 0};

  TF_Operation* inner_input1_add = Add(inner_params.body_inputs[0].oper, two,
                                       inner_params.body_graph, s_, "add2");
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  inner_params.body_outputs[0] = {inner_input1_add, 0};

  // Finalize inner graph
  TF_Output inner_outputs[2] = {{nullptr, -1}};
  TF_FinishWhile(&inner_params, s_, inner_outputs);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);

  TF_Operation* input1_add =
      Add(params_->body_inputs[0], inner_outputs[1], params_->body_graph, s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  params_->body_outputs[0] = {input1_add, 0};

  params_->body_outputs[1] = inner_outputs[1];

  // Finalize outer graph
  ExpectOK();

  // Check for a few expected nodes
  const char* node_name = "test_loop/cond/scalar";
  EXPECT_TRUE(TF_GraphOperationByName(graph_, node_name) != nullptr);
  node_name = "test_loop/body/add";
  EXPECT_TRUE(TF_GraphOperationByName(graph_, node_name) != nullptr);
  node_name = "test_loop/body/inner_loop/body/one";
  EXPECT_TRUE(TF_GraphOperationByName(graph_, node_name) != nullptr);
  node_name = "test_loop/body/inner_loop/cond/less_than";
  EXPECT_TRUE(TF_GraphOperationByName(graph_, node_name) != nullptr);

  // Run the graph
  Run({0, 0});
  ExpectOutputValue(0, 8);
  ExpectOutputValue(1, 3);
}

TEST_F(CApiWhileLoopTest, BadCondOutput) {
  Init(1);
  params_->body_outputs[0] = params_->body_inputs[0];
  ExpectError(TF_INVALID_ARGUMENT,
              "TF_WhileParams `cond_output` field isn't set");
}

TEST_F(CApiWhileLoopTest, BadBodyOutput) {
  Init(1);
  CreateCondGraph();
  ExpectError(TF_INVALID_ARGUMENT,
              "TF_WhileParams `body_outputs[0]` field isn't set");
}

TEST_F(CApiWhileLoopTest, NullName) {
  Init(1);
  CreateCondGraph();
  params_->body_outputs[0] = params_->body_inputs[0];
  params_->name = nullptr;
  ExpectError(TF_INVALID_ARGUMENT, "TF_WhileParams `name` field is null");
}

TEST_F(CApiWhileLoopTest, WrongGraph) {
  Init(1);
  CreateCondGraph();
  // Set body output to output from outer graph
  params_->body_outputs[0] = inputs_[0];
  // TODO(skyewm): improve error message
  ExpectError(TF_INVALID_ARGUMENT,
              "Requested return node 'p0' not found in graph def");
}

TEST_F(CApiWhileLoopTest, BadTypes) {
  Init(1);
  CreateCondGraph();
  // Op that has a float input + output
  TF_OperationDescription* desc = TF_NewOperation(
      params_->body_graph, "FakeQuantWithMinMaxArgs", "float_op");
  TF_AddInput(desc, params_->body_inputs[0]);
  TF_FinishOperation(desc, s_);
  ASSERT_EQ(TF_INVALID_ARGUMENT, TF_GetCode(s_));
  string msg(TF_Message(s_));
  EXPECT_NE(msg.find("Input 'inputs' passed int32 expected float while "
                     "building NodeDef 'float_op'"),
            msg.npos);
  TF_AbortWhile(params_.get());
}

REGISTER_OP("TestOpWithNoGradient")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {float, double}")
    .Doc(R"doc(
Test op with no grad registered.

x: input
y: output
)doc");

class CApiGradientsTest : public ::testing::Test {
 protected:
  CApiGradientsTest()
      : s_(TF_NewStatus()),
        graph_(TF_NewGraph()),
        expected_graph_(TF_NewGraph()) {}

  ~CApiGradientsTest() override {
    TF_DeleteGraph(graph_);
    TF_DeleteGraph(expected_graph_);
    TF_DeleteStatus(s_);
  }

  void TestGradientsSuccess(bool grad_inputs_provided) {
    TF_Output inputs[2];
    TF_Output outputs[1];
    TF_Output grad_outputs[2];
    TF_Output expected_grad_outputs[2];

    BuildSuccessGraph(inputs, outputs);
    BuildExpectedGraph(grad_inputs_provided, expected_grad_outputs);

    AddGradients(grad_inputs_provided, inputs, 2, outputs, 1, grad_outputs);

    EXPECT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);

    // Compare that the graphs match.
    GraphDef expected_gdef;
    GraphDef gdef;
    EXPECT_TRUE(GetGraphDef(expected_graph_, &expected_gdef));
    EXPECT_TRUE(GetGraphDef(graph_, &gdef));
    TF_EXPECT_GRAPH_EQ(expected_gdef, gdef);

    // Compare that the output of the gradients of both graphs match.
    RunGraphsAndCompareOutputs(grad_outputs, expected_grad_outputs);
  }

  void TestGradientsError(bool grad_inputs_provided) {
    TF_Output inputs[1];
    TF_Output outputs[1];
    TF_Output grad_outputs[1];

    BuildErrorGraph(inputs, outputs);

    AddGradients(grad_inputs_provided, inputs, 1, outputs, 1, grad_outputs);

    string expected_msg =
        "No gradient defined for op: TestOpWithNoGradient. Please see "
        "https://www.tensorflow.org/code/"
        "tensorflow/cc/gradients/README.md"
        " for instructions on how to add C++ gradients.";
    EXPECT_EQ(expected_msg, TF_Message(s_));
  }

  // Run the graph and ensure that the gradient values are as expected.
  void RunGraphsAndCompareOutputs(TF_Output* grad_outputs,
                                  TF_Output* expected_grad_outputs) {
    std::unique_ptr<CSession> csession(new CSession(graph_, s_));
    std::unique_ptr<CSession> expected_csession(
        new CSession(expected_graph_, s_));

    std::vector<TF_Output> grad_outputs_vec;
    grad_outputs_vec.assign(grad_outputs, grad_outputs + 2);
    csession->SetOutputs(grad_outputs_vec);
    csession->Run(s_);
    ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
    TF_Tensor* out0 = csession->output_tensor(0);
    TF_Tensor* out1 = csession->output_tensor(1);

    std::vector<TF_Output> expected_grad_outputs_vec;
    expected_grad_outputs_vec.assign(expected_grad_outputs,
                                     expected_grad_outputs + 2);
    expected_csession->SetOutputs(expected_grad_outputs_vec);
    expected_csession->Run(s_);
    ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
    TF_Tensor* expected_out0 = expected_csession->output_tensor(0);
    TF_Tensor* expected_out1 = expected_csession->output_tensor(1);

    CompareTensors(out0, expected_out0);
    CompareTensors(out1, expected_out1);
  }

  void CompareTensors(TF_Tensor* a, TF_Tensor* b) {
    float* a_data = static_cast<float*>(TF_TensorData(a));
    float* b_data = static_cast<float*>(TF_TensorData(b));
    EXPECT_EQ(*a_data, *b_data);
  }

  void AddGradients(bool grad_inputs_provided, TF_Output* inputs, int ninputs,
                    TF_Output* outputs, int noutputs, TF_Output* grad_outputs) {
    if (grad_inputs_provided) {
      TF_Output grad_inputs[1];
      const float grad_inputs_val[] = {1.0, 1.0, 1.0, 1.0};
      TF_Operation* grad_inputs_op =
          FloatConst2x2(graph_, s_, grad_inputs_val, "GradInputs");
      grad_inputs[0] = TF_Output{grad_inputs_op, 0};
      TF_AddGradients(graph_, outputs, noutputs, inputs, ninputs, grad_inputs,
                      s_, grad_outputs);
    } else {
      TF_AddGradients(graph_, outputs, noutputs, inputs, ninputs, nullptr, s_,
                      grad_outputs);
    }
  }

  void BuildErrorGraph(TF_Output* inputs, TF_Output* outputs) {
    const float const0_val[] = {1.0, 2.0, 3.0, 4.0};
    TF_Operation* const0 = FloatConst2x2(graph_, s_, const0_val, "Const_0");
    TF_Operation* nograd = NoGradientOp(graph_, s_, const0, "NoGrad");
    inputs[0] = TF_Output{const0, 0};
    outputs[0] = TF_Output{nograd, 0};
    EXPECT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  }

  void BuildSuccessGraph(TF_Output* inputs, TF_Output* outputs) {
    // Construct the following graph:
    //            |
    //           z|
    //            |
    //          MatMul
    //         /       \
    //        ^         ^
    //        |         |
    //       x|        y|
    //        |         |
    //        |         |
    //      Const_0    Const_1
    //
    const float const0_val[] = {1.0, 2.0, 3.0, 4.0};
    const float const1_val[] = {1.0, 0.0, 0.0, 1.0};
    TF_Operation* const0 = FloatConst2x2(graph_, s_, const0_val, "Const_0");
    TF_Operation* const1 = FloatConst2x2(graph_, s_, const1_val, "Const_1");
    TF_Operation* matmul = MatMul(graph_, s_, const0, const1, "MatMul");
    inputs[0] = TF_Output{const0, 0};
    inputs[1] = TF_Output{const1, 0};
    outputs[0] = TF_Output{matmul, 0};
    EXPECT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  }

  void BuildExpectedGraph(bool grad_inputs_provided,
                          TF_Output* expected_grad_outputs) {
    // The expected graph looks like this if grad_inputs_provided.
    // If grad_inputs_provided is false, Const_0 will be a OnesLike op.
    //      ^             ^
    //    dy|           dx|        // MatMul Gradient Graph
    //      |             |
    //   MatMul_2      MatMul_1
    //   ^   ^          ^    ^
    //   |   |----------|    |
    //   |        ^          |
    //   |      dz|          |
    //   |        |          |
    //   |     Const_3       |
    //   |                   |
    //   |        ^          |
    //   |       z|          |     // MatMul Forward Graph
    //   |        |          |
    //   |      MatMul       |
    //   |     /       \     |
    //   |    ^         ^    |
    //   |    |         |    |
    //   |---x|        y|----|
    //        |         |
    //        |         |
    //      Const_0   Const_1
    //
    const float const0_val[] = {1.0, 2.0, 3.0, 4.0};
    const float const1_val[] = {1.0, 0.0, 0.0, 1.0};
    TF_Operation* const0 =
        FloatConst2x2(expected_graph_, s_, const0_val, "Const_0");
    TF_Operation* const1 =
        FloatConst2x2(expected_graph_, s_, const1_val, "Const_1");
    TF_Operation* matmul =
        MatMul(expected_graph_, s_, const0, const1, "MatMul");

    TF_Operation* const3;
    if (grad_inputs_provided) {
      const float const3_val[] = {1.0, 1.0, 1.0, 1.0};
      const3 = FloatConst2x2(expected_graph_, s_, const3_val, "GradInputs");
    } else {
      const3 = OnesLike(expected_graph_, s_, matmul, "OnesLike");
    }

    TF_Operation* matmul1 =
        MatMul(expected_graph_, s_, const3, const1, "MatMul_1", false, true);
    TF_Operation* matmul2 =
        MatMul(expected_graph_, s_, const0, const3, "MatMul_2", true, false);
    expected_grad_outputs[0] = {matmul1, 0};
    expected_grad_outputs[1] = {matmul2, 0};
  }

  TF_Tensor* FloatTensor2x2(const float* values) {
    const int64_t dims[2] = {2, 2};
    TF_Tensor* t = TF_AllocateTensor(TF_FLOAT, dims, 2, sizeof(float) * 4);
    memcpy(TF_TensorData(t), values, sizeof(float) * 4);
    return t;
  }

  TF_Operation* FloatConst2x2(TF_Graph* graph, TF_Status* s,
                              const float* values, const char* name) {
    unique_tensor_ptr tensor(FloatTensor2x2(values), TF_DeleteTensor);
    TF_OperationDescription* desc = TF_NewOperation(graph, "Const", name);
    TF_SetAttrTensor(desc, "value", tensor.get(), s);
    if (TF_GetCode(s) != TF_OK) return nullptr;
    TF_SetAttrType(desc, "dtype", TF_FLOAT);
    TF_Operation* op = TF_FinishOperation(desc, s);
    EXPECT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
    return op;
  }

  TF_Operation* MatMul(TF_Graph* graph, TF_Status* s, TF_Operation* l,
                       TF_Operation* r, const char* name,
                       bool transpose_a = false, bool transpose_b = false) {
    TF_OperationDescription* desc = TF_NewOperation(graph, "MatMul", name);
    if (transpose_a) {
      TF_SetAttrBool(desc, "transpose_a", 1);
    }
    if (transpose_b) {
      TF_SetAttrBool(desc, "transpose_b", 1);
    }
    TF_AddInput(desc, {l, 0});
    TF_AddInput(desc, {r, 0});
    TF_Operation* op = TF_FinishOperation(desc, s);
    EXPECT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
    return op;
  }

  TF_Operation* OnesLike(TF_Graph* graph, TF_Status* s, TF_Operation* in,
                         const char* name) {
    TF_OperationDescription* desc = TF_NewOperation(graph, "OnesLike", name);
    TF_AddInput(desc, {in, 0});
    TF_Operation* op = TF_FinishOperation(desc, s);
    EXPECT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
    return op;
  }

  TF_Operation* NoGradientOp(TF_Graph* graph, TF_Status* s, TF_Operation* in,
                             const char* name) {
    TF_OperationDescription* desc =
        TF_NewOperation(graph, "TestOpWithNoGradient", name);
    TF_AddInput(desc, {in, 0});
    TF_Operation* op = TF_FinishOperation(desc, s);
    EXPECT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
    return op;
  }

  TF_Status* s_;
  TF_Graph* graph_;
  TF_Graph* expected_graph_;
};

TEST_F(CApiGradientsTest, Gradients_GradInputs) { TestGradientsSuccess(true); }

TEST_F(CApiGradientsTest, Gradients_NoGradInputs) {
  TestGradientsSuccess(false);
}

TEST_F(CApiGradientsTest, OpWithNoGradientRegistered_GradInputs) {
  TestGradientsError(true);
}

TEST_F(CApiGradientsTest, OpWithNoGradientRegistered_NoGradInputs) {
  TestGradientsError(false);
}

// Create a tensor with values of type TF_INT8 provided by `values`.
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

void StringVectorToArrays(const std::vector<string>& v,
                          std::unique_ptr<const void* []>* ptrs,
                          std::unique_ptr<size_t[]>* lens) {
  ptrs->reset(new const void*[v.size()]);
  lens->reset(new size_t[v.size()]);
  for (size_t i = 0; i < v.size(); ++i) {
    (*ptrs)[i] = v[i].data();
    (*lens)[i] = v[i].size();
  }
}

// REGISTER_OP for CApiTestAttributesTest test cases.
// Registers two ops, each with a single attribute called 'v'.
// The attribute in one op will have a type 'type', the other
// will have list(type).
#define ATTR_TEST_REGISTER_OP(type)                            \
  REGISTER_OP("CApiAttributesTestOp" #type).Attr("v: " #type); \
  REGISTER_OP("CApiAttributesTestOpList" #type).Attr("v: list(" #type ")")
ATTR_TEST_REGISTER_OP(string);
ATTR_TEST_REGISTER_OP(int);
ATTR_TEST_REGISTER_OP(float);
ATTR_TEST_REGISTER_OP(bool);
ATTR_TEST_REGISTER_OP(type);
ATTR_TEST_REGISTER_OP(shape);
ATTR_TEST_REGISTER_OP(tensor);
#undef ATTR_TEST_REGISTER_OP

class CApiAttributesTest : public ::testing::Test {
 protected:
  CApiAttributesTest()
      : s_(TF_NewStatus()), graph_(TF_NewGraph()), counter_(0) {}

  ~CApiAttributesTest() override {
    TF_DeleteGraph(graph_);
    TF_DeleteStatus(s_);
  }

  TF_OperationDescription* init(string type) {
    // Construct op_name to match the name used by REGISTER_OP in the
    // ATTR_TEST_REGISTER calls above.
    string op_name = "CApiAttributesTestOp";
    if (type.find("list(") == 0) {
      op_name += "List";
      type = type.replace(0, 5, "");
      type = type.replace(type.size() - 1, 1, "");
    }
    op_name += type;
    return TF_NewOperation(
        graph_, op_name.c_str(),
        ::tensorflow::strings::StrCat("name", counter_++).c_str());
  }

  TF_Status* s_;

 private:
  TF_Graph* graph_;
  int counter_;
};

// Helper macros for the TF_OperationGetAttr* tests.
// TODO(ashankar): Use gmock matchers instead?
// (https://github.com/google/googletest/blob/master/googlemock/docs/CookBook.md#writing-new-parameterized-matchers-quickly)
// That will require setting up the tensorflow build with gmock.
#define EXPECT_TF_META(attr_name, expected_list_size, expected_type, \
                       expected_total_size)                          \
  do {                                                               \
    auto m = TF_OperationGetAttrMetadata(oper, attr_name, s_);       \
    EXPECT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);              \
    const unsigned char e = expected_list_size >= 0 ? 1 : 0;         \
    EXPECT_EQ(e, m.is_list);                                         \
    EXPECT_EQ(expected_list_size, m.list_size);                      \
    EXPECT_EQ(expected_type, m.type);                                \
    EXPECT_EQ(expected_total_size, m.total_size);                    \
  } while (0)

TEST_F(CApiAttributesTest, String) {
  auto desc = init("string");
  TF_SetAttrString(desc, "v", "bunny", 5);

  auto oper = TF_FinishOperation(desc, s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  EXPECT_TF_META("v", -1, TF_ATTR_STRING, 5);
  std::unique_ptr<char[]> value(new char[5]);

  TF_OperationGetAttrString(oper, "v", value.get(), 5, s_);
  EXPECT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  EXPECT_EQ("bunny", string(static_cast<const char*>(value.get()), 5));
}

TEST_F(CApiAttributesTest, StringList) {
  std::vector<string> list = {"bugs", "bunny", "duck"};
  std::unique_ptr<const void* []> list_ptrs;
  std::unique_ptr<size_t[]> list_lens;
  StringVectorToArrays(list, &list_ptrs, &list_lens);
  int list_total_size = 0;
  for (const auto& s : list) {
    list_total_size += s.size();
  }

  auto desc = init("list(string)");
  TF_SetAttrStringList(desc, "v", list_ptrs.get(), list_lens.get(),
                       list.size());

  auto oper = TF_FinishOperation(desc, s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);

  EXPECT_TF_META("v", list.size(), TF_ATTR_STRING, list_total_size);
  std::unique_ptr<void* []> values(new void*[list.size()]);
  std::unique_ptr<size_t[]> lens(new size_t[list.size()]);
  std::unique_ptr<char[]> storage(new char[list_total_size]);
  TF_OperationGetAttrStringList(oper, "v", values.get(), lens.get(),
                                list.size(), storage.get(), list_total_size,
                                s_);
  EXPECT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  for (size_t i = 0; i < list.size(); ++i) {
    EXPECT_EQ(list[i].size(), lens[i]) << i;
    EXPECT_EQ(list[i], string(static_cast<const char*>(values[i]), lens[i]))
        << i;
  }
}

TEST_F(CApiAttributesTest, Int) {
  auto desc = init("int");
  TF_SetAttrInt(desc, "v", 31415);

  auto oper = TF_FinishOperation(desc, s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  EXPECT_TF_META("v", -1, TF_ATTR_INT, -1);

  int64_t value;
  TF_OperationGetAttrInt(oper, "v", &value, s_);
  EXPECT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  EXPECT_EQ(31415, value);
}

TEST_F(CApiAttributesTest, IntList) {
  const int64_t list[] = {1, 2, 3, 4};
  const size_t list_size = TF_ARRAYSIZE(list);

  auto desc = init("list(int)");
  TF_SetAttrIntList(desc, "v", list, list_size);

  auto oper = TF_FinishOperation(desc, s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);

  int64_t values[list_size];
  EXPECT_TF_META("v", list_size, TF_ATTR_INT, -1);
  TF_OperationGetAttrIntList(oper, "v", values, list_size, s_);
  EXPECT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  EXPECT_TRUE(std::equal(std::begin(list), std::end(list), std::begin(values)));
}

TEST_F(CApiAttributesTest, Float) {
  auto desc = init("float");
  TF_SetAttrFloat(desc, "v", 2.718);

  auto oper = TF_FinishOperation(desc, s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  EXPECT_TF_META("v", -1, TF_ATTR_FLOAT, -1);

  float value;
  TF_OperationGetAttrFloat(oper, "v", &value, s_);
  EXPECT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  EXPECT_FLOAT_EQ(2.718, value);
}

TEST_F(CApiAttributesTest, FloatList) {
  const float list[] = {1.414, 2.718, 3.1415};
  const size_t list_size = TF_ARRAYSIZE(list);

  auto desc = init("list(float)");
  TF_SetAttrFloatList(desc, "v", list, list_size);

  auto oper = TF_FinishOperation(desc, s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);

  float values[list_size];
  EXPECT_TF_META("v", list_size, TF_ATTR_FLOAT, -1);
  TF_OperationGetAttrFloatList(oper, "v", values, list_size, s_);
  EXPECT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  EXPECT_TRUE(std::equal(std::begin(list), std::end(list), std::begin(values)));
}

TEST_F(CApiAttributesTest, Bool) {
  auto desc = init("bool");
  TF_SetAttrBool(desc, "v", 1);

  auto oper = TF_FinishOperation(desc, s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  EXPECT_TF_META("v", -1, TF_ATTR_BOOL, -1);

  unsigned char value;
  TF_OperationGetAttrBool(oper, "v", &value, s_);
  EXPECT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  EXPECT_EQ(1, value);
}

TEST_F(CApiAttributesTest, BoolList) {
  const unsigned char list[] = {0, 1, 1, 0, 0, 1, 1};
  const size_t list_size = TF_ARRAYSIZE(list);

  auto desc = init("list(bool)");
  TF_SetAttrBoolList(desc, "v", list, list_size);

  auto oper = TF_FinishOperation(desc, s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);

  unsigned char values[list_size];
  EXPECT_TF_META("v", list_size, TF_ATTR_BOOL, -1);
  TF_OperationGetAttrBoolList(oper, "v", values, list_size, s_);
  EXPECT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  EXPECT_TRUE(std::equal(std::begin(list), std::end(list), std::begin(values)));
}

TEST_F(CApiAttributesTest, Type) {
  auto desc = init("type");
  TF_SetAttrType(desc, "v", TF_COMPLEX128);

  auto oper = TF_FinishOperation(desc, s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  EXPECT_TF_META("v", -1, TF_ATTR_TYPE, -1);

  TF_DataType value;
  TF_OperationGetAttrType(oper, "v", &value, s_);
  EXPECT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  EXPECT_EQ(TF_COMPLEX128, value);
}

TEST_F(CApiAttributesTest, TypeList) {
  const TF_DataType list[] = {TF_FLOAT, TF_DOUBLE, TF_HALF, TF_COMPLEX128};
  const size_t list_size = TF_ARRAYSIZE(list);

  auto desc = init("list(type)");
  TF_SetAttrTypeList(desc, "v", list, list_size);

  auto oper = TF_FinishOperation(desc, s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);

  TF_DataType values[list_size];
  EXPECT_TF_META("v", list_size, TF_ATTR_TYPE, -1);
  TF_OperationGetAttrTypeList(oper, "v", values, list_size, s_);
  EXPECT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  EXPECT_TRUE(std::equal(std::begin(list), std::end(list), std::begin(values)));
}

TEST_F(CApiAttributesTest, Shape) {
  // Unknown shape
  auto desc = init("shape");
  TF_SetAttrShape(desc, "v", nullptr, -1);
  auto oper = TF_FinishOperation(desc, s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  EXPECT_TF_META("v", -1, TF_ATTR_SHAPE, -1);
  TF_OperationGetAttrShape(oper, "v", nullptr, 10, s_);
  EXPECT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);

  // Partially specified shape
  const int64_t partial_shape[] = {17, -1};
  const size_t sz = TF_ARRAYSIZE(partial_shape);
  desc = init("shape");
  TF_SetAttrShape(desc, "v", partial_shape, sz);
  oper = TF_FinishOperation(desc, s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  EXPECT_TF_META("v", -1, TF_ATTR_SHAPE, sz);
  int64_t values[sz];
  TF_OperationGetAttrShape(oper, "v", values, sz, s_);
  EXPECT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  EXPECT_TRUE(
      std::equal(std::begin(partial_shape), std::end(partial_shape), values));
}

TEST_F(CApiAttributesTest, ShapeList) {
  const int64_t shape_1[] = {1, 3};
  const int64_t shape_2[] = {2, 4, 6};
  const int64_t* list[] = {&shape_1[0], &shape_2[0]};
  const size_t list_size = TF_ARRAYSIZE(list);
  const int ndims[] = {TF_ARRAYSIZE(shape_1), TF_ARRAYSIZE(shape_2)};
  const int total_ndims = 5;  // ndims[0] + ndims[1]

  auto desc = init("list(shape)");
  TF_SetAttrShapeList(desc, "v", list, ndims, list_size);
  auto oper = TF_FinishOperation(desc, s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);

  EXPECT_TF_META("v", list_size, TF_ATTR_SHAPE, total_ndims);
  int64_t* values[list_size];
  int values_ndims[list_size];
  int64_t storage[total_ndims];
  TF_OperationGetAttrShapeList(oper, "v", values, values_ndims, list_size,
                               storage, total_ndims, s_);
  EXPECT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  for (size_t i = 0; i < list_size; ++i) {
    EXPECT_EQ(ndims[i], values_ndims[i]) << i;
    for (int j = 0; j < values_ndims[i]; ++j) {
      EXPECT_EQ(list[i][j], values[i][j]) << "(" << i << ", " << j << ")";
    }
  }
}

TEST_F(CApiAttributesTest, TensorShapeProto) {
  const tensorflow::int64 pts[] = {2, 4, -1, 8};
  tensorflow::TensorShapeProto proto;
  tensorflow::PartialTensorShape(pts).AsProto(&proto);
  string bytes;
  proto.SerializeToString(&bytes);

  auto desc = init("shape");
  TF_SetAttrTensorShapeProto(desc, "v", bytes.data(), bytes.length(), s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  auto oper = TF_FinishOperation(desc, s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);

  EXPECT_TF_META("v", -1, TF_ATTR_SHAPE, 4);
  TF_Buffer* value = TF_NewBuffer();
  TF_OperationGetAttrTensorShapeProto(oper, "v", value, s_);
  EXPECT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  EXPECT_EQ(bytes.length(), value->length);
  EXPECT_EQ(0, memcmp(bytes.data(), value->data, value->length));
  TF_DeleteBuffer(value);
}

TEST_F(CApiAttributesTest, TensorShapeProtoList) {
  string bytes1, bytes2;
  tensorflow::TensorShapeProto proto;

  const tensorflow::int64 pts1[] = {2, 4, -1, 8};
  tensorflow::PartialTensorShape(pts1).AsProto(&proto);
  proto.SerializeToString(&bytes1);

  const tensorflow::int64 pts2[] = {1, 3, 5, 7};
  tensorflow::PartialTensorShape(pts2).AsProto(&proto);
  proto.SerializeToString(&bytes2);

  std::unique_ptr<const void* []> list_ptrs;
  std::unique_ptr<size_t[]> list_lens;
  const std::vector<string> list = {bytes1, bytes2};
  StringVectorToArrays(list, &list_ptrs, &list_lens);

  auto desc = init("list(shape)");
  TF_SetAttrTensorShapeProtoList(desc, "v", list_ptrs.get(), list_lens.get(),
                                 list.size(), s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  auto oper = TF_FinishOperation(desc, s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);

  EXPECT_TF_META("v", 2, TF_ATTR_SHAPE, 8);
  TF_Buffer* values[2];
  TF_OperationGetAttrTensorShapeProtoList(oper, "v", values, 2, s_);
  EXPECT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  for (int i = 0; i < 2; ++i) {
    int le = list_lens[i];
    int la = values[i]->length;
    const void* e = list_ptrs[i];
    const void* a = values[i]->data;
    EXPECT_EQ(le, la) << i;
    EXPECT_EQ(0, memcmp(e, a, std::min(le, la))) << i;
    TF_DeleteBuffer(values[i]);
  }
}

TEST_F(CApiAttributesTest, Tensor) {
  const char tensor[] = {5, 7};
  const int64_t dims[] = {1, 2};
  const size_t ndims = TF_ARRAYSIZE(dims);

  auto desc = init("tensor");
  unique_tensor_ptr v(Int8Tensor(dims, ndims, tensor), TF_DeleteTensor);
  TF_SetAttrTensor(desc, "v", v.get(), s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);

  auto oper = TF_FinishOperation(desc, s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);

  EXPECT_TF_META("v", -1, TF_ATTR_TENSOR, -1);
  TF_Tensor* value;
  TF_OperationGetAttrTensor(oper, "v", &value, s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  ASSERT_NE(nullptr, value);
  EXPECT_EQ(TF_INT8, TF_TensorType(value));
  EXPECT_EQ(ndims, TF_NumDims(value));
  for (int i = 0; i < TF_NumDims(value); ++i) {
    EXPECT_EQ(dims[i], TF_Dim(value, i)) << i;
  }
  EXPECT_EQ(sizeof(char) * TF_ARRAYSIZE(tensor), TF_TensorByteSize(value));
  EXPECT_EQ(0, memcmp(tensor, TF_TensorData(value), TF_TensorByteSize(value)));
  TF_DeleteTensor(value);
}

TEST_F(CApiAttributesTest, TensorList) {
  const char tensor1[] = {5, 7};
  const int64_t dims1[] = {1, 2};
  const size_t ndims1 = TF_ARRAYSIZE(dims1);

  const char tensor2[] = {2, 4, 6, 8};
  const int64_t dims2[] = {2, 2};
  const size_t ndims2 = TF_ARRAYSIZE(dims2);

  auto desc = init("list(tensor)");
  TF_Tensor* tmp[] = {
      Int8Tensor(dims1, ndims1, tensor1), Int8Tensor(dims2, ndims2, tensor2),
  };
  TF_SetAttrTensorList(desc, "v", tmp, TF_ARRAYSIZE(tmp), s_);
  for (int i = 0; i < TF_ARRAYSIZE(tmp); ++i) {
    TF_DeleteTensor(tmp[i]);
  }
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  auto oper = TF_FinishOperation(desc, s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);

  EXPECT_TF_META("v", 2, TF_ATTR_TENSOR, -1);
  TF_Tensor* values[2];
  TF_OperationGetAttrTensorList(oper, "v", &values[0], TF_ARRAYSIZE(values),
                                s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);

  const char* tensor_data[] = {&tensor1[0], &tensor2[0]};
  const size_t tensor_size[] = {TF_ARRAYSIZE(tensor1), TF_ARRAYSIZE(tensor2)};
  const int64_t* tensor_dims[] = {&dims1[0], &dims2[0]};
  const size_t tensor_ndims[] = {ndims1, ndims2};
  for (int i = 0; i < 2; ++i) {
    TF_Tensor* v = values[i];
    ASSERT_NE(nullptr, v) << i;
    EXPECT_EQ(TF_INT8, TF_TensorType(v)) << i;
    EXPECT_EQ(tensor_ndims[i], TF_NumDims(v)) << i;
    for (int j = 0; j < TF_NumDims(v); ++j) {
      EXPECT_EQ(tensor_dims[i][j], TF_Dim(v, j))
          << "Tensor #" << i << ", dimension #" << j;
    }
    EXPECT_EQ(sizeof(char) * tensor_size[i], TF_TensorByteSize(v)) << i;
    EXPECT_EQ(0,
              memcmp(tensor_data[i], TF_TensorData(v), TF_TensorByteSize(v)));
    TF_DeleteTensor(v);
  }
}

TEST_F(CApiAttributesTest, EmptyList) {
  auto desc = init("list(int)");
  TF_SetAttrIntList(desc, "v", nullptr, 0);
  auto oper = TF_FinishOperation(desc, s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  EXPECT_TF_META("v", 0, TF_ATTR_INT, -1);
}

TEST_F(CApiAttributesTest, Errors) {
  auto desc = init("int");
  TF_SetAttrInt(desc, "v", 3);
  auto oper = TF_FinishOperation(desc, s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  TF_OperationGetAttrString(oper, "v", nullptr, 0, s_);
  EXPECT_EQ(TF_INVALID_ARGUMENT, TF_GetCode(s_)) << TF_Message(s_);
}
#undef EXPECT_TF_META

// TODO(josh11b): Test:
// * TF_SetDevice(desc, "/job:worker");
// * control inputs / outputs
// * targets
// * TF_DeleteGraph() before TF_DeleteSession()

}  // namespace
