/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/c/c_api_experimental.h"
#include "tensorflow/c/c_test_util.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/tensorflow_server.pb.h"

namespace tensorflow {
namespace {

void TestFakeIteratorStack() {
  TF_Status* s = TF_NewStatus();
  TF_Graph* graph = TF_NewGraph();

  TF_Operation* get_next = TF_MakeFakeIteratorGetNextWithDatasets(graph, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  CSession csession(graph, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  // Run the graph.
  const float base_value = 42.0;
  for (int i = 0; i < 3; ++i) {
    csession.SetOutputs({get_next});
    csession.Run(s);
    ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
    TF_Tensor* out = csession.output_tensor(0);
    ASSERT_TRUE(out != nullptr);
    ASSERT_EQ(TF_FLOAT, TF_TensorType(out));
    ASSERT_EQ(0, TF_NumDims(out));  // scalar
    ASSERT_EQ(sizeof(float), TF_TensorByteSize(out));
    float* output_contents = static_cast<float*>(TF_TensorData(out));
    ASSERT_EQ(base_value + i, *output_contents);
  }

  // This should error out since we've exhausted the iterator.
  csession.Run(s);
  ASSERT_EQ(TF_OUT_OF_RANGE, TF_GetCode(s)) << TF_Message(s);

  // Clean up
  csession.CloseAndDelete(s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  TF_DeleteGraph(graph);
  TF_DeleteStatus(s);
}

TEST(CAPI_EXPERIMENTAL, FakeIteratorGetNext) { TestFakeIteratorStack(); }

TEST(CAPI_EXPERIMENTAL, ImagenetIteratorGetNext) {
  TF_Status* s = TF_NewStatus();
  TF_Graph* graph = TF_NewGraph();

  const string file_path = tensorflow::io::JoinPath(
      tensorflow::testing::TensorFlowSrcRoot(), "c/testdata/tf_record");
  VLOG(1) << "data file path is " << file_path;
  const int batch_size = 64;
  TF_Operation* get_next = TF_MakeFileBasedIteratorGetNextWithDatasets(
      graph, file_path.c_str(), batch_size, /*is_mnist*/ false, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  CSession csession(graph, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  // Run the graph.
  // The two output tensors should look like:
  // Tensor("IteratorGetNext:0", shape=(batch_size, 224, 224, 3), dtype=float32)
  // Tensor("IteratorGetNext:1", shape=(batch_size, ), dtype=int32)
  for (int i = 0; i < 3; ++i) {
    LOG(INFO) << "Running iter " << i;
    csession.SetOutputs({{get_next, 0}, {get_next, 1}});
    csession.Run(s);
    ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

    {
      TF_Tensor* image = csession.output_tensor(0);
      ASSERT_TRUE(image != nullptr);
      ASSERT_EQ(TF_FLOAT, TF_TensorType(image));
      // Confirm shape is 224 X 224 X 3
      ASSERT_EQ(4, TF_NumDims(image));
      ASSERT_EQ(batch_size, TF_Dim(image, 0));
      ASSERT_EQ(224, TF_Dim(image, 1));
      ASSERT_EQ(224, TF_Dim(image, 2));
      ASSERT_EQ(3, TF_Dim(image, 3));
      ASSERT_EQ(sizeof(float) * batch_size * 224 * 224 * 3,
                TF_TensorByteSize(image));
    }

    {
      TF_Tensor* label = csession.output_tensor(1);
      ASSERT_TRUE(label != nullptr);
      ASSERT_EQ(TF_INT32, TF_TensorType(label));
      ASSERT_EQ(1, TF_NumDims(label));
      ASSERT_EQ(batch_size, TF_Dim(label, 0));
      ASSERT_EQ(sizeof(int32) * batch_size, TF_TensorByteSize(label));
    }
  }

  // Clean up
  csession.CloseAndDelete(s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  TF_DeleteGraph(graph);
  TF_DeleteStatus(s);
}

TEST(CAPI_EXPERIMENTAL, GetServerDefTest) {
  const string expected_text_proto(R"(cluster {
  job {
    name: "worker"
    tasks {
      key: 0
      value: "tpuserver:0"
    }
    tasks {
      key: 1
      value: "localhost:1"
    }
  }
}
job_name: "worker"
task_index: 1
protocol: "grpc"
)");

  TF_Status* status = TF_NewStatus();
  TF_Buffer* result = TFE_GetServerDef(expected_text_proto.c_str(), status);
  EXPECT_EQ(TF_GetCode(status), TF_OK);

  ServerDef actual;
  ASSERT_TRUE(actual.ParseFromArray(result->data, result->length));
  string actual_text_proto;
  tensorflow::protobuf::TextFormat::PrintToString(actual, &actual_text_proto);
  EXPECT_EQ(expected_text_proto, actual_text_proto);

  const string malformed_text_proto(R"(cluster {
  job {
    name: "worker")");
  TF_Buffer* null_result =
      TFE_GetServerDef(malformed_text_proto.c_str(), status);
  EXPECT_NE(TF_GetCode(status), TF_OK);
  EXPECT_TRUE(tensorflow::str_util::StrContains(
      TF_Message(status), "Invalid text proto for ServerDef"));
  EXPECT_EQ(null_result, nullptr);

  // Cleanup
  TF_DeleteBuffer(result);
  TF_DeleteStatus(status);
}

TEST(CAPI_EXPERIMENTAL, IsStateful) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  int assign = TF_OpIsStateful("AssignAddVariableOp", status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  EXPECT_EQ(assign, 1);
  int id = TF_OpIsStateful("Identity", status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  EXPECT_EQ(id, 0);
}

TEST(CAPI_EXPERIMENTAL, TFE_ExecuteOpInNewThreadTest_Simple) {
  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_Context* ctx = TFE_NewContext(opts, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  TFE_TensorHandle* m = TestMatrixTensorHandle();

  TFE_Op* matmul_op = MatMulOp(ctx, m, m);

  TFE_TensorHandle* retvals[1] = {nullptr};
  int num_retvals = 1;

  auto* r =
      TFE_ExecuteOpInNewThread(matmul_op, &retvals[0], &num_retvals, status);

  TFE_ExecuteOpNotificationWaitAndDelete(r, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  TF_Tensor* t = TFE_TensorHandleResolve(retvals[0], status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  float product[4] = {0};
  EXPECT_EQ(sizeof(product), TF_TensorByteSize(t));
  memcpy(&product[0], TF_TensorData(t), TF_TensorByteSize(t));
  TF_DeleteTensor(t);
  EXPECT_EQ(7, product[0]);
  EXPECT_EQ(10, product[1]);
  EXPECT_EQ(15, product[2]);
  EXPECT_EQ(22, product[3]);

  TFE_DeleteOp(matmul_op);
  TFE_DeleteTensorHandle(m);

  TFE_DeleteTensorHandle(retvals[0]);
  TFE_DeleteContext(ctx);
  TF_DeleteStatus(status);
}

// Perform a send/recv test. Recv blocks, so they need to be executed
// asynchronously.
TEST(CAPI_EXPERIMENTAL, TFE_ExecuteOpInNewThreadTest_Blocking) {
  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_Context* ctx = TFE_NewContext(opts, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  // Returns a 2x2 float32 Tensor on the CPU, with data 1., 2., 3., 4.
  TFE_TensorHandle* m = TestMatrixTensorHandle();

  // Build a send op.
  TFE_Op* send_op = TFE_NewOp(ctx, "_Send", status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_OpAddInput(send_op, m, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  string tensor_name = "Tensor";
  TFE_OpSetAttrType(send_op, "T", TF_FLOAT);
  TFE_OpSetAttrString(send_op, "tensor_name", tensor_name.c_str(),
                      tensor_name.size());
  string send_device = "/job:localhost/replica:0/task:0/device:CPU:0";
  TFE_OpSetAttrString(send_op, "send_device", send_device.c_str(),
                      send_device.size());
  TFE_OpSetAttrInt(send_op, "send_device_incarnation", 1234);
  string recv_device = "/job:localhost/replica:0/task:0/device:CPU:0";
  TFE_OpSetAttrString(send_op, "recv_device", recv_device.c_str(),
                      recv_device.size());
  TFE_OpSetAttrBool(send_op, "client_terminated", true);

  // Build a recv op.
  TFE_Op* recv_op = TFE_NewOp(ctx, "_Recv", status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  TFE_OpSetAttrType(recv_op, "tensor_type", TF_FLOAT);
  TFE_OpSetAttrString(recv_op, "tensor_name", tensor_name.c_str(),
                      tensor_name.size());
  TFE_OpSetAttrString(recv_op, "send_device", send_device.c_str(),
                      send_device.size());
  TFE_OpSetAttrInt(recv_op, "send_device_incarnation", 1234);
  TFE_OpSetAttrString(recv_op, "recv_device", recv_device.c_str(),
                      recv_device.size());
  TFE_OpSetAttrBool(recv_op, "client_terminated", true);

  TFE_TensorHandle* send_retvals;
  int send_num_retvals = 0;
  auto* send_result = TFE_ExecuteOpInNewThread(send_op, &send_retvals,
                                               &send_num_retvals, status);

  TFE_TensorHandle* recv_retvals[1] = {nullptr};
  int recv_num_retvals = 1;
  auto* recv_result = TFE_ExecuteOpInNewThread(recv_op, &recv_retvals[0],
                                               &recv_num_retvals, status);

  TFE_ExecuteOpNotificationWaitAndDelete(send_result, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_ExecuteOpNotificationWaitAndDelete(recv_result, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  TF_Tensor* t = TFE_TensorHandleResolve(recv_retvals[0], status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  float product[4] = {0};
  EXPECT_EQ(sizeof(product), TF_TensorByteSize(t));
  memcpy(&product[0], TF_TensorData(t), TF_TensorByteSize(t));
  TF_DeleteTensor(t);
  EXPECT_EQ(1, product[0]);
  EXPECT_EQ(2, product[1]);
  EXPECT_EQ(3, product[2]);
  EXPECT_EQ(4, product[3]);

  TFE_DeleteOp(send_op);
  TFE_DeleteOp(recv_op);
  TFE_DeleteTensorHandle(m);

  TFE_DeleteTensorHandle(recv_retvals[0]);
  TFE_DeleteContext(ctx);
  TF_DeleteStatus(status);
}

}  // namespace
}  // namespace tensorflow
