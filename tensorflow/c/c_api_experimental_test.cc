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

#include "absl/types/optional.h"
#include "tensorflow/c/c_api_internal.h"
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
  EXPECT_TRUE(absl::StrContains(TF_Message(status),
                                "Invalid text proto for ServerDef"));
  EXPECT_EQ(null_result, nullptr);

  // Cleanup
  TF_DeleteBuffer(result);
  TF_DeleteStatus(status);
}

TEST(CAPI_EXPERIMENTAL, InitializeTPUSystemTest) {
  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_Context* ctx = TFE_NewContext(opts, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);
  TF_Buffer* buf = TF_NewBuffer();
  TFE_InitializeTPUSystem(ctx, "localhost", buf, status);
  // Note that this assumes TPUs are not available for this test.
  CHECK_EQ(TF_NOT_FOUND, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteBuffer(buf);
  TF_DeleteStatus(status);
  TFE_DeleteContext(ctx);
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

class ShapeInferenceTest : public ::testing::Test {
 protected:
  ShapeInferenceTest()
      : status_(TF_NewStatus()), tfe_context_options_(TFE_NewContextOptions()) {
    tfe_context_ = TFE_NewContext(tfe_context_options_, status_);
    CHECK_EQ(TF_OK, TF_GetCode(status_)) << TF_Message(status_);
  }

  ~ShapeInferenceTest() override {
    TFE_DeleteContextOptions(tfe_context_options_);
    TFE_DeleteContext(tfe_context_);
    TF_DeleteStatus(status_);
  }

  // Checks the expected result of shape inference for the given `op`.
  void CheckOutputShapes(
      TFE_Op* op,
      const std::vector<absl::optional<std::vector<int64_t>>>& input_shapes_vec,
      const std::vector<TF_Tensor*>& input_tensors,
      const absl::optional<std::vector<int64_t>>& expected_shape) {
    // Create input_shapes.
    TF_ShapeAndTypeList* input_shapes =
        TF_NewShapeAndTypeList(input_shapes_vec.size());
    for (size_t i = 0; i < input_shapes_vec.size(); ++i) {
      const auto& input_shape = input_shapes_vec[i];
      if (input_shape.has_value()) {
        TF_ShapeAndTypeListSetShape(input_shapes, i, input_shape->data(),
                                    input_shape->size());
      } else {
        TF_ShapeAndTypeListSetUnknownShape(input_shapes, i);
      }
    }
    TF_ShapeAndTypeList* output_shapes;
    TFE_InferShapes(op, input_shapes,
                    input_tensors.empty()
                        ? nullptr
                        : const_cast<TF_Tensor**>(input_tensors.data()),
                    /*input_tensors_as_shapes*/ nullptr,
                    /*input_resource_shapes_and_types*/ nullptr, &output_shapes,
                    /*output_resource_shapes_and_types*/ nullptr, status_);
    CHECK_EQ(TF_OK, TF_GetCode(status_)) << TF_Message(status_);
    CHECK_EQ(output_shapes->num_items, 1);

    int num_dims = output_shapes->items[0].num_dims;
    int64_t* dims = output_shapes->items[0].dims;

    if (!expected_shape.has_value()) {
      EXPECT_EQ(num_dims, -1);
      EXPECT_EQ(dims, nullptr);
      return;
    }

    EXPECT_EQ(num_dims, expected_shape->size());
    for (size_t i = 0; i < num_dims; ++i) {
      EXPECT_EQ(dims[i], (*expected_shape)[i]);
    }
    TF_DeleteShapeAndTypeList(input_shapes);
    TF_DeleteShapeAndTypeList(output_shapes);
  }

  absl::optional<std::vector<int64_t>> make_shape(
      std::vector<int64_t>&& dims) const {
    return absl::make_optional(dims);
  }

  absl::optional<std::vector<int64_t>> unknown_shape() const {
    return absl::nullopt;
  }

  static constexpr int64_t kUnknownDim =
      shape_inference::InferenceContext::kUnknownDim;
  TF_Status* status_;
  TFE_ContextOptions* tfe_context_options_;
  TFE_Context* tfe_context_;
};

TEST_F(ShapeInferenceTest, InfersShapesFromInputShapes) {
  TFE_Op* matmul_op;
  matmul_op = TFE_NewOp(tfe_context_, "MatMul", status_);
  CHECK_EQ(TF_OK, TF_GetCode(status_)) << TF_Message(status_);

  // Infer shape when everything is known.
  CheckOutputShapes(matmul_op,
                    /*input_shapes*/ {make_shape({3, 2}), make_shape({2, 4})},
                    /*input_tensors*/ {},
                    /*expected_shape*/ make_shape({3, 4}));

  // Infer shape when second operand has unknown shape.
  CheckOutputShapes(matmul_op,
                    /*input_shapes*/ {make_shape({3, 2}), unknown_shape()},
                    /*input_tensors*/ {},
                    /*expected_shape*/ make_shape({3, kUnknownDim}));

  // Infer shape when some dimensions are unknown.
  CheckOutputShapes(
      matmul_op,
      /*input_shapes*/ {make_shape({kUnknownDim, 2}), make_shape({2, 4})},
      /*input_tensors*/ {},
      /*expected_shape*/ make_shape({kUnknownDim, 4}));

  // Infer shape when everything is unknown.
  CheckOutputShapes(matmul_op,
                    /*input_shapes*/ {unknown_shape(), unknown_shape()},
                    /*input_tensors*/ {},
                    /*expected_shape*/ make_shape({kUnknownDim, kUnknownDim}));

  TFE_DeleteOp(matmul_op);
  // TODO(bgogul): Add some death tests where status is not OK.
}

TEST_F(ShapeInferenceTest, InfersShapesFromInputTensors) {
  // Prepare some tensors for shape.
  TF_Tensor* tensor_1X6 = Int32Tensor({1, 6});
  CHECK_EQ(TF_OK, TF_GetCode(status_)) << TF_Message(status_);
  TF_Tensor* tensor_1X1X6 = Int32Tensor({1, 1, 6});
  CHECK_EQ(TF_OK, TF_GetCode(status_)) << TF_Message(status_);

  TFE_Op* reshape_op = TFE_NewOp(tfe_context_, "Reshape", status_);
  CHECK_EQ(TF_OK, TF_GetCode(status_)) << TF_Message(status_);
  TFE_OpSetAttrType(reshape_op, "T", TF_FLOAT);
  TFE_OpSetAttrType(reshape_op, "Tshape", TF_INT32);
  CheckOutputShapes(reshape_op,
                    /* input_shapes*/ {unknown_shape(), unknown_shape()},
                    /* input_tensors*/ {nullptr, tensor_1X6},
                    /*expected_shape*/ make_shape({1, 6}));
  TFE_DeleteOp(reshape_op);
  reshape_op = nullptr;

  TFE_Op* fill_op = TFE_NewOp(tfe_context_, "Fill", status_);
  CHECK_EQ(TF_OK, TF_GetCode(status_)) << TF_Message(status_);
  TFE_OpSetAttrType(fill_op, "T", TF_FLOAT);
  TFE_OpSetAttrType(fill_op, "Tshape", TF_INT32);

  float five = 5.0;
  TFE_TensorHandle* scalar = TestScalarTensorHandle(five);
  TF_Tensor* scalarTensor = TFE_TensorHandleResolve(scalar, status_);
  CHECK_EQ(TF_OK, TF_GetCode(status_)) << TF_Message(status_);
  CheckOutputShapes(fill_op,
                    /* input_shapes*/ {unknown_shape(), unknown_shape()},
                    /* input_tensors*/ {tensor_1X1X6, scalarTensor},
                    /*expected_shape*/ make_shape({1, 1, 6}));
  TFE_DeleteOp(fill_op);
  fill_op = nullptr;

  TFE_DeleteTensorHandle(scalar);
  TF_DeleteTensor(scalarTensor);
  TF_DeleteTensor(tensor_1X1X6);
  TF_DeleteTensor(tensor_1X6);
}

}  // namespace
}  // namespace tensorflow
