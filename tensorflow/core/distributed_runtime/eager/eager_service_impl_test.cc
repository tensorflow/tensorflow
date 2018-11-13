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

#include "tensorflow/core/distributed_runtime/eager/eager_service_impl.h"

#include <string.h>

#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.h"
#include "tensorflow/core/distributed_runtime/session_mgr.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/eager_service.pb.h"
#include "tensorflow/core/protobuf/tensorflow_server.pb.h"

namespace tensorflow {
namespace eager {
namespace {

class TestEagerServiceImpl : public EagerServiceImpl {
 public:
  explicit TestEagerServiceImpl(const WorkerEnv* env) : EagerServiceImpl(env) {}
  Status GetTensorHandle(const uint64 context_id,
                         const RemoteTensorHandleInternal& remote_handle,
                         tensorflow::TensorHandle** handle) {
    ServerContext* context = nullptr;
    TF_RETURN_IF_ERROR(GetServerContext(context_id, &context));
    core::ScopedUnref context_unref(context);

    return context->GetTensorHandle(remote_handle, handle);
  }
};

class EagerServiceImplTest : public ::testing::Test {
 public:
  EagerServiceImplTest()
      : rendezvous_mgr_(&worker_env_),
        session_mgr_(new SessionMgr(
            &worker_env_, "/job:localhost/replica:0/task:0/device:CPU:0",
            std::unique_ptr<WorkerCacheInterface>(),
            [](const ServerDef& server_def,
               WorkerCacheInterface** worker_cache) {
              *worker_cache = nullptr;
              return Status::OK();
            })) {
    worker_env_.env = Env::Default();

    worker_env_.rendezvous_mgr = &rendezvous_mgr_;
    worker_env_.session_mgr = session_mgr_.get();

    Device* device = DeviceFactory::NewDevice(
        "CPU", {}, "/job:localhost/replica:0/task:0/device:CPU:0");

    worker_env_.local_devices = {device};

    device_mgr_.reset(new DeviceMgr(worker_env_.local_devices));
    worker_env_.device_mgr = device_mgr_.get();
  }

 protected:
  WorkerEnv worker_env_;
  tensorflow::RpcRendezvousMgr rendezvous_mgr_;
  std::unique_ptr<SessionMgr> session_mgr_;
  std::unique_ptr<DeviceMgr> device_mgr_;
};

void SetTensorProto(TensorProto* tensor_proto) {
  int64_t dims[] = {2, 2};
  float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
  TF_Tensor* t = TF_AllocateTensor(
      TF_FLOAT, &dims[0], sizeof(dims) / sizeof(int64_t), sizeof(data));
  memcpy(TF_TensorData(t), &data[0], TF_TensorByteSize(t));
  tensorflow::Tensor tensor;
  TF_ASSERT_OK(tensorflow::TF_TensorToTensor(t, &tensor));
  tensor.AsProtoTensorContent(tensor_proto);
  TF_DeleteTensor(t);
}

void AddOperationToEnqueueRequest(
    int64 id, const string& name,
    const std::vector<std::pair<int64, int32>>& inputs,
    const std::unordered_map<string, AttrValue>& attrs, const string& device,
    EnqueueRequest* request) {
  auto* operation = request->add_queue()->mutable_operation();

  operation->set_id(id);
  operation->set_name(name);
  operation->set_device(device);

  for (const auto& tensor_handle_pair : inputs) {
    auto* input = operation->add_inputs();
    input->set_op_id(tensor_handle_pair.first);
    input->set_output_num(tensor_handle_pair.second);
  }

  for (const auto& attr_entry : attrs) {
    (*operation->mutable_attrs())[attr_entry.first] = attr_entry.second;
  }
}

tensorflow::FunctionDef MatMulFunction() {
  tensorflow::FunctionDef def;
  CHECK(tensorflow::protobuf::TextFormat::ParseFromString(
      "    signature {"
      "      name: 'MatMulFunction'"
      "      input_arg {"
      "        name: 'a'"
      "        type: DT_FLOAT"
      "      }"
      "      output_arg {"
      "        name: 'm'"
      "        type: DT_FLOAT"
      "      }"
      "    }"
      "    node_def {"
      "      name: 'matmul'"
      "      op: 'MatMul'"
      "      input: 'a'"
      "      input: 'a'"
      "      attr {"
      "        key: 'T'"
      "        value {"
      "          type: DT_FLOAT"
      "        }"
      "      }"
      "    }"
      "    ret {"
      "      key: 'm'"
      "      value: 'matmul:product'"
      "    }",
      &def));
  return def;
}

// Test creates a context and attempts to execute some ops.
TEST_F(EagerServiceImplTest, BasicTest) {
  TestEagerServiceImpl eager_service_impl(&worker_env_);

  CreateContextRequest request;
  request.mutable_server_def()->set_job_name("localhost");
  request.mutable_server_def()->set_task_index(0);
  request.set_rendezvous_id(random::New64());
  CreateContextResponse response;

  TF_ASSERT_OK(eager_service_impl.CreateContext(&request, &response));

  uint64 context_id = response.context_id();

  EnqueueRequest remote_enqueue_request;
  remote_enqueue_request.set_context_id(context_id);
  EnqueueResponse remote_enqueue_response;

  std::unordered_map<string, AttrValue> const_attrs;
  AttrValue val;
  val.set_type(tensorflow::DataType::DT_FLOAT);
  const_attrs.insert({"dtype", val});
  val.Clear();
  SetTensorProto(val.mutable_tensor());
  const_attrs.insert({"value", val});

  AddOperationToEnqueueRequest(1, "Const", {}, const_attrs,
                               "/job:localhost/replica:0/task:0/device:CPU:0",
                               &remote_enqueue_request);

  std::unordered_map<string, AttrValue> attrs;
  val.Clear();
  val.set_type(tensorflow::DataType::DT_FLOAT);
  attrs.insert({"T", val});
  val.Clear();
  val.set_b(false);
  attrs.insert({"transpose_a", val});
  attrs.insert({"transpose_b", val});

  AddOperationToEnqueueRequest(2, "MatMul", {{1, 0}, {1, 0}}, attrs,
                               "/job:localhost/replica:0/task:0/device:CPU:0",
                               &remote_enqueue_request);

  TF_ASSERT_OK(eager_service_impl.Enqueue(&remote_enqueue_request,
                                          &remote_enqueue_response));

  auto& matmul_result_shape =
      remote_enqueue_response.queue_response(1).shape(0);
  EXPECT_EQ(matmul_result_shape.dim(0).size(), 2);
  EXPECT_EQ(matmul_result_shape.dim(1).size(), 2);

  tensorflow::TensorHandle* tensor_handle;
  TF_ASSERT_OK(eager_service_impl.GetTensorHandle(
      response.context_id(), RemoteTensorHandleInternal(2, 0), &tensor_handle));

  // This should be OK to do since we've placed all computation on the CPU
  // device.
  const tensorflow::Tensor* t = nullptr;
  TF_ASSERT_OK(tensor_handle->Tensor(&t));

  auto actual = t->flat<float>();

  EXPECT_EQ(4, actual.size());

  EXPECT_EQ(7, actual(0));
  EXPECT_EQ(10, actual(1));
  EXPECT_EQ(15, actual(2));
  EXPECT_EQ(22, actual(3));

  CloseContextRequest close_context_request;
  close_context_request.set_context_id(context_id);
  CloseContextResponse close_context_response;
  TF_ASSERT_OK(eager_service_impl.CloseContext(&close_context_request,
                                               &close_context_response));
}

// Test creates a context and attempts to execute a function.
TEST_F(EagerServiceImplTest, BasicFunctionTest) {
  TestEagerServiceImpl eager_service_impl(&worker_env_);

  CreateContextRequest request;
  request.mutable_server_def()->set_job_name("localhost");
  request.mutable_server_def()->set_task_index(0);
  request.set_rendezvous_id(random::New64());
  CreateContextResponse response;

  TF_ASSERT_OK(eager_service_impl.CreateContext(&request, &response));

  uint64 context_id = response.context_id();

  RegisterFunctionRequest register_function_request;
  register_function_request.set_context_id(context_id);
  *register_function_request.mutable_function_def() = MatMulFunction();
  RegisterFunctionResponse register_function_response;

  TF_ASSERT_OK(eager_service_impl.RegisterFunction(
      &register_function_request, &register_function_response));

  EnqueueRequest remote_enqueue_request;
  remote_enqueue_request.set_context_id(context_id);
  EnqueueResponse remote_enqueue_response;

  std::unordered_map<string, AttrValue> const_attrs;
  AttrValue val;
  val.set_type(tensorflow::DataType::DT_FLOAT);
  const_attrs.insert({"dtype", val});
  val.Clear();

  SetTensorProto(val.mutable_tensor());
  const_attrs.insert({"value", val});

  AddOperationToEnqueueRequest(1, "Const", {}, const_attrs,
                               "/job:localhost/replica:0/task:0/device:CPU:0",
                               &remote_enqueue_request);
  AddOperationToEnqueueRequest(
      2, "MatMulFunction", {{1, 0}}, std::unordered_map<string, AttrValue>(),
      "/job:localhost/replica:0/task:0/device:CPU:0", &remote_enqueue_request);

  TF_ASSERT_OK(eager_service_impl.Enqueue(&remote_enqueue_request,
                                          &remote_enqueue_response));

  const tensorflow::Tensor* t = nullptr;
  tensorflow::TensorHandle* tensor_handle;
  TF_ASSERT_OK(eager_service_impl.GetTensorHandle(
      response.context_id(), RemoteTensorHandleInternal(2, 0), &tensor_handle));
  TF_ASSERT_OK(tensor_handle->Tensor(&t));

  auto actual = t->flat<float>();
  EXPECT_EQ(4, actual.size());

  EXPECT_EQ(7, actual(0));
  EXPECT_EQ(10, actual(1));
  EXPECT_EQ(15, actual(2));
  EXPECT_EQ(22, actual(3));

  CloseContextRequest close_context_request;
  close_context_request.set_context_id(context_id);
  CloseContextResponse close_context_response;
  TF_ASSERT_OK(eager_service_impl.CloseContext(&close_context_request,
                                               &close_context_response));
}

// Test creates a context and attempts to send a tensor (using the RPC), and
// then use the tensor.
TEST_F(EagerServiceImplTest, SendTensorTest) {
  TestEagerServiceImpl eager_service_impl(&worker_env_);

  CreateContextRequest request;
  request.mutable_server_def()->set_job_name("localhost");
  request.mutable_server_def()->set_task_index(0);
  request.set_rendezvous_id(random::New64());
  CreateContextResponse response;

  TF_ASSERT_OK(eager_service_impl.CreateContext(&request, &response));

  uint64 context_id = response.context_id();

  SendTensorRequest send_tensor_request;
  send_tensor_request.set_context_id(context_id);
  send_tensor_request.set_op_id(1);
  SetTensorProto(send_tensor_request.add_tensors());
  SendTensorResponse send_tensor_response;

  TF_ASSERT_OK(eager_service_impl.SendTensor(&send_tensor_request,
                                             &send_tensor_response));

  EnqueueRequest remote_enqueue_request;
  remote_enqueue_request.set_context_id(context_id);
  EnqueueResponse remote_enqueue_response;

  std::unordered_map<string, AttrValue> attrs;
  AttrValue val;
  val.Clear();
  val.set_type(tensorflow::DataType::DT_FLOAT);
  attrs.insert({"T", val});
  val.Clear();
  val.set_b(false);
  attrs.insert({"transpose_a", val});
  attrs.insert({"transpose_b", val});

  AddOperationToEnqueueRequest(2, "MatMul", {{1, 0}, {1, 0}}, attrs,
                               "/job:localhost/replica:0/task:0/device:CPU:0",
                               &remote_enqueue_request);

  TF_ASSERT_OK(eager_service_impl.Enqueue(&remote_enqueue_request,
                                          &remote_enqueue_response));

  const tensorflow::Tensor* t = nullptr;
  tensorflow::TensorHandle* tensor_handle;
  TF_ASSERT_OK(eager_service_impl.GetTensorHandle(
      response.context_id(), RemoteTensorHandleInternal(2, 0), &tensor_handle));
  TF_ASSERT_OK(tensor_handle->Tensor(&t));

  Device* device = nullptr;
  TF_ASSERT_OK(tensor_handle->Device(&device));
  EXPECT_NE(device, nullptr);
  EXPECT_EQ(device->name(), "/job:localhost/replica:0/task:0/device:CPU:0");

  auto actual = t->flat<float>();
  EXPECT_EQ(4, actual.size());

  EXPECT_EQ(7, actual(0));
  EXPECT_EQ(10, actual(1));
  EXPECT_EQ(15, actual(2));
  EXPECT_EQ(22, actual(3));

  CloseContextRequest close_context_request;
  close_context_request.set_context_id(context_id);
  CloseContextResponse close_context_response;
  TF_ASSERT_OK(eager_service_impl.CloseContext(&close_context_request,
                                               &close_context_response));
}

TEST_F(EagerServiceImplTest, KeepAliveTest) {
  TestEagerServiceImpl eager_service_impl(&worker_env_);

  CreateContextRequest request;
  request.mutable_server_def()->set_job_name("localhost");
  request.mutable_server_def()->set_task_index(0);
  request.set_rendezvous_id(random::New64());
  request.set_keep_alive_secs(3);
  CreateContextResponse response;

  TF_ASSERT_OK(eager_service_impl.CreateContext(&request, &response));

  worker_env_.env->SleepForMicroseconds(5 *
                                        tensorflow::EnvTime::kSecondsToMicros);

  KeepAliveRequest keep_alive_request;
  KeepAliveResponse keep_alive_response;

  keep_alive_request.set_context_id(response.context_id());

  Status status =
      eager_service_impl.KeepAlive(&keep_alive_request, &keep_alive_response);

  EXPECT_EQ(status.code(), error::INVALID_ARGUMENT);
  EXPECT_PRED_FORMAT2(::testing::IsSubstring, "Unable to find a context_id",
                      status.error_message());

  // Create a new context.
  request.set_rendezvous_id(random::New64());
  TF_ASSERT_OK(eager_service_impl.CreateContext(&request, &response));

  // The context should not be GC'd.
  worker_env_.env->SleepForMicroseconds(1 *
                                        tensorflow::EnvTime::kSecondsToMicros);

  keep_alive_request.set_context_id(response.context_id());

  TF_ASSERT_OK(
      eager_service_impl.KeepAlive(&keep_alive_request, &keep_alive_response));
}

}  // namespace
}  // namespace eager
}  // namespace tensorflow
