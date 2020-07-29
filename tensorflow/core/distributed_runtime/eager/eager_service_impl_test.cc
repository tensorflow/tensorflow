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

#include "absl/types/optional.h"
#include "absl/types/variant.h"
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/core/common_runtime/eager/kernel_and_device.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/distributed_runtime/eager/cluster_function_library_runtime.h"
#include "tensorflow/core/distributed_runtime/eager/remote_mgr.h"
#include "tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.h"
#include "tensorflow/core/distributed_runtime/session_mgr.h"
#include "tensorflow/core/distributed_runtime/test_utils.h"
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
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/protobuf/remote_tensor_handle.pb.h"
#include "tensorflow/core/protobuf/tensorflow_server.pb.h"

namespace tensorflow {
namespace eager {
namespace {

class TestEagerServiceImpl : public EagerServiceImpl {
 public:
  explicit TestEagerServiceImpl(const WorkerEnv* env) : EagerServiceImpl(env) {}
  Status GetEagerContext(const uint64 context_id, EagerContext** ctx) {
    ServerContext* context = nullptr;
    TF_RETURN_IF_ERROR(GetServerContext(context_id, &context));
    core::ScopedUnref context_unref(context);
    *ctx = context->Context();
    return Status::OK();
  }
  Status GetTensorHandle(const uint64 context_id,
                         const RemoteTensorHandleInternal& remote_handle,
                         tensorflow::TensorHandle** handle) {
    ServerContext* context = nullptr;
    TF_RETURN_IF_ERROR(GetServerContext(context_id, &context));
    core::ScopedUnref context_unref(context);

    return context->Context()->RemoteMgr()->GetTensorHandle(remote_handle,
                                                            handle);
  }
};

class FakeEagerClient : public EagerClient {
 public:
  FakeEagerClient() {}
  ~FakeEagerClient() override {}

  void SetServiceImpl(TestEagerServiceImpl* impl) { impl_ = impl; }

#define CLIENT_METHOD(method)                                         \
  void method##Async(const method##Request* request,                  \
                     method##Response* response, StatusCallback done) \
      override {                                                      \
    done(impl_->method(request, response));                           \
  }

  CLIENT_METHOD(CreateContext);
  CLIENT_METHOD(UpdateContext);
  CLIENT_METHOD(Enqueue);
  CLIENT_METHOD(WaitQueueDone);
  CLIENT_METHOD(KeepAlive);
  CLIENT_METHOD(CloseContext);
#undef CLIENT_METHOD

  void RunComponentFunctionAsync(CallOptions* call_opts,
                                 const RunComponentFunctionRequest* request,
                                 RunComponentFunctionResponse* response,
                                 StatusCallback done) override {
    impl_->RunComponentFunction(call_opts, request, response, std::move(done));
  }

  void StreamingEnqueueAsync(const EnqueueRequest* request,
                             EnqueueResponse* response,
                             StatusCallback done) override {
    done(impl_->Enqueue(request, response));
  }

  bool allow_multiple_pending_requests() const override { return false; }

 private:
  TestEagerServiceImpl* impl_;
};

class DummyEagerClientCache : public EagerClientCache {
 public:
  DummyEagerClientCache() : client_(new FakeEagerClient) {}
  Status GetClient(const string& target,
                   core::RefCountPtr<EagerClient>* client) override {
    client->reset(client_.get());
    client_->Ref();
    return Status::OK();
  }

 private:
  core::RefCountPtr<EagerClient> client_;
};

class FakeCache : public TestWorkerCache {
  Status GetEagerClientCache(
      std::unique_ptr<eager::EagerClientCache>* eager_client_cache) override {
    eager_client_cache->reset(new DummyEagerClientCache);
    return Status::OK();
  }

  void ListWorkers(std::vector<string>* workers) const override {
    workers->push_back("/job:localhost/replica:0/task:0");
  }
};

class EagerServiceImplTest : public ::testing::Test {
 public:
  EagerServiceImplTest()
      : rendezvous_mgr_(&worker_env_),
        session_mgr_(new SessionMgr(
            &worker_env_, "/job:localhost/replica:0/task:0/device:CPU:0",
            std::unique_ptr<WorkerCacheInterface>(new FakeCache),
            [](const ServerDef& server_def,
               WorkerCacheInterface** worker_cache) {
              *worker_cache = new FakeCache;
              return Status::OK();
            })) {
    worker_env_.env = Env::Default();

    worker_env_.rendezvous_mgr = &rendezvous_mgr_;
    worker_env_.session_mgr = session_mgr_.get();

    device_mgr_ = absl::make_unique<StaticDeviceMgr>(
        DeviceFactory::NewDevice("CPU", {}, "/job:localhost/replica:0/task:0"));
    worker_env_.local_devices = device_mgr_->ListDevices();
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

void BuildOperation(
    Operation* operation, int64 id, const string& name,
    const std::vector<absl::variant<TensorProto, std::pair<int64, int32>>>&
        inputs,
    const std::unordered_map<string, AttrValue>& attrs, const string& device) {
  operation->set_id(id);
  operation->set_name(name);
  operation->set_device(device);

  for (const auto& input : inputs) {
    if (input.index() == 0) {
      *operation->add_op_inputs()->mutable_tensor() =
          absl::get<TensorProto>(input);
    } else {
      const auto& tensor_handle_pair =
          absl::get<std::pair<int64, int32>>(input);
      auto* input = operation->add_op_inputs()->mutable_remote_handle();
      input->set_op_id(tensor_handle_pair.first);
      input->set_output_num(tensor_handle_pair.second);
      input->set_op_device(device);
      input->set_device(device);
    }
  }

  for (const auto& attr_entry : attrs) {
    (*operation->mutable_attrs())[attr_entry.first] = attr_entry.second;
  }
}

void AddOperationToEnqueueRequest(
    int64 id, const string& name,
    const std::vector<absl::variant<TensorProto, std::pair<int64, int32>>>&
        inputs,
    const std::unordered_map<string, AttrValue>& attrs, const string& device,
    EnqueueRequest* request) {
  auto* operation = request->add_queue()->mutable_operation();
  BuildOperation(operation, id, name, inputs, attrs, device);
}

void AddOperationToRunComponentFunctionRequest(
    int64 id, const string& name,
    const std::vector<absl::variant<TensorProto, std::pair<int64, int32>>>&
        inputs,
    const std::unordered_map<string, AttrValue>& attrs, const string& device,
    RunComponentFunctionRequest* request) {
  auto* operation = request->mutable_operation();
  operation->set_is_function(true);
  operation->set_is_component_function(true);
  BuildOperation(operation, id, name, inputs, attrs, device);
}

tensorflow::NodeDef MatMulFunctionNodeDef() {
  tensorflow::NodeDef def;
  CHECK(tensorflow::protobuf::TextFormat::ParseFromString(
      "    name: 'matmul_func'"
      "    op: 'MatMulFunction'"
      "    input: 'a'"
      "    input: 'a'"
      "    attr {"
      "      key: 'T'"
      "      value {"
      "        type: DT_FLOAT"
      "      }"
      "    }",
      &def));
  return def;
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
      "      attr {"
      "        key: 'transpose_a'"
      "        value {"
      "          b: false"
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

tensorflow::FunctionDef MatMulNestedFunction() {
  tensorflow::FunctionDef def;
  CHECK(tensorflow::protobuf::TextFormat::ParseFromString(
      "    signature {"
      "      name: 'MatMulNestedFunction'"
      "      input_arg {"
      "        name: 'a'"
      "        type: DT_FLOAT"
      "      }"
      "      output_arg {"
      "        name: 'matmul_nested'"
      "        type: DT_FLOAT"
      "      }"
      "    }"
      "    node_def {"
      "      name: 'matmul_nested'"
      "      op: 'MatMulFunction'"
      "      input: 'a'"
      "      attr {"
      "        key: 'T'"
      "        value {"
      "          type: DT_FLOAT"
      "        }"
      "      }"
      "    }"
      "    ret {"
      "      key: 'matmul_nested'"
      "      value: 'matmul_nested:m:0'"
      "    }",
      &def));
  return def;
}

tensorflow::FunctionDef SingleRecvNodeFunction() {
  tensorflow::FunctionDef def;
  CHECK(tensorflow::protobuf::TextFormat::ParseFromString(
      "    signature {"
      "      name: 'SingleRecvNodeFunction'"
      "      input_arg {"
      "        name: 'a'"
      "        type: DT_FLOAT"
      "      }"
      "      output_arg {"
      "        name: 'recv_tensor'"
      "        type: DT_FLOAT"
      "      }"
      "    }"
      "    node_def {"
      "      name: 'recv_node'"
      "      op: '_Recv'"
      "      device: '/job:localhost/replica:0/task:0/device:CPU:0'"
      "      attr {"
      "        key: 'client_terminated'"
      "        value {"
      "          b: true"
      "        }"
      "      }"
      "      attr {"
      "        key: 'recv_device'"
      "        value {"
      "          s: '/job:localhost/replica:0/task:0/device:CPU:0'"
      "        }"
      "      }"
      "      attr {"
      "        key: 'send_device'"
      "        value {"
      "          s: '/job:localhost/replica:0/task:0/device:CPU:0'"
      "        }"
      "      }"
      "      attr {"
      "        key: 'send_device_incarnation'"
      "        value {"
      "          i: 1"
      "        }"
      "      }"
      "      attr {"
      "        key: 'tensor_name'"
      "        value {"
      "          s: 't0'"
      "        }"
      "      }"
      "      attr {"
      "        key: 'tensor_type'"
      "        value {"
      "          type: DT_FLOAT"
      "        }"
      "      }"
      "    }"
      "    ret {"
      "      key: 'recv_tensor'"
      "      value: 'recv_node:tensor:0'"
      "    }",
      &def));
  return def;
}

// Test creates a context and attempts to execute some ops.
TEST_F(EagerServiceImplTest, BasicTest) {
  TestEagerServiceImpl eager_service_impl(&worker_env_);

  uint64 context_id = random::New64();

  CreateContextRequest request;
  request.mutable_server_def()->set_job_name("localhost");
  request.mutable_server_def()->set_task_index(0);
  request.set_context_id(context_id);
  CreateContextResponse response;

  TF_ASSERT_OK(eager_service_impl.CreateContext(&request, &response));

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

  AddOperationToEnqueueRequest(
      2, "MatMul", {std::make_pair(1, 0), std::make_pair(1, 0)}, attrs,
      "/job:localhost/replica:0/task:0/device:CPU:0", &remote_enqueue_request);

  TF_ASSERT_OK(eager_service_impl.Enqueue(&remote_enqueue_request,
                                          &remote_enqueue_response));

  auto& matmul_result_shape =
      remote_enqueue_response.queue_response(1).shape(0);
  EXPECT_EQ(matmul_result_shape.dim(0).size(), 2);
  EXPECT_EQ(matmul_result_shape.dim(1).size(), 2);

  tensorflow::TensorHandle* tensor_handle;
  TF_ASSERT_OK(eager_service_impl.GetTensorHandle(
      context_id, RemoteTensorHandleInternal(2, 0), &tensor_handle));

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
  close_context_request.set_context_view_id(0);
  CloseContextResponse close_context_response;
  TF_ASSERT_OK(eager_service_impl.CloseContext(&close_context_request,
                                               &close_context_response));
}

class EagerServiceImplFunctionTest : public EagerServiceImplTest {
 public:
  EagerServiceImplFunctionTest() : EagerServiceImplTest() {}

  // Creates a context and attempts to execute a function.
  void TestFunction(const RegisterFunctionOp& register_op,
                    const string& function_name,
                    const bool local_inputs = false) {
    TestEagerServiceImpl eager_service_impl(&worker_env_);

    uint64 context_id = random::New64();

    CreateContextRequest request;
    request.mutable_server_def()->set_job_name("localhost");
    request.mutable_server_def()->set_task_index(0);
    request.set_context_id(context_id);
    CreateContextResponse response;

    TF_ASSERT_OK(eager_service_impl.CreateContext(&request, &response));

    EnqueueRequest enqueue_request;
    enqueue_request.set_context_id(context_id);
    *enqueue_request.add_queue()->mutable_register_function() = register_op;
    EnqueueResponse enqueue_response;

    TF_ASSERT_OK(
        eager_service_impl.Enqueue(&enqueue_request, &enqueue_response));

    EnqueueRequest remote_enqueue_request;
    remote_enqueue_request.set_context_id(context_id);
    EnqueueResponse remote_enqueue_response;

    if (local_inputs) {
      TensorProto tensor_proto;
      SetTensorProto(&tensor_proto);
      AddOperationToEnqueueRequest(
          2, function_name, {tensor_proto},
          std::unordered_map<string, AttrValue>(),
          "/job:localhost/replica:0/task:0/device:CPU:0",
          &remote_enqueue_request);

    } else {
      std::unordered_map<string, AttrValue> const_attrs;
      AttrValue val;
      val.set_type(tensorflow::DataType::DT_FLOAT);
      const_attrs.insert({"dtype", val});
      val.Clear();

      SetTensorProto(val.mutable_tensor());
      const_attrs.insert({"value", val});

      AddOperationToEnqueueRequest(
          1, "Const", {}, const_attrs,
          "/job:localhost/replica:0/task:0/device:CPU:0",
          &remote_enqueue_request);
      AddOperationToEnqueueRequest(
          2, function_name, {std::make_pair(1, 0)},
          std::unordered_map<string, AttrValue>(),
          "/job:localhost/replica:0/task:0/device:CPU:0",
          &remote_enqueue_request);
    }

    TF_ASSERT_OK(eager_service_impl.Enqueue(&remote_enqueue_request,
                                            &remote_enqueue_response));

    const tensorflow::Tensor* t = nullptr;
    tensorflow::TensorHandle* tensor_handle;
    TF_ASSERT_OK(eager_service_impl.GetTensorHandle(
        context_id, RemoteTensorHandleInternal(2, 0), &tensor_handle));
    TF_ASSERT_OK(tensor_handle->Tensor(&t));

    auto actual = t->flat<float>();
    EXPECT_EQ(4, actual.size());

    EXPECT_EQ(7, actual(0));
    EXPECT_EQ(10, actual(1));
    EXPECT_EQ(15, actual(2));
    EXPECT_EQ(22, actual(3));

    CloseContextRequest close_context_request;
    close_context_request.set_context_id(context_id);
    close_context_request.set_context_view_id(0);
    CloseContextResponse close_context_response;
    TF_ASSERT_OK(eager_service_impl.CloseContext(&close_context_request,
                                                 &close_context_response));
  }

  // Creates a context and attempts to execute a component function.
  void TestComponentFunction(const RegisterFunctionOp& register_op,
                             const string& function_name,
                             const bool test_cancel) {
    TestEagerServiceImpl eager_service_impl(&worker_env_);
    uint64 context_id = random::New64();

    // Create context.
    CreateContextRequest request;
    request.mutable_server_def()->set_job_name("localhost");
    request.mutable_server_def()->set_task_index(0);
    request.set_context_id(context_id);
    CreateContextResponse response;
    TF_ASSERT_OK(eager_service_impl.CreateContext(&request, &response));

    // Register function.
    EnqueueRequest enqueue_request;
    enqueue_request.set_context_id(context_id);
    *enqueue_request.add_queue()->mutable_register_function() = register_op;
    EnqueueResponse enqueue_response;
    TF_ASSERT_OK(
        eager_service_impl.Enqueue(&enqueue_request, &enqueue_response));

    // First run an op to generate input for function.
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
    TF_ASSERT_OK(eager_service_impl.Enqueue(&remote_enqueue_request,
                                            &remote_enqueue_response));

    // Run function with input from the previous op.
    RunComponentFunctionRequest run_comp_func_request;
    run_comp_func_request.set_context_id(context_id);
    RunComponentFunctionResponse run_comp_func_response;
    AddOperationToRunComponentFunctionRequest(
        2, function_name, {std::make_pair(1, 0)},
        std::unordered_map<string, AttrValue>(),
        "/job:localhost/replica:0/task:0/device:CPU:0", &run_comp_func_request);

    CallOptions call_opts;
    Notification n;
    Status status;
    eager_service_impl.RunComponentFunction(&call_opts, &run_comp_func_request,
                                            &run_comp_func_response,
                                            [&status, &n](const Status& s) {
                                              status.Update(s);
                                              n.Notify();
                                            });
    if (test_cancel) {
      call_opts.StartCancel();
    }
    n.WaitForNotification();
    if (test_cancel) {
      EXPECT_TRUE(errors::IsCancelled(status)) << status.error_message();
    } else {
      TF_ASSERT_OK(status);
      // Retrieve the output.
      const tensorflow::Tensor* t = nullptr;
      tensorflow::TensorHandle* tensor_handle;
      TF_ASSERT_OK(eager_service_impl.GetTensorHandle(
          context_id, RemoteTensorHandleInternal(2, 0), &tensor_handle));
      TF_ASSERT_OK(tensor_handle->Tensor(&t));

      auto actual = t->flat<float>();
      EXPECT_EQ(4, actual.size());

      EXPECT_EQ(7, actual(0));
      EXPECT_EQ(10, actual(1));
      EXPECT_EQ(15, actual(2));
      EXPECT_EQ(22, actual(3));
    }

    CloseContextRequest close_context_request;
    close_context_request.set_context_id(context_id);
    close_context_request.set_context_view_id(0);
    CloseContextResponse close_context_response;
    TF_ASSERT_OK(eager_service_impl.CloseContext(&close_context_request,
                                                 &close_context_response));
  }
};

TEST_F(EagerServiceImplFunctionTest, BasicFunctionTest) {
  RegisterFunctionOp register_op;
  *register_op.mutable_function_def() = MatMulFunction();
  TestFunction(register_op, "MatMulFunction");
}

TEST_F(EagerServiceImplFunctionTest, FunctionWithLocalInputsTest) {
  RegisterFunctionOp register_op;
  *register_op.mutable_function_def() = MatMulFunction();
  TestFunction(register_op, "MatMulFunction", /*local_inputs=*/true);
}

TEST_F(EagerServiceImplFunctionTest, NestedFunctionTest) {
  RegisterFunctionOp register_op;
  *register_op.mutable_function_def() = MatMulNestedFunction();
  *register_op.mutable_library()->add_function() = MatMulFunction();
  TestFunction(register_op, "MatMulNestedFunction");
}

TEST_F(EagerServiceImplFunctionTest, ComponentFunctionTest) {
  RegisterFunctionOp register_op;
  *register_op.mutable_function_def() = MatMulFunction();
  TestComponentFunction(register_op, "MatMulFunction", false);
}

TEST_F(EagerServiceImplFunctionTest, ComponentFunctionCancellationTest) {
  RegisterFunctionOp register_op;
  *register_op.mutable_function_def() = SingleRecvNodeFunction();
  TestComponentFunction(register_op, "SingleRecvNodeFunction", true);
}

class FunctionWithRemoteInputsTest : public EagerServiceImplTest {
 public:
  FunctionWithRemoteInputsTest()
      : EagerServiceImplTest(), eager_service_impl_(&worker_env_) {
    remote_device_mgr_ = absl::make_unique<StaticDeviceMgr>(
        DeviceFactory::NewDevice("CPU", {}, "/job:localhost/replica:0/task:1"));
    context_id_ = random::New64();
  }

  class TestExecuteNodeArgs : public EagerKernelArgs {
   public:
    TestExecuteNodeArgs(
        gtl::InlinedVector<TensorValue, 4>&& tensor_args,
        std::function<Status(const int, eager::RemoteTensorHandle*)>
            serialize_remote_handle)
        : EagerKernelArgs(std::move(tensor_args)),
          serialize_remote_handle_(std::move(serialize_remote_handle)) {}

    bool HasRemoteOrPackedInputs() const override { return true; }

    Status GetRemoteArg(const FunctionArgIndex& index,
                        eager::RemoteTensorHandle* val) const override {
      return serialize_remote_handle_(index.index, val);
    }

   private:
    std::function<Status(const int, eager::RemoteTensorHandle*)>
        serialize_remote_handle_;
  };

  bool MatMulHasAttrWithDefaultValue(const tensorflow::FunctionDef& fdef) {
    for (const auto& node : fdef.node_def()) {
      if (node.op() == "MatMul") {
        return node.attr().find("transpose_a") != node.attr().end();
      }
    }
    return false;
  }

  void Init() {
    CreateContextRequest request;
    request.mutable_server_def()->set_job_name("localhost");
    request.mutable_server_def()->set_task_index(0);
    request.set_context_id(context_id_);
    CreateContextResponse response;
    TF_ASSERT_OK(eager_service_impl_.CreateContext(&request, &response));

    // Make the fake EagerClient use the local eager_service_impl.
    EagerContext* ctx = nullptr;
    TF_ASSERT_OK(eager_service_impl_.GetEagerContext(context_id_, &ctx));
    Device* device;
    TF_ASSERT_OK(ctx->FindDeviceFromName(local_device_.c_str(), &device));
    core::RefCountPtr<EagerClient> client;
    TF_ASSERT_OK(ctx->GetClient(device, &client));
    FakeEagerClient* fake_client = static_cast<FakeEagerClient*>(client.get());
    fake_client->SetServiceImpl(&eager_service_impl_);

    // Create an input on local_device for MatMulFunction.
    EnqueueRequest remote_enqueue_request;
    remote_enqueue_request.set_context_id(context_id_);
    EnqueueResponse remote_enqueue_response;
    std::unordered_map<string, AttrValue> const_attrs;
    AttrValue val;
    val.set_type(tensorflow::DataType::DT_FLOAT);
    const_attrs.insert({"dtype", val});
    val.Clear();
    SetTensorProto(val.mutable_tensor());
    const_attrs.insert({"value", val});
    AddOperationToEnqueueRequest(1, "Const", {}, const_attrs, local_device_,
                                 &remote_enqueue_request);
    TF_EXPECT_OK(eager_service_impl_.Enqueue(&remote_enqueue_request,
                                             &remote_enqueue_response));
    eager_cluster_flr_ = absl::make_unique<EagerClusterFunctionLibraryRuntime>(
        context_id_, ctx, device_mgr_.get());

    fdef_ = MatMulFunction();
    TF_ASSERT_OK(func_lib_def_.AddFunctionDef(fdef_));
    eager_pflr_ = absl::make_unique<ProcessFunctionLibraryRuntime>(
        remote_device_mgr_.get(), Env::Default(), /*config=*/
        nullptr, TF_GRAPH_DEF_VERSION, &func_lib_def_, OptimizerOptions(),
        /*thread_pool=*/nullptr, eager_cluster_flr_.get(),
        /*custom_kernel_creator=*/nullptr, /*session_metadata=*/nullptr,
        Rendezvous::Factory{[this](const int64 step_id,
                                   const DeviceMgr* device_mgr,
                                   Rendezvous** r) {
          *r = worker_env_.rendezvous_mgr->Find(step_id);
          return Status::OK();
        }});
  }

  void CheckOutputTensorAndClose(const Tensor& tensor) {
    auto actual = tensor.flat<float>();
    EXPECT_EQ(4, actual.size());
    EXPECT_EQ(7, actual(0));
    EXPECT_EQ(10, actual(1));
    EXPECT_EQ(15, actual(2));
    EXPECT_EQ(22, actual(3));

    CloseContextRequest close_context_request;
    close_context_request.set_context_id(context_id_);
    close_context_request.set_context_view_id(0);
    CloseContextResponse close_context_response;
    TF_ASSERT_OK(eager_service_impl_.CloseContext(&close_context_request,
                                                  &close_context_response));
  }

  void CheckOutputsAndClose(const int64 op_id) {
    const tensorflow::Tensor* t = nullptr;
    tensorflow::TensorHandle* tensor_handle;
    TF_ASSERT_OK(eager_service_impl_.GetTensorHandle(
        context_id_, RemoteTensorHandleInternal(2, 0), &tensor_handle));
    TF_ASSERT_OK(tensor_handle->Tensor(&t));
    CheckOutputTensorAndClose(*t);
  }

 protected:
  const string local_device_ = "/job:localhost/replica:0/task:0/device:CPU:0";
  const string remote_device_ = "/job:localhost/replica:0/task:1/device:CPU:0";
  TestEagerServiceImpl eager_service_impl_;
  std::unique_ptr<DeviceMgr> remote_device_mgr_;
  uint64 context_id_;
  tensorflow::FunctionDef fdef_;
  std::unique_ptr<ProcessFunctionLibraryRuntime> eager_pflr_;
  std::unique_ptr<EagerClusterFunctionLibraryRuntime> eager_cluster_flr_;
  FunctionLibraryDefinition func_lib_def_{OpRegistry::Global(), {}};
};

// Test executes a remote function through
// ProcessFunctionLibraryRuntime(EagerClusterFunctionLibraryRuntime).
TEST_F(FunctionWithRemoteInputsTest, EagerPFLRTest) {
  Init();
  // Instantiate MatMulFunction on remote_device.
  FunctionLibraryRuntime::InstantiateOptions options;
  options.target = remote_device_;
  options.is_multi_device_function = true;
  options.input_devices.push_back(local_device_);
  FunctionLibraryRuntime::Handle handle;
  EXPECT_TRUE(MatMulHasAttrWithDefaultValue(fdef_));
  TF_ASSERT_OK(eager_pflr_->Instantiate(
      fdef_.signature().name(), AttrSlice(&fdef_.attr()), options, &handle));
  EagerContext* ctx = nullptr;
  TF_ASSERT_OK(eager_service_impl_.GetEagerContext(context_id_, &ctx));
  for (const string& func_name : ctx->FuncLibDef()->ListFunctionNames()) {
    const FunctionDef* fdef = ctx->FuncLibDef()->Find(func_name);
    EXPECT_TRUE(fdef != nullptr);
    if (absl::StartsWith(func_name, "MatMulFunction")) {
      EXPECT_FALSE(MatMulHasAttrWithDefaultValue(*fdef));
    }
  }
  bool is_cross_process = false;
  TF_CHECK_OK(eager_pflr_->IsCrossProcess(handle, &is_cross_process));
  EXPECT_TRUE(is_cross_process);

  // Run MatMulFunction on remote_device.
  FunctionLibraryRuntime::Options opts;
  const uint64 op_id = 2;
  opts.op_id = op_id;
  Notification done;
  Status status;
  RemoteTensorHandle input;
  input.set_op_id(1);
  input.set_output_num(0);
  input.set_op_device(local_device_);
  input.set_device(local_device_);
  std::vector<RemoteTensorHandle> inputs = {input};
  std::vector<Tensor> outputs;
  gtl::InlinedVector<TensorValue, 4> tensor_args = {TensorValue()};
  TestExecuteNodeArgs args(
      std::move(tensor_args),
      [&inputs](const int i, RemoteTensorHandle* handle) -> Status {
        *handle = inputs.at(i);
        return Status::OK();
      });
  eager_pflr_->Run(opts, handle, args, &outputs,
                   [&status, &done](const Status& s) {
                     status = s;
                     done.Notify();
                   });
  done.WaitForNotification();
  TF_ASSERT_OK(status);
  CheckOutputsAndClose(op_id);
}

// Test executes a remote function with local input and output tensors.
TEST_F(FunctionWithRemoteInputsTest,
       EagerClusterFLRTestWithLocalInputAndOutput) {
  Init();
  // Instantiate MatMulFunction on remote_device.
  FunctionLibraryRuntime::Handle handle;
  EXPECT_TRUE(MatMulHasAttrWithDefaultValue(fdef_));
  Status status;
  Notification instantiate_done;
  eager_cluster_flr_->Instantiate(
      fdef_.signature().name(), func_lib_def_, AttrSlice(&fdef_.attr()),
      FunctionLibraryRuntime::InstantiateOptions(), &handle,
      [&status, &instantiate_done](const Status& s) {
        status = s;
        instantiate_done.Notify();
      });
  instantiate_done.WaitForNotification();
  TF_ASSERT_OK(status);
  EagerContext* ctx = nullptr;
  TF_ASSERT_OK(eager_service_impl_.GetEagerContext(context_id_, &ctx));
  for (const string& func_name : ctx->FuncLibDef()->ListFunctionNames()) {
    const FunctionDef* fdef = ctx->FuncLibDef()->Find(func_name);
    EXPECT_TRUE(fdef != nullptr);
    if (absl::StartsWith(func_name, "MatMulFunction")) {
      EXPECT_FALSE(MatMulHasAttrWithDefaultValue(*fdef));
    }
  }
  const tensorflow::Tensor* input_tensor = nullptr;
  tensorflow::TensorHandle* tensor_handle;
  TF_ASSERT_OK(eager_service_impl_.GetTensorHandle(
      context_id_, RemoteTensorHandleInternal(1, 0), &tensor_handle));
  TF_ASSERT_OK(tensor_handle->Tensor(&input_tensor));

  // Send input_tensor to the remote device, execute MatMulFunction on the
  // remote device, and send the output back.
  FunctionLibraryRuntime::Options opts;
  Notification execute_done;
  std::vector<Tensor> inputs = {*input_tensor};
  std::vector<Tensor> outputs;
  eager_cluster_flr_->Run(opts, handle, inputs, &outputs,
                          [&status, &execute_done](const Status& s) {
                            status = s;
                            execute_done.Notify();
                          });
  execute_done.WaitForNotification();
  TF_ASSERT_OK(status);
  EXPECT_EQ(outputs.size(), 1);
  CheckOutputTensorAndClose(outputs.at(0));
}

// Test executes a remote function through KernelAndDeviceFunc::Run.
TEST_F(FunctionWithRemoteInputsTest, KernelAndDeviceFuncTest) {
  Init();
  Device* local_device;
  TF_ASSERT_OK(device_mgr_->LookupDevice(local_device_, &local_device));
  std::vector<Device*> input_dev_ptrs;
  input_dev_ptrs.push_back(local_device);
  FunctionLibraryRuntime* flr = eager_pflr_->GetFLR(remote_device_);
  EagerContext* ctx = nullptr;
  TF_ASSERT_OK(eager_service_impl_.GetEagerContext(context_id_, &ctx));
  core::RefCountPtr<KernelAndDeviceFunc> kernel = nullptr;
  const int64 op_id = 2;
  kernel.reset(new KernelAndDeviceFunc(
      flr, eager_pflr_.get(), std::move(input_dev_ptrs),
      /*composite_devices=*/{}, /*input_resource_dtypes_and_shapes=*/{},
      /*runner=*/nullptr,
      /*collective_executor=*/nullptr, local_device, fdef_.signature().name(),
      [ctx](const int64 step_id) { return ctx->CreateRendezvous(step_id); },
      [=]() { return op_id; }));

  // Instantiate MatMulFunction on remote_device.
  const NodeDef node_def = MatMulFunctionNodeDef();
  TF_ASSERT_OK(kernel->InstantiateFunc({}, node_def, nullptr));

  // Run MatMulFunction on remote_device.
  gtl::InlinedVector<TensorValue, 4> input_tensors = {TensorValue()};
  RemoteTensorHandle input;
  input.set_op_id(1);
  input.set_output_num(0);
  input.set_op_device(local_device_);
  input.set_device(local_device_);
  std::vector<RemoteTensorHandle> remote_handles = {input};
  TestExecuteNodeArgs inputs(
      std::move(input_tensors),
      [&remote_handles](const int index, RemoteTensorHandle* handle) -> Status {
        *handle = remote_handles.at(index);
        return Status::OK();
      });
  std::vector<Tensor> outputs;

  TF_ASSERT_OK(kernel->Run(/*step_container=*/nullptr, inputs, &outputs,
                           /*cancellation_manager=*/nullptr,
                           /*remote_func_params=*/absl::nullopt));

  CheckOutputsAndClose(op_id);
}

// Test executes a remote function through KernelAndDeviceFunc::RunAsync.
TEST_F(FunctionWithRemoteInputsTest, KernelAndDeviceFuncAsyncTest) {
  Init();
  Device* local_device;
  TF_ASSERT_OK(device_mgr_->LookupDevice(local_device_, &local_device));
  std::vector<Device*> input_dev_ptrs;
  input_dev_ptrs.push_back(local_device);
  FunctionLibraryRuntime* flr = eager_pflr_->GetFLR(remote_device_);
  EagerContext* ctx = nullptr;
  TF_ASSERT_OK(eager_service_impl_.GetEagerContext(context_id_, &ctx));
  core::RefCountPtr<KernelAndDeviceFunc> kernel = nullptr;
  const int64 op_id = 2;
  kernel.reset(new KernelAndDeviceFunc(
      flr, eager_pflr_.get(), std::move(input_dev_ptrs),
      /*composite_devices=*/{}, /*input_resource_dtypes_and_shapes=*/{},
      /*runner=*/nullptr,
      /*collective_executor=*/nullptr, local_device, fdef_.signature().name(),
      [ctx](const int64 step_id) { return ctx->CreateRendezvous(step_id); },
      [=]() { return op_id; }));

  // Instantiate MatMulFunction on remote_device.
  const NodeDef node_def = MatMulFunctionNodeDef();
  TF_ASSERT_OK(kernel->InstantiateFunc({}, node_def, nullptr));

  // Run MatMulFunction on remote_device.
  gtl::InlinedVector<TensorValue, 4> input_tensors = {TensorValue()};
  RemoteTensorHandle input;
  input.set_op_id(1);
  input.set_output_num(0);
  input.set_op_device(local_device_);
  input.set_device(local_device_);
  std::vector<RemoteTensorHandle> remote_handles = {input};
  TestExecuteNodeArgs inputs(
      std::move(input_tensors),
      [&remote_handles](const int index, RemoteTensorHandle* handle) -> Status {
        *handle = remote_handles.at(index);
        return Status::OK();
      });
  std::vector<Tensor> outputs;

  Status status;
  Notification n;
  kernel->RunAsync(/*step_container=*/nullptr, inputs, &outputs,
                   /*cancellation_manager=*/nullptr,
                   /*remote_func_params=*/absl::nullopt,
                   [&status, &n](const Status& s) {
                     status = s;
                     n.Notify();
                   });
  n.WaitForNotification();
  TF_ASSERT_OK(status);
  CheckOutputsAndClose(op_id);
}

// Test creates a context and attempts to send a tensor (using the RPC), and
// then use the tensor.
TEST_F(EagerServiceImplTest, SendTensorTest) {
  TestEagerServiceImpl eager_service_impl(&worker_env_);

  uint64 context_id = random::New64();

  CreateContextRequest request;
  request.mutable_server_def()->set_job_name("localhost");
  request.mutable_server_def()->set_task_index(0);
  request.set_context_id(context_id);
  CreateContextResponse response;

  TF_ASSERT_OK(eager_service_impl.CreateContext(&request, &response));

  EnqueueRequest remote_enqueue_request;
  remote_enqueue_request.set_context_id(context_id);
  EnqueueResponse remote_enqueue_response;

  auto* send_tensor = remote_enqueue_request.add_queue()->mutable_send_tensor();
  send_tensor->set_op_id(1);
  SetTensorProto(send_tensor->add_tensors());

  std::unordered_map<string, AttrValue> attrs;
  AttrValue val;
  val.Clear();
  val.set_type(tensorflow::DataType::DT_FLOAT);
  attrs.insert({"T", val});
  val.Clear();
  val.set_b(false);
  attrs.insert({"transpose_a", val});
  attrs.insert({"transpose_b", val});

  AddOperationToEnqueueRequest(
      2, "MatMul", {std::make_pair(1, 0), std::make_pair(1, 0)}, attrs,
      "/job:localhost/replica:0/task:0/device:CPU:0", &remote_enqueue_request);

  TF_ASSERT_OK(eager_service_impl.Enqueue(&remote_enqueue_request,
                                          &remote_enqueue_response));

  const tensorflow::Tensor* t = nullptr;
  tensorflow::TensorHandle* tensor_handle;
  TF_ASSERT_OK(eager_service_impl.GetTensorHandle(
      context_id, RemoteTensorHandleInternal(2, 0), &tensor_handle));
  TF_ASSERT_OK(tensor_handle->Tensor(&t));

  Device* device = absl::get<Device*>(tensor_handle->device());
  EXPECT_EQ(device, nullptr);

  auto actual = t->flat<float>();
  EXPECT_EQ(4, actual.size());

  EXPECT_EQ(7, actual(0));
  EXPECT_EQ(10, actual(1));
  EXPECT_EQ(15, actual(2));
  EXPECT_EQ(22, actual(3));

  CloseContextRequest close_context_request;
  close_context_request.set_context_id(context_id);
  close_context_request.set_context_view_id(0);
  CloseContextResponse close_context_response;
  TF_ASSERT_OK(eager_service_impl.CloseContext(&close_context_request,
                                               &close_context_response));
}

// Test serializes and sends a pack TensorHandle.
TEST_F(EagerServiceImplTest, SendPackedHandleTest) {
  TestEagerServiceImpl eager_service_impl(&worker_env_);

  const string device0 = "/job:localhost/replica:0/task:0/device:CPU:0";
  const string device1 = "/job:localhost/replica:0/task:1/device:CPU:0";
  const string device2 = "/job:localhost/replica:0/task:2/device:CPU:0";
  const string composite_device =
      "/job:localhost/replica:0/task:0/device:COMPOSITE:0";

  uint64 context_id = random::New64();
  CreateContextRequest request;
  auto* server_def = request.mutable_server_def();
  server_def->set_job_name("localhost");
  server_def->set_task_index(0);
  request.add_cluster_device_attributes()->set_name(device0);
  request.add_cluster_device_attributes()->set_name(device1);
  request.add_cluster_device_attributes()->set_name(device2);
  request.set_context_id(context_id);
  CreateContextResponse response;

  TF_ASSERT_OK(eager_service_impl.CreateContext(&request, &response));

  EnqueueRequest remote_enqueue_request;
  remote_enqueue_request.set_context_id(context_id);
  EnqueueResponse remote_enqueue_response;

  // Copy a tensor to device0
  auto* send_tensor = remote_enqueue_request.add_queue()->mutable_send_tensor();
  send_tensor->set_op_id(1);
  SetTensorProto(send_tensor->add_tensors());

  // Copy a packed handle to device0
  auto* send_packed_handle =
      remote_enqueue_request.add_queue()->mutable_send_packed_handle();
  send_packed_handle->set_op_id(3);
  RemoteTensorHandle* remote_handle =
      send_packed_handle->add_handles()->mutable_remote_handle();
  remote_handle->set_op_id(send_tensor->op_id());
  remote_handle->set_output_num(0);
  remote_handle->set_op_device(device0);
  remote_handle->set_device(device0);

  SendPackedHandleOp::LocalTensorHandle* lcoal_handle =
      send_packed_handle->add_handles()->mutable_local_handle();
  SetTensorProto(lcoal_handle->mutable_tensor());
  lcoal_handle->set_device(device1);

  remote_handle = send_packed_handle->add_handles()->mutable_remote_handle();
  remote_handle->set_op_id(2);
  remote_handle->set_output_num(5);
  remote_handle->set_op_device(device2);
  remote_handle->set_device(device2);

  TF_ASSERT_OK(eager_service_impl.Enqueue(&remote_enqueue_request,
                                          &remote_enqueue_response));

  tensorflow::TensorHandle* packed_handle;
  TF_ASSERT_OK(eager_service_impl.GetTensorHandle(
      context_id, RemoteTensorHandleInternal(3, 0), &packed_handle));

  EXPECT_EQ(packed_handle->Type(), TensorHandle::PACKED);
  EXPECT_EQ(packed_handle->NumPackedHandles(), 3);
  EXPECT_EQ(absl::get<Device*>(packed_handle->device())->name(),
            composite_device);

  TensorHandle* handle0 = nullptr;
  TF_ASSERT_OK(packed_handle->ExtractPackedHandle(0, &handle0));
  EXPECT_EQ(handle0->Type(), TensorHandle::LOCAL);
  EXPECT_EQ(handle0->op_device()->name(), device0);
  const Tensor* t0 = nullptr;
  TF_ASSERT_OK(handle0->Tensor(&t0));
  auto actual = t0->flat<float>();
  EXPECT_EQ(4, actual.size());
  EXPECT_EQ(1.0, actual(0));
  EXPECT_EQ(2.0, actual(1));
  EXPECT_EQ(3.0, actual(2));
  EXPECT_EQ(4.0, actual(3));

  TensorHandle* handle1 = nullptr;
  TF_ASSERT_OK(packed_handle->ExtractPackedHandle(1, &handle1));
  EXPECT_EQ(handle1->Type(), TensorHandle::LOCAL);
  EXPECT_EQ(handle1->op_device()->name(), device1);
  const Tensor* t1 = nullptr;
  TF_ASSERT_OK(handle0->Tensor(&t1));
  EXPECT_EQ(t1, t0);

  TensorHandle* handle2 = nullptr;
  TF_ASSERT_OK(packed_handle->ExtractPackedHandle(2, &handle2));
  EXPECT_EQ(handle2->Type(), TensorHandle::REMOTE);
  EXPECT_EQ(handle2->op_device()->name(), device2);
  int64 op_id;
  int32 output_num;
  TF_ASSERT_OK(handle2->RemoteAddress(absl::get<Device*>(handle2->device()),
                                      /*wait_until_ready=*/true, &op_id,
                                      &output_num));
  EXPECT_EQ(op_id, 2);
  EXPECT_EQ(output_num, 5);

  CloseContextRequest close_context_request;
  close_context_request.set_context_id(context_id);
  close_context_request.set_context_view_id(0);
  CloseContextResponse close_context_response;
  TF_ASSERT_OK(eager_service_impl.CloseContext(&close_context_request,
                                               &close_context_response));
}

// Test requests sent to the eager service on master.
TEST_F(EagerServiceImplTest, RequestsToMasterTest) {
  tensorflow::Rendezvous* rendezvous =
      new tensorflow::IntraProcessRendezvous(device_mgr_.get());
  // Create a master eager context.
  tensorflow::EagerContext* ctx = new tensorflow::EagerContext(
      SessionOptions(),
      tensorflow::ContextDevicePlacementPolicy::DEVICE_PLACEMENT_SILENT,
      tensorflow::ContextMirroringPolicy::MIRRORING_NONE, /*async=*/false,
      /*lazy_copy_function_remote_inputs=*/false, device_mgr_.get(), false,
      rendezvous, GetDefaultCustomKernelCreator());
  const uint64 context_id = random::New64();

  // Set RemoteMgr to ctx.
  auto remote_mgr =
      absl::make_unique<tensorflow::eager::RemoteMgr>(/*is_master=*/true, ctx);
  TF_ASSERT_OK(ctx->InitializeRemoteWorker(
      /*remote_eager_workers=*/nullptr, /*remote_device_mgr=*/nullptr,
      /*remote_contexts=*/{}, context_id, /*context_view_id=*/0,
      /*rendezvous_creator=*/nullptr,
      /*cluster_flr=*/nullptr, std::move(remote_mgr),
      /*resource_deallocator=*/nullptr));

  TestEagerServiceImpl eager_service_impl(&worker_env_);

  EnqueueRequest remote_enqueue_request;
  remote_enqueue_request.set_context_id(context_id);
  EnqueueResponse remote_enqueue_response;

  auto* send_tensor = remote_enqueue_request.add_queue()->mutable_send_tensor();
  send_tensor->set_op_id(1);
  SetTensorProto(send_tensor->add_tensors());

  // Unable to handle the request since there is no eager context.
  Status status = eager_service_impl.Enqueue(&remote_enqueue_request,
                                             &remote_enqueue_response);
  EXPECT_EQ(error::INVALID_ARGUMENT, status.code());
  EXPECT_TRUE(absl::StrContains(
      status.error_message(),
      "Unable to find a context_id matching the specified one"));

  // The request can be handled after adding the master eager context to
  // service.
  TF_ASSERT_OK(eager_service_impl.CreateMasterContext(context_id, ctx));
  TF_ASSERT_OK(eager_service_impl.Enqueue(&remote_enqueue_request,
                                          &remote_enqueue_response));
  ctx->Unref();
}

TEST_F(EagerServiceImplTest, KeepAliveTest) {
  TestEagerServiceImpl eager_service_impl(&worker_env_);

  uint64 context_id = random::New64();
  CreateContextRequest request;
  request.mutable_server_def()->set_job_name("localhost");
  request.mutable_server_def()->set_task_index(0);
  request.set_context_id(context_id);
  request.set_keep_alive_secs(3);
  CreateContextResponse response;

  TF_ASSERT_OK(eager_service_impl.CreateContext(&request, &response));

  worker_env_.env->SleepForMicroseconds(5 *
                                        tensorflow::EnvTime::kSecondsToMicros);

  KeepAliveRequest keep_alive_request;
  KeepAliveResponse keep_alive_response;

  keep_alive_request.set_context_id(context_id);

  Status status =
      eager_service_impl.KeepAlive(&keep_alive_request, &keep_alive_response);

  EXPECT_EQ(status.code(), error::INVALID_ARGUMENT);
  EXPECT_PRED_FORMAT2(::testing::IsSubstring, "Unable to find a context_id",
                      status.error_message());

  uint64 new_context_id = random::New64();
  // Create a new context.
  request.set_context_id(new_context_id);
  TF_ASSERT_OK(eager_service_impl.CreateContext(&request, &response));

  // The context should not be GC'd.
  worker_env_.env->SleepForMicroseconds(1 *
                                        tensorflow::EnvTime::kSecondsToMicros);

  keep_alive_request.set_context_id(new_context_id);

  TF_ASSERT_OK(
      eager_service_impl.KeepAlive(&keep_alive_request, &keep_alive_response));
}

}  // namespace
}  // namespace eager
}  // namespace tensorflow
