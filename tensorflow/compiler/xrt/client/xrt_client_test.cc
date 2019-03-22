/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xrt/client/xrt_client.h"

#include <memory>

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xrt/client/xrt_grpc_eager_client.h"
#include "tensorflow/compiler/xrt/client/xrt_tf_client.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_channel.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_session.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_testlib.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/cluster.pb.h"
#include "tensorflow/core/protobuf/eager_service.pb.h"
#include "tensorflow/core/protobuf/tensorflow_server.pb.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

class XrtClientTest : public ::testing::Test {
 protected:
  XrtClientTest() {
    string binary_path =
        absl::StrCat(testing::TensorFlowSrcRoot(),
                     "/compiler/xrt/client/xrt_testlib_server");

    TF_CHECK_OK(test::TestCluster::MakeTestCluster(
        binary_path, SessionOptions(), /*n=*/1, &cluster_));

    CHECK_EQ(cluster_->targets().size(), 1);
    JobDef* job = cluster_def_.add_job();
    job->set_name("localhost");
    (*job->mutable_tasks())[0] = cluster_->targets()[0];
  }

  xla::StatusOr<std::shared_ptr<XrtContext>> MakeContext();

  std::unique_ptr<test::TestCluster> cluster_;
  ClusterDef cluster_def_;
};

// Test some connection basics using XrtGrpcEagerClient directly.
TEST_F(XrtClientTest, XrtGrpcEagerClientWorks) {
  ChannelCreationFunction channel_func =
      ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<GrpcChannelCache> channel_cache,
                          GetGrpcChannelCache(cluster_def_, channel_func));
  XrtGrpcEagerClientCache client_cache(channel_cache);

  TF_ASSERT_OK_AND_ASSIGN(
      XrtGrpcEagerClient * client,
      client_cache.GetClient("/job:localhost/task:0/replica:0"));

  // Create and destroy a context to verify we can make RPCs.
  eager::CreateContextRequest request;
  ServerDef* server_def = request.mutable_server_def();
  *server_def->mutable_cluster() = cluster_def_;
  server_def->set_job_name("localhost");
  server_def->set_protocol("grpc");
  request.set_keep_alive_secs(60);
  request.set_rendezvous_id(random::New64());

  eager::CreateContextResponse create_response;
  TF_ASSERT_OK(client->SyncCall(&XrtGrpcEagerClient::CreateContextAsync,
                                &request, &create_response));

  eager::CloseContextRequest close_request;
  close_request.set_context_id(create_response.context_id());

  eager::CloseContextResponse close_response;
  TF_ASSERT_OK(client->SyncCall(&XrtGrpcEagerClient::CloseContextAsync,
                                &close_request, &close_response));
}

// Tests that we can connect to a server using the higher-level XrtTfClient API,
// transfer tensors to the device, run an Add operator, and retrieve the result.
TEST_F(XrtClientTest, XrtTfClientWorks) {
  ChannelCreationFunction channel_func =
      ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<GrpcChannelCache> channel_cache,
                          GetGrpcChannelCache(cluster_def_, channel_func));

  auto client = std::make_shared<XrtTfClient>(cluster_def_, channel_cache);
  TF_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<XrtTfContext> context,
      XrtTfContext::Create(XrtTfContext::Options(), client, /*job=*/"localhost",
                           /*task=*/0));

  auto a_proto = absl::make_unique<TensorProto>();
  a_proto->set_dtype(DT_INT32);
  a_proto->add_int_val(47);
  XrtTensorHandle a =
      context->SendTensor(std::move(a_proto), context->cpu_device_id());
  auto b_proto = absl::make_unique<TensorProto>();
  b_proto->set_dtype(DT_INT32);
  b_proto->mutable_tensor_shape()->add_dim()->set_size(2);
  b_proto->add_int_val(-101);
  b_proto->add_int_val(3);
  XrtTensorHandle b =
      context->SendTensor(std::move(b_proto), context->cpu_device_id());

  protobuf::Map<string, AttrValue> attrs;
  attrs["T"] = MakeAttrValue(DT_INT32);
  std::vector<XrtTensorHandle> add_outputs = context->EnqueueOp(
      "Add", {&a, &b}, /*output_arity=*/1, attrs, context->cpu_device_id());
  ASSERT_EQ(add_outputs.size(), 1);

  std::shared_ptr<XrtRecvTensorFuture> future =
      context->RecvTensor(add_outputs[0], DT_INT32, /*host_memory=*/false);

  TF_ASSERT_OK_AND_ASSIGN(RecvTensorResponse * response, future->Get());
  const TensorProto& out_proto = response->tensor();
  EXPECT_EQ(out_proto.dtype(), DT_INT32);

  ASSERT_EQ(out_proto.tensor_content().size(), sizeof(int32) * 2);
  std::vector<int32> out(2);
  out_proto.tensor_content().CopyToArray(reinterpret_cast<char*>(out.data()));
  // TODO(phawkins): handle endian conversion.
  EXPECT_EQ(out[0], -54);
  EXPECT_EQ(out[1], 50);
}

xla::StatusOr<std::shared_ptr<XrtContext>> XrtClientTest::MakeContext() {
  ChannelCreationFunction channel_func =
      ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
  TF_ASSIGN_OR_RETURN(std::shared_ptr<GrpcChannelCache> channel_cache,
                      GetGrpcChannelCache(cluster_def_, channel_func));

  auto client = std::make_shared<XrtTfClient>(cluster_def_, channel_cache);
  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<XrtTfContext> tf_context,
      XrtTfContext::Create(XrtTfContext::Options(), client, /*job=*/"localhost",
                           /*task=*/0));

  TF_ASSIGN_OR_RETURN(auto context, XrtContext::Create(tf_context, "XLA_CPU"));

  // There should be exactly one XLA_CPU device.
  TF_RET_CHECK(context->device_count() == 1);
  return context;
}

// Tests that we can use the XRT client to perform some simple operations.
TEST_F(XrtClientTest, XrtClientWorks) {
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<XrtContext> context, MakeContext());

  ASSERT_TRUE(context->tf_context() != nullptr);

  EXPECT_EQ(context->tf_device_ids().size(), 1);

  ASSERT_EQ(context->device_mesh_coordinates().size(), 1);
  ASSERT_EQ(context->device_mesh_coordinates()[0].value_size(), 1);
  EXPECT_EQ(context->device_mesh_coordinates()[0].value(0), 0);

  // Tests sending a literal to and from the device.
  xla::Shape shape = xla::ShapeUtil::MakeShape(xla::F32, {3, 4, 5});
  TF_ASSERT_OK_AND_ASSIGN(xla::Literal a,
                          xla::LiteralUtil::CreateRandomLiteral<xla::F32>(
                              shape,
                              /*mean=*/7.0, /*stddev=*/13.5));
  TF_ASSERT_OK_AND_ASSIGN(auto buffer, XrtBuffer::FromLiteral(context, 0, a));
  TF_ASSERT_OK_AND_ASSIGN(xla::Literal b, buffer->ToLiteral());
  EXPECT_TRUE(xla::LiteralTestUtil::Equal(a, b));

  // Run a simple computation, fetch its output, and check it is what we expect.
  auto build_computation = [&]() {
    xla::XlaBuilder builder("test_computation");
    xla::XlaOp p = xla::Parameter(&builder, 0, shape, "param");
    xla::Add(p, p);
    return builder.Build();
  };
  TF_ASSERT_OK_AND_ASSIGN(xla::XlaComputation computation, build_computation());

  TF_ASSERT_OK_AND_ASSIGN(xla::DeviceAssignment assignment,
                          xla::ComputationPlacer().AssignDevices(1, 1));
  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          XrtExecutable::Compile(context, computation.proto(),
                                                 {shape}, shape, assignment));
  EXPECT_EQ(executable->device_assignment(), assignment);
  TF_ASSERT_OK_AND_ASSIGN(auto c_buffer, executable->Execute({buffer}));

  xla::Literal expected = a.Clone();
  for (float& elem : expected.data<float>()) {
    elem *= 2;
  }
  TF_ASSERT_OK_AND_ASSIGN(xla::Literal out, c_buffer->ToLiteral());
  EXPECT_TRUE(xla::LiteralTestUtil::Equal(expected, out));

  // Explicitly delete the executable, and then compile and run a different
  // computation.
  executable->Delete();

  auto build_sub_computation = [&]() {
    xla::XlaBuilder builder("test_computation");
    xla::XlaOp p = xla::Parameter(&builder, 0, shape, "p");
    xla::XlaOp q = xla::Parameter(&builder, 1, shape, "q");
    xla::Sub(p, q);
    return builder.Build();
  };
  TF_ASSERT_OK_AND_ASSIGN(xla::XlaComputation sub_computation,
                          build_sub_computation());

  TF_ASSERT_OK_AND_ASSIGN(
      auto sub_executable,
      XrtExecutable::Compile(context, sub_computation.proto(), {shape, shape},
                             shape, assignment));
  TF_ASSERT_OK_AND_ASSIGN(auto buffer_out,
                          sub_executable->Execute({c_buffer, buffer}));
  TF_ASSERT_OK_AND_ASSIGN(xla::Literal sub_out, buffer_out->ToLiteral());
  EXPECT_TRUE(xla::LiteralTestUtil::Equal(a, sub_out));
}

TEST_F(XrtClientTest, ErrorsPropagateCorrectly) {
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<XrtContext> context, MakeContext());
  xla::Shape shape = xla::ShapeUtil::MakeShape(xla::F32, {3, 4, 5});
  TF_ASSERT_OK_AND_ASSIGN(xla::Literal a,
                          xla::LiteralUtil::CreateRandomLiteral<xla::F32>(
                              shape,
                              /*mean=*/7.0, /*stddev=*/13.5));
  TF_ASSERT_OK_AND_ASSIGN(auto buffer, XrtBuffer::FromLiteral(context, 0, a));

  auto build_computation = [&]() {
    xla::XlaBuilder builder("test_computation");
    xla::XlaOp p = xla::Parameter(&builder, 0, shape, "p");
    xla::XlaOp q = xla::Parameter(&builder, 1, shape, "q");
    xla::Add(p, q);
    return builder.Build();
  };
  TF_ASSERT_OK_AND_ASSIGN(xla::XlaComputation computation, build_computation());

  TF_ASSERT_OK_AND_ASSIGN(xla::DeviceAssignment assignment,
                          xla::ComputationPlacer().AssignDevices(1, 1));
  // Call Compile() with an arity mismatch.
  TF_ASSERT_OK_AND_ASSIGN(auto sub_executable,
                          XrtExecutable::Compile(context, computation.proto(),
                                                 {shape}, shape, assignment));
  TF_ASSERT_OK_AND_ASSIGN(auto buffer_out, sub_executable->Execute({buffer}));

  // The compilation error should be reported when we consumer the computation's
  // output.
  EXPECT_FALSE(buffer_out->ToLiteral().ok());

  // Further, we expect a clean shutdown at this point.
  context = nullptr;
}

TEST_F(XrtClientTest, TupleDestructuringAndDelete) {
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<XrtContext> context, MakeContext());

  // Tests sending a literal to and from the device.
  xla::Shape a_shape = xla::ShapeUtil::MakeShape(xla::F32, {3, 4, 5});
  TF_ASSERT_OK_AND_ASSIGN(xla::Literal a,
                          xla::LiteralUtil::CreateRandomLiteral<xla::F32>(
                              a_shape,
                              /*mean=*/7.0, /*stddev=*/13.5));

  xla::Shape b_shape = xla::ShapeUtil::MakeShape(xla::F64, {2, 7});
  TF_ASSERT_OK_AND_ASSIGN(xla::Literal b,
                          xla::LiteralUtil::CreateRandomLiteral<xla::F64>(
                              b_shape,
                              /*mean=*/3.15, /*stddev=*/-2.1));
  xla::Literal tuple = xla::LiteralUtil::MakeTuple({&a, &b});
  TF_ASSERT_OK_AND_ASSIGN(auto buffer,
                          XrtBuffer::FromLiteral(context, 0, tuple));

  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::shared_ptr<XrtBuffer>> pieces,
                          buffer->DestructureTuple());

  // Explicitly delete the tuple, which should have no effect on its
  // constituents.
  buffer->Delete();

  TF_ASSERT_OK_AND_ASSIGN(xla::Literal a_out, pieces[0]->ToLiteral());
  TF_ASSERT_OK_AND_ASSIGN(xla::Literal b_out, pieces[1]->ToLiteral());
  EXPECT_TRUE(xla::LiteralTestUtil::Equal(a, a_out));
  EXPECT_TRUE(xla::LiteralTestUtil::Equal(b, b_out));

  // Explicitly delete one of the pieces, use RAII to delete the other.
  pieces[1]->Delete();
}

}  // namespace tensorflow
