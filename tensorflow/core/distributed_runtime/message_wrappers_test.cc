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

#include "tensorflow/core/distributed_runtime/message_wrappers.h"

#include "tensorflow/core/framework/cost_graph.pb.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

static Tensor TensorA() {
  Tensor a_tensor(DT_INT32, TensorShape({2, 2}));
  test::FillValues<int32>(&a_tensor, {3, 2, -1, 0});
  return a_tensor;
}

static Tensor TensorB() {
  Tensor b_tensor(DT_INT32, TensorShape({1, 2}));
  test::FillValues<int32>(&b_tensor, {1, 2});
  return b_tensor;
}

static void BuildRunStepRequest(MutableRunStepRequestWrapper* request) {
  request->set_session_handle("handle");
  request->set_partial_run_handle("partial_handle");
  request->add_feed("feed_a:0", TensorA());
  request->add_feed("feed_b:0", TensorB());
  request->add_fetch("fetch_x:0");
  request->add_fetch("fetch_y:0");
  request->add_target("target_i");
  request->add_target("target_j");
  request->mutable_options()->set_timeout_in_ms(37);
}

static void CheckRunStepRequest(const RunStepRequestWrapper& request) {
  EXPECT_EQ("handle", request.session_handle());
  EXPECT_EQ("partial_handle", request.partial_run_handle());
  EXPECT_EQ(2, request.num_feeds());
  EXPECT_EQ("feed_a:0", request.feed_name(0));
  EXPECT_EQ("feed_b:0", request.feed_name(1));
  Tensor val;
  TF_EXPECT_OK(request.FeedValue(0, &val));
  test::ExpectTensorEqual<int32>(TensorA(), val);
  TF_EXPECT_OK(request.FeedValue(1, &val));
  test::ExpectTensorEqual<int32>(TensorB(), val);

  EXPECT_EQ(2, request.num_fetches());
  EXPECT_EQ("fetch_x:0", request.fetch_name(0));
  EXPECT_EQ("fetch_y:0", request.fetch_name(1));
  EXPECT_EQ("target_i", request.target_name(0));
  EXPECT_EQ("target_j", request.target_name(1));
  EXPECT_EQ(37, request.options().timeout_in_ms());
}

static void BuildRunGraphRequest(
    const RunStepRequestWrapper& run_step_request,
    MutableRunGraphRequestWrapper* run_graph_request) {
  run_graph_request->set_graph_handle("graph_handle");
  run_graph_request->set_step_id(13);
  run_graph_request->mutable_exec_opts()->set_record_timeline(true);
  TF_EXPECT_OK(run_graph_request->AddSendFromRunStepRequest(run_step_request, 0,
                                                            "send_0"));
  TF_EXPECT_OK(run_graph_request->AddSendFromRunStepRequest(run_step_request, 1,
                                                            "send_1"));
  run_graph_request->add_recv_key("recv_2");
  run_graph_request->add_recv_key("recv_3");
  run_graph_request->set_is_partial(true);
}

static void CheckRunGraphRequest(const RunGraphRequestWrapper& request) {
  EXPECT_EQ("graph_handle", request.graph_handle());
  EXPECT_EQ(13, request.step_id());
  EXPECT_FALSE(request.exec_opts().record_costs());
  EXPECT_TRUE(request.exec_opts().record_timeline());
  EXPECT_EQ(2, request.num_sends());
  Tensor val;
  TF_EXPECT_OK(request.SendValue(0, &val));
  test::ExpectTensorEqual<int32>(TensorA(), val);
  TF_EXPECT_OK(request.SendValue(1, &val));
  test::ExpectTensorEqual<int32>(TensorB(), val);
  EXPECT_TRUE(request.is_partial());
  EXPECT_FALSE(request.is_last_partial_run());
}

static void BuildRunGraphResponse(
    MutableRunGraphResponseWrapper* run_graph_response) {
  run_graph_response->AddRecv("recv_2", TensorA());
  run_graph_response->AddRecv("recv_3", TensorB());
  run_graph_response->mutable_step_stats()->add_dev_stats()->set_device(
      "/cpu:0");
  run_graph_response->mutable_cost_graph()->add_node()->set_name("cost_node");
}

static void CheckRunGraphResponse(MutableRunGraphResponseWrapper* response) {
  EXPECT_EQ(2, response->num_recvs());
  EXPECT_EQ("recv_2", response->recv_key(0));
  EXPECT_EQ("recv_3", response->recv_key(1));
  Tensor val;
  TF_EXPECT_OK(response->RecvValue(0, &val));
  test::ExpectTensorEqual<int32>(TensorA(), val);
  TF_EXPECT_OK(response->RecvValue(1, &val));
  test::ExpectTensorEqual<int32>(TensorB(), val);
  EXPECT_EQ(1, response->mutable_step_stats()->dev_stats_size());
  EXPECT_EQ("/cpu:0", response->mutable_step_stats()->dev_stats(0).device());
  EXPECT_EQ(1, response->mutable_cost_graph()->node_size());
  EXPECT_EQ("cost_node", response->mutable_cost_graph()->node(0).name());
}

static void BuildRunStepResponse(
    MutableRunGraphResponseWrapper* run_graph_response,
    MutableRunStepResponseWrapper* run_step_response) {
  TF_EXPECT_OK(run_step_response->AddTensorFromRunGraphResponse(
      "fetch_x:0", run_graph_response, 0));
  TF_EXPECT_OK(run_step_response->AddTensorFromRunGraphResponse(
      "fetch_y:0", run_graph_response, 1));
  *run_step_response->mutable_metadata()->mutable_step_stats() =
      *run_graph_response->mutable_step_stats();
}

static void CheckRunStepResponse(
    const MutableRunStepResponseWrapper& response) {
  EXPECT_EQ(2, response.num_tensors());
  EXPECT_EQ("fetch_x:0", response.tensor_name(0));
  EXPECT_EQ("fetch_y:0", response.tensor_name(1));
  Tensor val;
  TF_EXPECT_OK(response.TensorValue(0, &val));
  test::ExpectTensorEqual<int32>(TensorA(), val);
  TF_EXPECT_OK(response.TensorValue(1, &val));
  test::ExpectTensorEqual<int32>(TensorB(), val);
  EXPECT_EQ(1, response.metadata().step_stats().dev_stats_size());
  EXPECT_EQ("/cpu:0", response.metadata().step_stats().dev_stats(0).device());
}

TEST(MessageWrappers, RunStepRequest_Basic) {
  InMemoryRunStepRequest in_memory_request;
  BuildRunStepRequest(&in_memory_request);
  CheckRunStepRequest(in_memory_request);

  MutableProtoRunStepRequest proto_request;
  BuildRunStepRequest(&proto_request);
  CheckRunStepRequest(proto_request);

  CheckRunStepRequest(ProtoRunStepRequest(&in_memory_request.ToProto()));
  CheckRunStepRequest(ProtoRunStepRequest(&proto_request.ToProto()));
}

TEST(MessageWrappers, RunGraphRequest_Basic) {
  InMemoryRunStepRequest in_memory_run_step_request;
  BuildRunStepRequest(&in_memory_run_step_request);

  MutableProtoRunStepRequest mutable_proto_run_step_request;
  BuildRunStepRequest(&mutable_proto_run_step_request);

  ProtoRunStepRequest proto_run_step_request(
      &mutable_proto_run_step_request.ToProto());

  // Client -(in memory)-> Master -(in memory)-> Worker.
  {
    InMemoryRunGraphRequest request;
    BuildRunGraphRequest(in_memory_run_step_request, &request);
    CheckRunGraphRequest(request);
    CheckRunGraphRequest(ProtoRunGraphRequest(&request.ToProto()));
  }

  // Client -(mutable proto)-> Master -(in memory)-> Worker.
  {
    InMemoryRunGraphRequest request;
    BuildRunGraphRequest(mutable_proto_run_step_request, &request);
    CheckRunGraphRequest(request);
    CheckRunGraphRequest(ProtoRunGraphRequest(&request.ToProto()));
  }

  // Client -(proto)-> Master -(in memory)-> Worker.
  {
    InMemoryRunGraphRequest request;
    BuildRunGraphRequest(proto_run_step_request, &request);
    CheckRunGraphRequest(request);
    CheckRunGraphRequest(ProtoRunGraphRequest(&request.ToProto()));
  }

  // Client -(in memory)-> Master -(mutable proto)-> Worker.
  {
    MutableProtoRunGraphRequest request;
    BuildRunGraphRequest(in_memory_run_step_request, &request);
    CheckRunGraphRequest(request);
    CheckRunGraphRequest(ProtoRunGraphRequest(&request.ToProto()));
  }

  // Client -(mutable proto)-> Master -(mutable proto)-> Worker.
  {
    MutableProtoRunGraphRequest request;
    BuildRunGraphRequest(mutable_proto_run_step_request, &request);
    CheckRunGraphRequest(request);
    CheckRunGraphRequest(ProtoRunGraphRequest(&request.ToProto()));
  }

  // Client -(proto)-> Master -(mutable proto)-> Worker.
  {
    MutableProtoRunGraphRequest request;
    BuildRunGraphRequest(proto_run_step_request, &request);
    CheckRunGraphRequest(request);
    CheckRunGraphRequest(ProtoRunGraphRequest(&request.ToProto()));
  }
}

TEST(MessageWrappers, RunGraphResponse_Basic) {
  InMemoryRunGraphResponse in_memory_response;
  BuildRunGraphResponse(&in_memory_response);
  CheckRunGraphResponse(&in_memory_response);

  OwnedProtoRunGraphResponse owned_proto_response;
  BuildRunGraphResponse(&owned_proto_response);
  CheckRunGraphResponse(&owned_proto_response);

  RunGraphResponse response_proto;
  NonOwnedProtoRunGraphResponse non_owned_proto_response(&response_proto);
  BuildRunGraphResponse(&non_owned_proto_response);
  CheckRunGraphResponse(&non_owned_proto_response);
}

TEST(MessageWrappers, RunStepResponse_Basic) {
  {
    // Worker -(in memory)-> Master -(in memory)-> Client.
    InMemoryRunGraphResponse run_graph_response;
    BuildRunGraphResponse(&run_graph_response);
    InMemoryRunStepResponse response;
    BuildRunStepResponse(&run_graph_response, &response);
    CheckRunStepResponse(response);
  }

  {
    // Worker -(in memory)-> Master -(owned proto)-> Client.
    InMemoryRunGraphResponse run_graph_response;
    BuildRunGraphResponse(&run_graph_response);
    OwnedProtoRunStepResponse response;
    BuildRunStepResponse(&run_graph_response, &response);
    CheckRunStepResponse(response);
  }

  {
    // Worker -(in memory)-> Master -(non-owned proto)-> Client.
    InMemoryRunGraphResponse run_graph_response;
    BuildRunGraphResponse(&run_graph_response);
    RunStepResponse response_proto;
    NonOwnedProtoRunStepResponse response(&response_proto);
    BuildRunStepResponse(&run_graph_response, &response);
    CheckRunStepResponse(response);
  }

  {
    // Worker -(owned proto)-> Master -(in memory)-> Client.
    OwnedProtoRunGraphResponse run_graph_response;
    BuildRunGraphResponse(&run_graph_response);
    InMemoryRunStepResponse response;
    BuildRunStepResponse(&run_graph_response, &response);
    CheckRunStepResponse(response);
  }

  {
    // Worker -(owned proto)-> Master -(owned proto)-> Client.
    OwnedProtoRunGraphResponse run_graph_response;
    BuildRunGraphResponse(&run_graph_response);
    OwnedProtoRunStepResponse response;
    BuildRunStepResponse(&run_graph_response, &response);
    CheckRunStepResponse(response);
  }

  {
    // Worker -(owned proto)-> Master -(non-owned proto)-> Client.
    OwnedProtoRunGraphResponse run_graph_response;
    BuildRunGraphResponse(&run_graph_response);
    RunStepResponse response_proto;
    NonOwnedProtoRunStepResponse response(&response_proto);
    BuildRunStepResponse(&run_graph_response, &response);
    CheckRunStepResponse(response);
  }

  {
    // Worker -(non-owned proto)-> Master -(in memory)-> Client.
    RunGraphResponse run_graph_response_proto;
    NonOwnedProtoRunGraphResponse run_graph_response(&run_graph_response_proto);
    BuildRunGraphResponse(&run_graph_response);
    InMemoryRunStepResponse response;
    BuildRunStepResponse(&run_graph_response, &response);
    CheckRunStepResponse(response);
  }

  {
    // Worker -(non-owned proto)-> Master -(owned proto)-> Client.
    RunGraphResponse run_graph_response_proto;
    NonOwnedProtoRunGraphResponse run_graph_response(&run_graph_response_proto);
    BuildRunGraphResponse(&run_graph_response);
    OwnedProtoRunStepResponse response;
    BuildRunStepResponse(&run_graph_response, &response);
    CheckRunStepResponse(response);
  }

  {
    // Worker -(non-owned proto)-> Master -(non-owned proto)-> Client.
    RunGraphResponse run_graph_response_proto;
    NonOwnedProtoRunGraphResponse run_graph_response(&run_graph_response_proto);
    BuildRunGraphResponse(&run_graph_response);
    RunStepResponse response_proto;
    NonOwnedProtoRunStepResponse response(&response_proto);
    BuildRunStepResponse(&run_graph_response, &response);
    CheckRunStepResponse(response);
  }
}

}  // namespace tensorflow
