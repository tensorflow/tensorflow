/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/distributed_runtime/master.h"

#include <map>
#include <memory>

#include "grpcpp/grpcpp.h"
#include "Eigen/Core"  // from @eigen_archive
#include "tensorflow/core/distributed_runtime/rpc/grpc_channel.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_master_service_impl.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_testlib.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/master.pb.h"

namespace tensorflow {

class MasterTest : public ::testing::Test {
 protected:
  MasterTest() {
    std::vector<string> targets;
    SessionOptions options;
    (*options.config.mutable_device_count())["CPU"] = 1;
    (*options.config.mutable_device_count())["GPU"] = 0;
    TF_CHECK_OK(test::TestCluster::MakeTestCluster(
        test::TestClusterConfig().Options(options).Jobs(
            {test::TestJob{/*job_name=*/"localhost", /*num_tasks=*/2}}),
        &cluster_));
    SharedGrpcChannelPtr channel_ptr;
    TF_CHECK_OK(NewHostPortGrpcChannel(
        cluster_->targets()[0], &options.config.rpc_options(), &channel_ptr));
    master_ = grpc::MasterService::NewStub(channel_ptr);
  }

  std::unique_ptr<test::TestCluster> cluster_;
  std::unique_ptr<grpc::MasterService::Stub> master_;

  // Helpers for MasterService.{CreateSession,RunStep,CloseSession}
  // rpc calls.

  absl::Status CreateSession(const GraphDef& def, string* handle,
                             int64_t* initial_version) {
    ::grpc::ClientContext ctx;
    CreateSessionRequest req;
    *(req.mutable_graph_def()) = def;
    // Invokes placement frequently.
    req.mutable_config()->set_placement_period(1);
    CreateSessionResponse resp;
    const absl::Status s =
        FromGrpcStatus(master_->CreateSession(&ctx, req, &resp));
    if (s.ok()) {
      *handle = resp.session_handle();
      *initial_version = resp.graph_version();
    }
    return s;
  }

  absl::Status ExtendSession(const string& handle, const GraphDef& def,
                             int64_t current_version, int64_t* new_version) {
    ::grpc::ClientContext ctx;
    ExtendSessionRequest req;
    req.set_session_handle(handle);
    *(req.mutable_graph_def()) = def;
    req.set_current_graph_version(current_version);
    ExtendSessionResponse resp;
    const absl::Status s =
        FromGrpcStatus(master_->ExtendSession(&ctx, req, &resp));
    if (s.ok()) {
      *new_version = resp.new_graph_version();
    }
    return s;
  }

  absl::Status RunStep(
      const string& handle,
      const std::vector<std::pair<string, const Tensor*> >& feed,
      const std::map<string, Tensor*>& fetch) {
    ::grpc::ClientContext ctx;
    RunStepRequest req;
    req.set_session_handle(handle);
    for (const auto& p : feed) {
      const string& feed_name = p.first;
      const Tensor* feed_tensor = p.second;
      auto f = req.add_feed();
      f->set_name(feed_name);
      feed_tensor->AsProtoTensorContent(f->mutable_tensor());
    }
    for (const auto& p : fetch) {
      const string& fetch_name = p.first;
      req.add_fetch(fetch_name);
    }
    RunStepResponse resp;
    const absl::Status s = FromGrpcStatus(master_->RunStep(&ctx, req, &resp));
    if (s.ok()) {
      for (const auto& fetch_resp : resp.tensor()) {
        auto it = fetch.find(fetch_resp.name());
        CHECK(it != fetch.end());
        CHECK(it->second->FromProto(fetch_resp.tensor()));
      }
    }
    return s;
  }

  absl::Status CloseSession(const string& handle) {
    ::grpc::ClientContext ctx;
    CloseSessionRequest req;
    req.set_session_handle(handle);
    CloseSessionResponse resp;
    return FromGrpcStatus(master_->CloseSession(&ctx, req, &resp));
  }

  absl::Status Reset() {
    ::grpc::ClientContext ctx;
    ResetRequest req;
    ResetResponse resp;
    return FromGrpcStatus(master_->Reset(&ctx, req, &resp));
  }
};

TEST_F(MasterTest, CreateClose) {
  GraphDef def;  // Empty.
  string handle;
  int64_t initial_version;
  TF_ASSERT_OK(CreateSession(def, &handle, &initial_version));
  EXPECT_TRUE(absl::IsAborted(CloseSession("randombits")));
  EXPECT_TRUE(CloseSession(handle).ok());
}

TEST_F(MasterTest, ListDevices) {
  ::grpc::ClientContext ctx;
  ListDevicesRequest req;
  ListDevicesResponse resp;
  const absl::Status s = FromGrpcStatus(master_->ListDevices(&ctx, req, &resp));
  TF_EXPECT_OK(s);
  EXPECT_EQ(1, resp.local_device_size());
  EXPECT_EQ("CPU", resp.local_device(0).device_type());
}

TEST_F(MasterTest, Reset) {
  GraphDef def;  // Empty.
  string s1, s2;
  int64_t initial_version1, initial_version2;
  TF_ASSERT_OK(CreateSession(def, &s1, &initial_version1));
  TF_ASSERT_OK(CreateSession(def, &s2, &initial_version2));
  EXPECT_TRUE(Reset().ok());
  EXPECT_TRUE(absl::IsAborted(CloseSession(s1)));
  EXPECT_TRUE(absl::IsAborted(CloseSession(s2)));
}

TEST_F(MasterTest, Extend) {
  GraphDef def_0;  // Empty.
  string handle;
  int64_t initial_version;
  TF_ASSERT_OK(CreateSession(def_0, &handle, &initial_version));

  Tensor A_expected(DT_FLOAT, TensorShape({2, 2}));
  test::FillValues<float>(&A_expected, {3.0, 2.0, -1.0, 0.0});

  Tensor x_expected(DT_FLOAT, TensorShape({2, 1}));
  test::FillValues<float>(&x_expected, {2.0, 2.0});

  Graph graph_1(OpRegistry::Global());
  test::graph::Constant(&graph_1, A_expected, "A");
  GraphDef def_1;
  test::graph::ToGraphDef(&graph_1, &def_1);
  int64_t version_1;
  TF_ASSERT_OK(ExtendSession(handle, def_1, initial_version, &version_1));
  EXPECT_GT(version_1, initial_version);
  Tensor A(DT_FLOAT, TensorShape({2, 2}));
  TF_ASSERT_OK(RunStep(handle, {}, {{"A:0", &A}}));
  test::ExpectTensorEqual<float>(A, A_expected);

  Graph graph_2(OpRegistry::Global());
  test::graph::Constant(&graph_2, x_expected, "x");
  GraphDef def_2;
  test::graph::ToGraphDef(&graph_2, &def_2);
  int64_t version_2;
  EXPECT_TRUE(absl::IsAborted(
      ExtendSession("randombits", def_2, version_1, &version_2)));
  TF_ASSERT_OK(ExtendSession(handle, def_2, version_1, &version_2));
  EXPECT_GT(version_2, version_1);

  Tensor x(DT_FLOAT, TensorShape({2, 1}));
  TF_ASSERT_OK(RunStep(handle, {}, {{"A:0", &A}, {"x:0", &x}}));
  test::ExpectTensorEqual<float>(A, A_expected);
  test::ExpectTensorEqual<float>(x, x_expected);

  TF_ASSERT_OK(CloseSession(handle));
}

TEST_F(MasterTest, ExtendUpdateStatefulFails) {
  GraphDef def_0;  // Empty.
  string handle;
  int64_t initial_version;
  TF_ASSERT_OK(CreateSession(def_0, &handle, &initial_version));

  Graph graph_1(OpRegistry::Global());
  test::graph::Var(&graph_1, DT_FLOAT, TensorShape({512}));
  GraphDef def_1;
  test::graph::ToGraphDef(&graph_1, &def_1);

  int64_t version_1, version_2;
  TF_ASSERT_OK(ExtendSession(handle, def_1, initial_version, &version_1));
  EXPECT_GT(version_1, initial_version);
  EXPECT_TRUE(absl::IsInvalidArgument(
      ExtendSession(handle, def_1, version_1, &version_2)));
  TF_ASSERT_OK(CloseSession(handle));
}

TEST_F(MasterTest, ExtendTwiceFails) {
  GraphDef def_0;  // Empty.
  string handle;
  int64_t initial_version;
  TF_ASSERT_OK(CreateSession(def_0, &handle, &initial_version));

  Graph graph_1(OpRegistry::Global());
  test::graph::Var(&graph_1, DT_FLOAT, TensorShape({512}));
  GraphDef def_1;
  test::graph::ToGraphDef(&graph_1, &def_1);

  int64_t version_1;
  TF_ASSERT_OK(ExtendSession(handle, def_1, initial_version, &version_1));
  EXPECT_GT(version_1, initial_version);
  EXPECT_TRUE(absl::IsAborted(
      ExtendSession(handle, def_1, initial_version, &version_1)));
  TF_ASSERT_OK(CloseSession(handle));
}

TEST_F(MasterTest, ConcurrentExtendOnlyOneSucceeds) {
  GraphDef def_0;  // Empty.
  string handle;
  int64_t initial_version;
  TF_ASSERT_OK(CreateSession(def_0, &handle, &initial_version));

  Graph graph_1(OpRegistry::Global());
  test::graph::Var(&graph_1, DT_FLOAT, TensorShape({512}));
  GraphDef def_1;
  test::graph::ToGraphDef(&graph_1, &def_1);

  Notification n;
  mutex mu;
  int succeeded = 0;
  int failed = 0;
  auto extend_fn = [this, handle, def_1, initial_version, &n, &mu, &succeeded,
                    &failed]() {
    n.WaitForNotification();
    int64_t new_version;
    absl::Status s =
        ExtendSession(handle, def_1, initial_version, &new_version);
    EXPECT_TRUE(s.ok() || absl::IsAborted(s));
    {
      mutex_lock l(mu);
      if (s.ok()) {
        ++succeeded;
      } else {
        ++failed;
      }
    }
  };

  // Run 100 concurrent Extend calls and expect only one to succeed.
  {
    thread::ThreadPool thread_pool(Env::Default(), "extend_pool", 100);
    for (int i = 0; i < 100; ++i) {
      thread_pool.Schedule(extend_fn);
    }
    n.Notify();
  }

  EXPECT_EQ(failed, 99);
  EXPECT_EQ(succeeded, 1);
  TF_ASSERT_OK(CloseSession(handle));
}

TEST_F(MasterTest, ConcurrentExtendAndRun) {
  Graph graph_0(OpRegistry::Global());
  Tensor a_tensor(DT_FLOAT, TensorShape({2, 2}));
  test::FillValues<float>(&a_tensor, {3, 2, -1, 0});
  test::graph::Constant(&graph_0, a_tensor, "A");
  GraphDef def_0;
  test::graph::ToGraphDef(&graph_0, &def_0);

  string handle;
  int64_t initial_version;
  TF_ASSERT_OK(CreateSession(def_0, &handle, &initial_version));

  Graph graph_1(OpRegistry::Global());
  Tensor b_tensor(DT_FLOAT, TensorShape({2, 2}));
  test::FillValues<float>(&b_tensor, {1, 0, 0, 1});
  test::graph::Constant(&graph_1, b_tensor, "B");
  GraphDef def_1;
  test::graph::ToGraphDef(&graph_1, &def_1);

  Notification extend_done;
  Notification extend_can_start;

  auto get_a_fn = [this, handle, &extend_done]() {
    Tensor A(DT_FLOAT, TensorShape({2, 2}));
    while (!extend_done.HasBeenNotified()) {
      TF_ASSERT_OK(RunStep(handle, {}, {{"A:0", &A}}));
    }
    // Run at least once after the Extend has completed.
    TF_ASSERT_OK(RunStep(handle, {}, {{"A:0", &A}}));
  };

  auto get_a_and_b_fn = [this, handle, &extend_done, &extend_can_start]() {
    Tensor A(DT_FLOAT, TensorShape({2, 2}));
    Tensor B(DT_FLOAT, TensorShape({2, 2}));

    // Run at least once before the Extend has completed.
    EXPECT_TRUE(
        absl::IsNotFound(RunStep(handle, {}, {{"A:0", &A}, {"B:0", &B}})));
    extend_can_start.Notify();

    // Concurrent with the Extend, we will either fail (as above), or
    // succeed (as below).
    while (!extend_done.HasBeenNotified()) {
      absl::Status s = RunStep(handle, {}, {{"A:0", &A}, {"B:0", &B}});
      EXPECT_TRUE(absl::IsNotFound(s) || s.ok());
    }

    // Run at least once after the Extend has completed.
    TF_ASSERT_OK(RunStep(handle, {}, {{"A:0", &A}, {"B:0", &B}}));
  };

  auto extend_fn = [this, handle, def_1, initial_version, &extend_done,
                    &extend_can_start]() {
    extend_can_start.WaitForNotification();
    int64_t version_1;
    TF_ASSERT_OK(ExtendSession(handle, def_1, initial_version, &version_1));
    extend_done.Notify();
  };

  {
    thread::ThreadPool thread_pool(Env::Default(), "extend_pool", 3);
    thread_pool.Schedule(get_a_fn);
    thread_pool.Schedule(get_a_and_b_fn);
    thread_pool.Schedule(extend_fn);
  }

  TF_ASSERT_OK(CloseSession(handle));
}

TEST_F(MasterTest, EigenProblem) {
  // A = [3 2; -1 0]; x = rand(2, 1);
  // for i=1:100; x = A * x; end
  // We'll try to compute the largest eigenvalue for A.
  Graph graph(OpRegistry::Global());
  Tensor a_tensor(DT_FLOAT, TensorShape({2, 2}));
  // Store rows [3, 2] and [-1, 0] in row major format.
  test::FillValues<float>(&a_tensor, {3, 2, -1, 0});
  Node* a_node = test::graph::Constant(&graph, a_tensor);

  // x is from the feed.
  Tensor x_tensor(DT_FLOAT, TensorShape({2, 1}));
  test::FillValues<float>(&x_tensor, {0, 0});
  Node* x_node = test::graph::Constant(&graph, x_tensor);

  // y = A * x
  Node* y_node = test::graph::Matmul(&graph, a_node, x_node, false, false);

  GraphDef def;
  test::graph::ToGraphDef(&graph, &def);

  string handle;
  int64_t initial_version;
  TF_CHECK_OK(CreateSession(def, &handle, &initial_version));

  // Temps supporting the computation of the convergence condition.
  const Eigen::array<Eigen::DenseIndex, 1> sum_along_dim{0};
  const Eigen::array<Eigen::DenseIndex, 2> matrix_transpose{1, 0};
  Tensor x(DT_FLOAT, TensorShape({2, 1}));
  Tensor y(DT_FLOAT, TensorShape({2, 1}));
  Eigen::Tensor<float, 1, Eigen::RowMajor> y_square_sum;
  Eigen::Tensor<float, 2, Eigen::RowMajor> y_normalized(2, 1);
  y_normalized.setRandom();
  Eigen::Tensor<float, 1, Eigen::RowMajor> error_square_sum;
  float lambda;

  // The computation loop.
  bool converged = false;
  while (!converged) {
    // Run one step of the graph.
    auto x_matrix = x.matrix<float>();
    x_matrix = y_normalized;
    TF_EXPECT_OK(
        RunStep(handle, {{x_node->name(), &x}}, {{y_node->name() + ":0", &y}}));
    auto y_matrix = y.matrix<float>();

    // Client code computes the convergence condition.
    {
      lambda = y_matrix(0, 0) / x_matrix(0, 0);
      y_square_sum = y.matrix<float>().square().sum(sum_along_dim);
      const float norm = static_cast<float>(sqrt(y_square_sum(0)));
      y_normalized = y_matrix * (1 / norm);
      error_square_sum = (x_matrix - y_normalized).square().sum(sum_along_dim);
      VLOG(1) << "x = [" << x_matrix.shuffle(matrix_transpose) << "] y = ["
              << y_matrix.shuffle(matrix_transpose) << "] lambda = " << lambda;
      converged = sqrt(error_square_sum(0)) < 1e-10;
    }
  }
  EXPECT_NEAR(lambda, 2.0, 0.01);
  TF_EXPECT_OK(CloseSession(handle));
}

}  // namespace tensorflow
