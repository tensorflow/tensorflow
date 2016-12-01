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

#include "tensorflow/core/common_runtime/gpu/gpu_tracer.h"

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/common_runtime/direct_session.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace {

Session* CreateSession() {
  SessionOptions options;
  (*options.config.mutable_device_count())["CPU"] = 1;
  (*options.config.mutable_device_count())["GPU"] = 1;
  options.config.set_allow_soft_placement(true);
  return NewSession(options);
}

class GPUTracerTest : public ::testing::Test {
 public:
  void Initialize(std::initializer_list<float> a_values) {
    Graph graph(OpRegistry::Global());

    Tensor a_tensor(DT_FLOAT, TensorShape({2, 2}));
    test::FillValues<float>(&a_tensor, a_values);
    Node* a = test::graph::Constant(&graph, a_tensor);
    a->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:0");

    Tensor x_tensor(DT_FLOAT, TensorShape({2, 1}));
    test::FillValues<float>(&x_tensor, {1, 1});
    Node* x = test::graph::Constant(&graph, x_tensor);
    x->set_assigned_device_name("/job:localhost/replica:0/task:0/gpu:0");
    x_ = x->name();

    // y = A * x
    Node* y = test::graph::Matmul(&graph, a, x, false, false);
    y->set_assigned_device_name("/job:localhost/replica:0/task:0/gpu:0");
    y_ = y->name();

    // Use an Identity op to force a memcpy to CPU and back to GPU.
    Node* i = test::graph::Identity(&graph, y);
    i->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:0");

    Node* y_neg = test::graph::Unary(&graph, "Neg", i);
    y_neg_ = y_neg->name();
    y_neg->set_assigned_device_name("/job:localhost/replica:0/task:0/gpu:0");

    test::graph::ToGraphDef(&graph, &def_);
  }

 protected:
  void ExpectFailure(Status status, error::Code code) {
    EXPECT_FALSE(status.ok());
    if (!status.ok()) {
      LOG(INFO) << "Status message: " << status.error_message();
      EXPECT_EQ(code, status.code());
    }
  }

  string x_;
  string y_;
  string y_neg_;
  GraphDef def_;
};

TEST_F(GPUTracerTest, StartStop) {
  std::unique_ptr<GPUTracer> tracer;
  tracer.reset(CreateGPUTracer());
  if (!tracer) return;
  TF_EXPECT_OK(tracer->Start());
  TF_EXPECT_OK(tracer->Stop());
}

TEST_F(GPUTracerTest, StopBeforeStart) {
  std::unique_ptr<GPUTracer> tracer;
  tracer.reset(CreateGPUTracer());
  if (!tracer) return;
  TF_EXPECT_OK(tracer->Stop());
  TF_EXPECT_OK(tracer->Stop());
}

TEST_F(GPUTracerTest, CollectBeforeStart) {
  std::unique_ptr<GPUTracer> tracer;
  tracer.reset(CreateGPUTracer());
  if (!tracer) return;
  StepStats stats;
  StepStatsCollector collector(&stats);
  TF_EXPECT_OK(tracer->Collect(&collector));
  EXPECT_EQ(stats.dev_stats_size(), 0);
}

TEST_F(GPUTracerTest, CollectBeforeStop) {
  std::unique_ptr<GPUTracer> tracer;
  tracer.reset(CreateGPUTracer());
  if (!tracer) return;
  TF_EXPECT_OK(tracer->Start());
  StepStats stats;
  StepStatsCollector collector(&stats);
  Status status = tracer->Collect(&collector);
  ExpectFailure(status, tensorflow::error::FAILED_PRECONDITION);
  TF_EXPECT_OK(tracer->Stop());
}

TEST_F(GPUTracerTest, StartTwoTracers) {
  std::unique_ptr<GPUTracer> tracer1;
  tracer1.reset(CreateGPUTracer());
  std::unique_ptr<GPUTracer> tracer2;
  tracer2.reset(CreateGPUTracer());
  if (!tracer1 || !tracer2) return;

  TF_EXPECT_OK(tracer1->Start());
  Status status = tracer2->Start();
  ExpectFailure(status, tensorflow::error::UNAVAILABLE);
  TF_EXPECT_OK(tracer1->Stop());
  TF_EXPECT_OK(tracer2->Start());
  TF_EXPECT_OK(tracer2->Stop());
}

TEST_F(GPUTracerTest, RunWithTracer) {
  // On non-GPU platforms, we may not support GPUTracer.
  std::unique_ptr<GPUTracer> tracer;
  tracer.reset(CreateGPUTracer());
  if (!tracer) return;

  Initialize({3, 2, -1, 0});
  std::unique_ptr<Session> session(CreateSession());
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def_));
  std::vector<std::pair<string, Tensor>> inputs;

  // Request two targets: one fetch output and one non-fetched output.
  std::vector<string> output_names = {y_ + ":0"};
  std::vector<string> target_nodes = {y_neg_};
  std::vector<Tensor> outputs;

  TF_ASSERT_OK(tracer->Start());
  Status s = session->Run(inputs, output_names, target_nodes, &outputs);
  TF_ASSERT_OK(s);
  TF_ASSERT_OK(tracer->Stop());
  ASSERT_EQ(1, outputs.size());
  // The first output should be initialized and have the correct
  // output.
  auto mat = outputs[0].matrix<float>();
  ASSERT_TRUE(outputs[0].IsInitialized());
  EXPECT_FLOAT_EQ(5.0, mat(0, 0));
}

TEST_F(GPUTracerTest, TraceToStepStatsCollector) {
  std::unique_ptr<GPUTracer> tracer;
  tracer.reset(CreateGPUTracer());
  if (!tracer) return;

  Initialize({3, 2, -1, 0});
  std::unique_ptr<Session> session(CreateSession());
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def_));
  std::vector<std::pair<string, Tensor>> inputs;

  // Request two targets: one fetch output and one non-fetched output.
  std::vector<string> output_names = {y_ + ":0"};
  std::vector<string> target_nodes = {y_neg_};
  std::vector<Tensor> outputs;

  TF_ASSERT_OK(tracer->Start());
  Status s = session->Run(inputs, output_names, target_nodes, &outputs);
  TF_ASSERT_OK(s);

  TF_ASSERT_OK(tracer->Stop());
  StepStats stats;
  StepStatsCollector collector(&stats);
  TF_ASSERT_OK(tracer->Collect(&collector));
  // Depending on whether this runs on CPU or GPU, we will have a
  // different number of devices.
  EXPECT_GE(stats.dev_stats_size(), 1);
}

TEST_F(GPUTracerTest, RunWithTraceOption) {
  Initialize({3, 2, -1, 0});
  std::unique_ptr<Session> session(CreateSession());
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def_));
  std::vector<std::pair<string, Tensor>> inputs;

  // Request two targets: one fetch output and one non-fetched output.
  std::vector<string> output_names = {y_ + ":0"};
  std::vector<string> target_nodes = {y_neg_};
  std::vector<Tensor> outputs;

  // Prepares RunOptions and RunOutputs
  RunOptions run_options;
  run_options.set_trace_level(RunOptions::FULL_TRACE);
  RunMetadata run_metadata;
  Status s = session->Run(run_options, inputs, output_names, target_nodes,
                          &outputs, &run_metadata);
  TF_ASSERT_OK(s);
  ASSERT_TRUE(run_metadata.has_step_stats());
  // Depending on whether this runs on CPU or GPU, we will have a
  // different number of devices.
  EXPECT_GE(run_metadata.step_stats().dev_stats_size(), 1);
}

}  // namespace
}  // namespace tensorflow
