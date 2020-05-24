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

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/common_runtime/direct_session.h"
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
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/profiler/internal/profiler_interface.h"
#include "tensorflow/core/profiler/lib/profiler_session.h"
#include "tensorflow/core/profiler/utils/tf_xplane_visitor.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace profiler {

#if GOOGLE_CUDA
extern std::unique_ptr<ProfilerInterface> CreateGpuTracer(
    const ProfileOptions& options);
std::unique_ptr<ProfilerInterface> CreateGpuTracer() {
  ProfileOptions options = ProfilerSession::DefaultOptions();
  return CreateGpuTracer(options);
}
#else
// We don't have device tracer for non-cuda case.
std::unique_ptr<ProfilerInterface> CreateGpuTracer() { return nullptr; }
#endif

namespace {

std::unique_ptr<Session> CreateSession() {
  SessionOptions options;
  (*options.config.mutable_device_count())["CPU"] = 1;
  (*options.config.mutable_device_count())["GPU"] = 1;
  options.config.set_allow_soft_placement(true);
  return std::unique_ptr<Session>(NewSession(options));
}

class DeviceTracerTest : public ::testing::Test {
 public:
  void Initialize(std::initializer_list<float> a_values) {
    Graph graph(OpRegistry::Global());

    Tensor a_tensor(DT_FLOAT, TensorShape({2, 2}));
    test::FillValues<float>(&a_tensor, a_values);
    Node* a = test::graph::HostConstant(&graph, a_tensor);

    Tensor x_tensor(DT_FLOAT, TensorShape({2, 1}));
    test::FillValues<float>(&x_tensor, {1, 1});
    Node* x = test::graph::HostConstant(&graph, x_tensor);
    x_ = x->name();

    // y = A * x
    Node* y = test::graph::Matmul(&graph, a, x, false, false);
    y->set_assigned_device_name("/device:GPU:0");
    y_ = y->name();

    // Use an Identity op to force a memcpy to CPU and back to GPU.
    Node* i = test::graph::Identity(&graph, y);
    i->set_assigned_device_name("/cpu:0");

    Node* y_neg = test::graph::Unary(&graph, "Neg", i);
    y_neg_ = y_neg->name();
    y_neg->set_assigned_device_name("/device:GPU:0");

    test::graph::ToGraphDef(&graph, &def_);
  }

 protected:
  void ExpectFailure(const Status& status, error::Code code) {
    EXPECT_FALSE(status.ok()) << status.ToString();
    if (!status.ok()) {
      LOG(INFO) << "Status message: " << status.error_message();
      EXPECT_EQ(code, status.code()) << status.ToString();
    }
  }

  string x_;
  string y_;
  string y_neg_;
  GraphDef def_;
};

TEST_F(DeviceTracerTest, StartStop) {
  auto tracer = CreateGpuTracer();
  if (!tracer) return;
  TF_EXPECT_OK(tracer->Start());
  TF_EXPECT_OK(tracer->Stop());
}

TEST_F(DeviceTracerTest, StopBeforeStart) {
  auto tracer = CreateGpuTracer();
  if (!tracer) return;
  TF_EXPECT_OK(tracer->Stop());
  TF_EXPECT_OK(tracer->Stop());
}

TEST_F(DeviceTracerTest, CollectBeforeStart) {
  auto tracer = CreateGpuTracer();
  if (!tracer) return;
  RunMetadata run_metadata;
  TF_EXPECT_OK(tracer->CollectData(&run_metadata));
  EXPECT_EQ(run_metadata.step_stats().dev_stats_size(), 0);
}

TEST_F(DeviceTracerTest, CollectBeforeStop) {
  auto tracer = CreateGpuTracer();
  if (!tracer) return;
  TF_EXPECT_OK(tracer->Start());
  RunMetadata run_metadata;
  Status status = tracer->CollectData(&run_metadata);
  ExpectFailure(status, tensorflow::error::FAILED_PRECONDITION);
  TF_EXPECT_OK(tracer->Stop());
}

TEST_F(DeviceTracerTest, StartTwoTracers) {
  auto tracer1 = CreateGpuTracer();
  auto tracer2 = CreateGpuTracer();
  if (!tracer1 || !tracer2) return;

  TF_EXPECT_OK(tracer1->Start());
  Status status = tracer2->Start();
  ExpectFailure(status, tensorflow::error::UNAVAILABLE);
  TF_EXPECT_OK(tracer1->Stop());
  TF_EXPECT_OK(tracer2->Start());
  TF_EXPECT_OK(tracer2->Stop());
}

TEST_F(DeviceTracerTest, RunWithTracer) {
  // On non-GPU platforms, we may not support DeviceTracer.
  auto tracer = CreateGpuTracer();
  if (!tracer) return;

  Initialize({3, 2, -1, 0});
  auto session = CreateSession();
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

TEST_F(DeviceTracerTest, TraceToStepStatsCollector) {
  auto tracer = CreateGpuTracer();
  if (!tracer) return;

  Initialize({3, 2, -1, 0});
  auto session = CreateSession();
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
  RunMetadata run_metadata;
  TF_ASSERT_OK(tracer->CollectData(&run_metadata));
  // Depending on whether this runs on CPU or GPU, we will have a
  // different number of devices.
  EXPECT_GE(run_metadata.step_stats().dev_stats_size(), 1)
      << "Saw stats: " << run_metadata.DebugString();
}

TEST_F(DeviceTracerTest, RunWithTraceOption) {
  Initialize({3, 2, -1, 0});
  auto session = CreateSession();
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

TEST_F(DeviceTracerTest, TraceToXSpace) {
  auto tracer = CreateGpuTracer();
  if (!tracer) return;

  Initialize({3, 2, -1, 0});
  auto session = CreateSession();
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
  XSpace space;
  TF_ASSERT_OK(tracer->CollectData(&space));
  // At least one gpu plane and one host plane for launching events.
  const XPlane* host_plane = FindPlaneWithName(space, kCuptiDriverApiPlaneName);
  ASSERT_NE(host_plane, nullptr);
  EXPECT_EQ(host_plane->id(), kCuptiDriverApiPlaneId);

  const XPlane* device_plane =
      FindPlaneWithName(space, strings::StrCat(kGpuPlanePrefix, 0));
  ASSERT_NE(device_plane, nullptr);  // Check if device plane is serialized.
  EXPECT_EQ(device_plane->id(), kGpuPlaneBaseId);
  // one for MemcpyH2D, one for MemcpyD2H, two for Matmul (one from Eigen, one
  // from cudnn).
  EXPECT_EQ(device_plane->event_metadata_size(), 4);
  // Check if device capacity is serialized.
  XPlaneVisitor plane = CreateTfXPlaneVisitor(device_plane);
  EXPECT_NE(plane.GetStats(kDevCapClockRateKHz), nullptr);
  EXPECT_NE(plane.GetStats(kDevCapCoreCount), nullptr);
  EXPECT_NE(plane.GetStats(kDevCapMemoryBandwidth), nullptr);
  EXPECT_NE(plane.GetStats(kDevCapMemorySize), nullptr);
  EXPECT_NE(plane.GetStats(kDevCapComputeCapMajor), nullptr);
  EXPECT_NE(plane.GetStats(kDevCapComputeCapMinor), nullptr);

  // Check if the device events timestamps are set.
  int total_events = 0;
  plane.ForEachLine([&](const tensorflow::profiler::XLineVisitor& line) {
    line.ForEachEvent([&](const tensorflow::profiler::XEventVisitor& event) {
      EXPECT_GT(event.TimestampNs(), 0);
      EXPECT_GT(event.DurationNs(), 0);
      ++total_events;
    });
  });
  EXPECT_EQ(total_events, 5);
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
