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

#include <array>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_activity.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime.h"
#include "tensorflow/core/profiler/internal/gpu/cupti_collector.h"
#endif  // GOOGLE_CUDA
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
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/profiler/lib/profiler_interface.h"
#include "tensorflow/core/profiler/lib/profiler_session.h"
#include "tensorflow/core/profiler/utils/tf_xplane_visitor.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/device_name_utils.h"

// TODO(b/186367334)
#define CUPTI_NVBUG_3299481_WAR (10000 <= CUDA_VERSION && CUDA_VERSION < 11000)

namespace tensorflow {
namespace profiler {

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
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
  XSpace space;
  TF_EXPECT_OK(tracer->CollectData(&space));
  EXPECT_EQ(space.planes_size(), 0);
}

TEST_F(DeviceTracerTest, CollectBeforeStop) {
  auto tracer = CreateGpuTracer();
  if (!tracer) return;
  TF_EXPECT_OK(tracer->Start());
  XSpace space;
  Status status = tracer->CollectData(&space);
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

#if TENSORFLOW_USE_ROCM
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
  const XPlane* host_plane = FindPlaneWithName(space, kRoctracerApiPlaneName);
  ASSERT_NE(host_plane, nullptr);

  const XPlane* device_plane = FindPlaneWithName(space, GpuPlaneName(0));
  ASSERT_NE(device_plane, nullptr);  // Check if device plane is serialized.
  // The device plane should have at least five events: one for MemcpyH2D, one
  // for MemcpyD2H, two for Matmul (one from Eigen, one from cudnn), one for
  // memset.
  EXPECT_GE(device_plane->event_metadata_size(), 5);
  // Check if device capacity is serialized.
  XPlaneVisitor plane = CreateTfXPlaneVisitor(device_plane);

  // Check if the device events timestamps are set.
  int total_events = 0;
  plane.ForEachLine([&](const tensorflow::profiler::XLineVisitor& line) {
    line.ForEachEvent([&](const tensorflow::profiler::XEventVisitor& event) {
      EXPECT_GT(event.TimestampNs(), 0);
      EXPECT_GT(event.DurationNs(), 0);
      ++total_events;
    });
  });
  EXPECT_GE(total_events, 5);
}
#endif  // TENSORFLOW_USE_ROCM

#if GOOGLE_CUDA
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

  const XPlane* device_plane = FindPlaneWithName(space, GpuPlaneName(0));
  ASSERT_NE(device_plane, nullptr);  // Check if device plane is serialized.
  // The device plane should have at least five events: one for MemcpyH2D, one
  // for MemcpyD2H, two for Matmul (one from Eigen, one from cudnn), one for
  // memset.
  EXPECT_GE(device_plane->event_metadata_size(), 5);
  // Check if device capacity is serialized.
  XPlaneVisitor plane = CreateTfXPlaneVisitor(device_plane);
  EXPECT_TRUE(plane.GetStat(kDevCapClockRateKHz).has_value());
  EXPECT_TRUE(plane.GetStat(kDevCapCoreCount).has_value());
  EXPECT_TRUE(plane.GetStat(kDevCapMemoryBandwidth).has_value());
  EXPECT_TRUE(plane.GetStat(kDevCapMemorySize).has_value());
  EXPECT_TRUE(plane.GetStat(kDevCapComputeCapMajor).has_value());
  EXPECT_TRUE(plane.GetStat(kDevCapComputeCapMinor).has_value());

  // Check if the device events timestamps are set.
  int total_events = 0;
  plane.ForEachLine([&](const tensorflow::profiler::XLineVisitor& line) {
    line.ForEachEvent([&](const tensorflow::profiler::XEventVisitor& event) {
      EXPECT_GT(event.TimestampNs(), 0);
      EXPECT_GT(event.DurationNs(), 0);
      ++total_events;
    });
  });
  EXPECT_GE(total_events, 5);
}

TEST_F(DeviceTracerTest, CudaRuntimeResource) {
  auto tracer = CreateGpuTracer();
  if (!tracer) return;
  const size_t size_in_bytes = 8;
  const int8_t test_value = 7;
  TF_EXPECT_OK(tracer->Start());
  void* hostptr = 0;
  void* devptr = 0;

  // These six CUDA API calls will create 4 CUDA API and 4 device XEvents.
  ASSERT_EQ(cudaSuccess,
            cudaHostAlloc(&hostptr, size_in_bytes, cudaHostAllocPortable));
  ASSERT_EQ(cudaSuccess, cudaMalloc(&devptr, size_in_bytes));
  VLOG(3) << "Allocated device memory, addr: " << devptr;
  ASSERT_EQ(cudaSuccess, cudaMemset(devptr, test_value, size_in_bytes));
  ASSERT_EQ(cudaSuccess,
            cudaMemcpy(hostptr, devptr, size_in_bytes, cudaMemcpyDeviceToHost));
  VLOG(3) << "Free device memory, addr: " << devptr;
  auto* value_from_device = static_cast<uint8_t*>(hostptr);
  for (size_t idx = 0; idx < size_in_bytes; ++idx) {
    ASSERT_EQ(value_from_device[idx], test_value);
  }
  ASSERT_EQ(cudaSuccess, cudaFree(devptr));
  ASSERT_EQ(cudaSuccess, cudaFreeHost(hostptr));
  TF_EXPECT_OK(tracer->Stop());
  XSpace space;
  TF_EXPECT_OK(tracer->CollectData(&space));
  const XPlane* cupti_host_plane =
      FindPlaneWithName(space, kCuptiDriverApiPlaneName);
  ASSERT_NE(cupti_host_plane, nullptr);

  XPlaneVisitor host_plane = CreateTfXPlaneVisitor(cupti_host_plane);
  // Expect at least one XLine for CUPTI activity events. There may be an
  // additional line for CuptiTracerEventType Overhead.
  EXPECT_GE(host_plane.NumLines(), 1);

  // These follow the order in which they were invoked above.
  const std::array<StatType, 4> expected_event_stat_type = {
      kMemallocDetails,
      kMemsetDetails,
      kMemcpyDetails,
      kMemFreeDetails,
  };

  int event_idx = 0;

  host_plane.ForEachLine([&](const tensorflow::profiler::XLineVisitor& line) {
    VLOG(3) << "Line " << line.Id() << "\n";
    line.ForEachEvent([&](const tensorflow::profiler::XEventVisitor& event) {
      VLOG(3) << " Event " << *event.Type() << "\n";

      absl::optional<XStatVisitor> stat =
          event.GetStat(expected_event_stat_type[event_idx]);
      // The stat may not exist if we're looking at the wrong line.
      if (stat.has_value()) {
        event_idx += 1;
        VLOG(3) << "  Stat name=" << stat->Name() << " type=" << *stat->Type()
                << " " << stat->ToString() << "\n";
      }
    });
  });

  // Expect that all stats show up. One host side event for each API call.
  EXPECT_EQ(event_idx, expected_event_stat_type.size());

  const XPlane* cupti_device_plane = FindPlaneWithName(space, GpuPlaneName(0));
  ASSERT_NE(cupti_device_plane, nullptr);
  XPlaneVisitor device_plane = CreateTfXPlaneVisitor(cupti_device_plane);

  bool found_activity_memory_host = false;
  bool found_activity_memory_device = false;
  bool found_activity_memset = false;
  bool found_activity_memcpy = false;

  device_plane.ForEachLine([&](const tensorflow::profiler::XLineVisitor& line) {
    line.ForEachEvent([&](const tensorflow::profiler::XEventVisitor& event) {
      event.ForEachStat([&](XStatVisitor stat) {
        VLOG(3) << "  Stat name=" << stat.Name() << " type=" << *stat.Type()
                << " " << stat.ToString() << "\n";
        // These are the attributes set in cupti_collector::CreateXEvent.
        auto space_delimited_stats = absl::StrSplit(stat.StrOrRefValue(), " ");

        if (stat.Type() == StatType::kMemoryResidencyDetails) {
          size_t num_bytes = 0;
          size_t addr = 0;
          absl::string_view kind;
          for (const auto& detail : space_delimited_stats) {
            std::vector<absl::string_view> name_value =
                absl::StrSplit(detail, ":");
            // We allocated 8 bytes of device memory with cudaMalloc/Free.
            if (absl::StartsWith(detail, "num_bytes:")) {
              (void)absl::SimpleAtoi(name_value[1], &num_bytes);
            } else if (absl::StartsWith(detail, "addr:")) {
              std::stringstream hex_string;
              hex_string << std::hex << name_value[1];
              hex_string >> addr;
            } else if (absl::StartsWith(detail, "kind:")) {
              kind = name_value[1];
            }
          }

          if (addr == reinterpret_cast<size_t>(devptr) &&
              num_bytes == size_in_bytes) {
            found_activity_memory_device = true;
            EXPECT_EQ(kind,
                      GetMemoryKindName(CUPTI_ACTIVITY_MEMORY_KIND_DEVICE));
          } else if (addr == reinterpret_cast<size_t>(hostptr) &&
                     num_bytes == size_in_bytes) {
            found_activity_memory_host = true;
            EXPECT_EQ(kind,
                      GetMemoryKindName(CUPTI_ACTIVITY_MEMORY_KIND_PINNED));
          }
        } else if (stat.Type() == StatType::kMemsetDetails) {
          CHECK(!found_activity_memset);
          found_activity_memset = true;
          for (const auto& detail : space_delimited_stats) {
            std::vector<absl::string_view> name_value =
                absl::StrSplit(detail, ":");
            // We set 8 bytes of device memory with cudaMemset.
            if (absl::StartsWith(detail, "num_bytes:")) {
              size_t num_bytes = 0;
              (void)absl::SimpleAtoi(name_value[1], &num_bytes);
              EXPECT_EQ(num_bytes, 8);
            } else if (absl::StartsWith(detail, "kind:")) {
              EXPECT_EQ(name_value[1],
                        GetMemoryKindName(CUPTI_ACTIVITY_MEMORY_KIND_DEVICE));
            }
          }
        } else if (stat.Type() == StatType::kMemcpyDetails) {
          CHECK(!found_activity_memcpy);
          found_activity_memcpy = true;
          for (const auto& detail : space_delimited_stats) {
            std::vector<absl::string_view> name_value =
                absl::StrSplit(detail, ":");
            // We copied 8 bytes from device memory to pinned host memory.
            if (absl::StartsWith(detail, "num_bytes:")) {
              size_t num_bytes = 0;
              (void)absl::SimpleAtoi(name_value[1], &num_bytes);
              EXPECT_EQ(num_bytes, 8);
            } else if (absl::StartsWith(detail, "kind_src:")) {
              EXPECT_EQ(name_value[1],
                        GetMemoryKindName(CUPTI_ACTIVITY_MEMORY_KIND_DEVICE));
            } else if (absl::StartsWith(detail, "kind_dst:")) {
              EXPECT_EQ(name_value[1],
                        GetMemoryKindName(CUPTI_ACTIVITY_MEMORY_KIND_PINNED));
            }
          }
        }
      });
    });
  });

  // Expect these CUDA API activities to be found.
  EXPECT_TRUE(found_activity_memset);
  EXPECT_TRUE(found_activity_memcpy);

  EXPECT_FALSE(found_activity_memory_device);
#if CUPTI_NVBUG_3299481_WAR
  EXPECT_FALSE(found_activity_memory_host);
#endif
}

void BM_CuptiTracer_ActivityBufferOverhead(::testing::benchmark::State& state) {
  auto tracer = CreateGpuTracer();
  CHECK_NOTNULL(tracer);

  TF_EXPECT_OK(tracer->Start());
  for (auto s : state) {
    cudaFree(nullptr);
  }
  TF_EXPECT_OK(tracer->Stop());
}

BENCHMARK(BM_CuptiTracer_ActivityBufferOverhead);

void BM_CudaFree(::testing::benchmark::State& state) {
  for (auto s : state) {
    cudaFree(nullptr);
  }
}

BENCHMARK(BM_CudaFree);

#endif  // GOOGLE_CUDA

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
