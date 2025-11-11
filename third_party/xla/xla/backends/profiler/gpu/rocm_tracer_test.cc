/* Copyright 2025 The OpenXLA Authors. All Rights Reserved.

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

#include "xla/backends/profiler/gpu/rocm_tracer.h"

#include <cstdint>
#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "xla/backends/profiler/gpu/rocm_collector.h"
#include "xla/backends/profiler/gpu/rocm_tracer_utils.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {
namespace {

using tsl::profiler::XSpace;

// Minimal mock collector implementation based on RocmTraceCollectorImpl.
class TestRocmTraceCollector : public RocmTraceCollectorImpl {
 public:
  TestRocmTraceCollector(const RocmTraceCollectorOptions& options,
                         uint64_t start_walltime_ns, uint64_t start_gputime_ns)
      : RocmTraceCollectorImpl(options, start_walltime_ns, start_gputime_ns) {}

  void Export(XSpace* space) override {
    exported_ = true;
    exported_space_ = space;
  }

  void OnEventsDropped(const std::string& reason,
                       uint32_t correlation_id) override {
    dropped_reason_ = reason;
    dropped_id_ = correlation_id;
  }

  bool exported() const { return exported_; }
  const std::string& dropped_reason() const { return dropped_reason_; }
  uint32_t dropped_id() const { return dropped_id_; }
  XSpace* exported_space() const { return exported_space_; }

 private:
  bool exported_ = false;
  std::string dropped_reason_;
  uint32_t dropped_id_ = 0;
  XSpace* exported_space_ = nullptr;
};

// Utility to create valid options for the test collector.
std::unique_ptr<TestRocmTraceCollector> CreateTestCollector() {
  RocmTraceCollectorOptions options;
  options.max_callback_api_events = 2 * 1024 * 1024;
  options.max_activity_api_events = 2 * 1024 * 1024;
  options.max_annotation_strings = 1024 * 1024;
  options.num_gpus = 1;

  uint64_t walltime_ns = RocmTracer::GetTimestamp();
  uint64_t gputime_ns = RocmTracer::GetTimestamp();

  return std::make_unique<TestRocmTraceCollector>(options, walltime_ns,
                                                  gputime_ns);
}

TEST(RocmTracerTest, SingletonInstance) {
  LOG(INFO) << "Before RocmTracer::GetRocmTracerSingleton()";
  RocmTracer& tracer1 = RocmTracer::GetRocmTracerSingleton();
  RocmTracer& tracer2 = RocmTracer::GetRocmTracerSingleton();
  LOG(INFO) << "Before RocmTracer::GetRocmTracerSingleton()";
  EXPECT_EQ(&tracer1, &tracer2) << "RocmTracer must be a singleton";
}

TEST(RocmTracerTest, InitialStateIsAvailable) {
  RocmTracer& tracer = RocmTracer::GetRocmTracerSingleton();
  EXPECT_TRUE(tracer.IsAvailable())
      << "Tracer should be available before Enable()";
}

TEST(RocmTracerTest, EnableAndDisableLifecycle) {
  RocmTracer& tracer = RocmTracer::GetRocmTracerSingleton();
  auto collector = CreateTestCollector();

  RocmTracerOptions tracer_options{/*max_annotation_strings=*/128};
  tracer.Enable(tracer_options, collector.get());

  EXPECT_FALSE(tracer.IsAvailable())
      << "Tracer should not be available after Enable()";
  EXPECT_EQ(tracer.collector(), collector.get())
      << "Collector should be set after Enable()";
  ASSERT_NE(tracer.annotation_map(), nullptr)
      << "Annotation map should be initialized";

  tracer.Disable();

  EXPECT_TRUE(tracer.IsAvailable())
      << "Tracer should be available after Disable()";
}

TEST(RocmTracerTest, AnnotationMapWorks) {
  RocmTracer& tracer = RocmTracer::GetRocmTracerSingleton();
  AnnotationMap* map = tracer.annotation_map();
  ASSERT_NE(map, nullptr);

  uint64_t id = 42;
  std::string annotation = "matmul_fused_op";
  map->Add(id, annotation);

  absl::string_view result = map->LookUp(id);
  EXPECT_EQ(result, annotation);
}

}  // namespace
}  // namespace profiler
}  // namespace xla
