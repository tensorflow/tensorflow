/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/profiler/gpu/profile_with_cuda_kernels.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/backends/profiler/gpu/cupti_collector.h"
#include "xla/backends/profiler/gpu/cupti_error_manager.h"
#include "xla/backends/profiler/gpu/cupti_pm_sampler.h"
#include "xla/backends/profiler/gpu/cupti_tracer.h"
#include "xla/backends/profiler/gpu/cupti_wrapper.h"

namespace xla {
namespace profiler {
namespace test {

namespace {

using ::testing::DistanceFrom;
using ::testing::Each;
using ::testing::Lt;
using xla::profiler::CuptiTracer;
using xla::profiler::CuptiTracerCollectorOptions;
using xla::profiler::CuptiTracerOptions;
using xla::profiler::CuptiWrapper;

// Needed to create different cupti tracer for each test cases.
class TestableCuptiTracer : public CuptiTracer {
 public:
  explicit TestableCuptiTracer(CuptiErrorManager* error_manager)
      : CuptiTracer(error_manager) {}
};

std::atomic_uint64_t records_fp64 = 0;
std::atomic_uint64_t total_fp64 = 0;
std::atomic_uint64_t records_cycles = 0;
std::atomic_uint64_t total_cycles = 0;
std::atomic_bool skip_first = true;

void HandleRecords(PmSamples* samples) {
  // Validate some samples were recorded
  EXPECT_GT(samples->GetNumSamples(), 0);

  LOG(INFO) << "PM Sampling buffer flushed with " << samples->GetNumSamples()
            << " samples";

  // Validate we have the expected metrics
  const std::vector<std::string>& metrics = samples->GetMetrics();
  const std::vector<SamplerRange>& sampler_ranges = samples->GetSamplerRanges();
  auto back = sampler_ranges.back();
  auto front = sampler_ranges.front();
  double ranges_duration = back.end_timestamp_ns - front.start_timestamp_ns;
  double ns_per_sample = ranges_duration / samples->GetNumSamples();

  // First pass may have large initial sample duration
  if (skip_first) {
    skip_first = false;
  } else {
    EXPECT_GT(ns_per_sample, 500000.0 * 0.9);
    EXPECT_LT(ns_per_sample, 500000.0 * 1.1);
  }

  for (int i = 0; i < metrics.size(); i++) {
    double sum = 0;

    for (int j = 0; j < sampler_ranges.size(); j++) {
      sum += sampler_ranges[j].metric_values[i];
    }

    if ("sm__inst_executed_pipe_fp64.sum" == metrics[i]) {
      records_fp64 += 1;
      total_fp64 += sum;
    } else if ("sm__inst_executed_pipe_fmaheavy.sum" == metrics[i]) {
      records_fp64 += 1;
      total_fp64 += sum;
    } else if ("sm__cycles_active.sum" == metrics[i]) {
      records_cycles += 1;
      total_cycles += sum;
    } else {
      GTEST_FAIL() << "Unknown metric: " << metrics[i];
    }
  }
}

TEST(ProfilerCudaKernelSanityTest, SimpleAddSub) {
  // Ensure this is only run on CUPTI > 26 (paired w/ CUDA 12.6)
  if (CUPTI_API_VERSION < 24) {
    GTEST_SKIP() << "PM Sampling not supported on this version of CUPTI";
  }

  uint32_t cupti_version = 0;
  cuptiGetVersion(&cupti_version);
  EXPECT_EQ(CUPTI_API_VERSION, cupti_version);

  if (cupti_version < 24) {
    GTEST_SKIP() << "PM Sampling not supported on this version of CUPTI";
  }

  LOG(INFO) << "Starting PM Sampling test with CUPTI version " << cupti_version;

  constexpr int kNumElements = 256 * 1024;

  CuptiTracerCollectorOptions collector_options;
  collector_options.num_gpus = CuptiTracer::NumGpus();
  uint64_t start_walltime_ns = absl::GetCurrentTimeNanos();
  uint64_t start_gputime_ns = CuptiTracer::GetTimestamp();
  auto collector = CreateCuptiCollector(collector_options, start_walltime_ns,
                                        start_gputime_ns);

  CuptiPmSamplerOptions sampler_options;
  sampler_options.enable = true;
  // Metrics can be queried with Nsight Compute
  // ncu --query-metrics-collection pmsampling --chip <CHIP>
  // Any metrics marked with a particular Triage group naming should be
  // configurable in a single pass on this chip.  Other combinations may not be
  // possible in a single pass and are not valid for pm sampling.
  sampler_options.metrics = {"sm__cycles_active.sum",
                             "sm__inst_executed_pipe_fp64.sum"};
  sampler_options.process_samples = HandleRecords;

  CuptiTracerOptions tracer_options;
  tracer_options.enable_nvtx_tracking = false;

  tracer_options.pm_sampler_options = sampler_options;

  CuptiErrorManager error_manager(std::make_unique<CuptiWrapper>());
  TestableCuptiTracer tracer(&error_manager);

  std::vector<std::unique_ptr<tensorflow::profiler::XPlane>> xplanes;
  xplanes.reserve(collector_options.num_gpus);
  for (int i = 0; i < collector_options.num_gpus; ++i) {
    xplanes.push_back(std::make_unique<tensorflow::profiler::XPlane>());
  }
  auto err = tracer.Enable(tracer_options, collector.get(), xplanes);

  if (absl::IsPermissionDenied(err)) {
    GTEST_SKIP() << "PM Sampling requires root access";
  }

  // SimpleAddSub does num_elements * 4 integer add / subs
  std::vector<double> vec = SimpleAddSubWithProfiler(kNumElements);

  tracer.Disable();

  // Go for a second pass to ensure we can sample again, with an un-sampled func
  absl::SleepFor(absl::Milliseconds(1000));
  skip_first = true;
  vec = SimpleAddSubWithProfiler(kNumElements);

  xplanes.clear();
  xplanes.reserve(collector_options.num_gpus);
  for (int i = 0; i < collector_options.num_gpus; ++i) {
    xplanes.push_back(std::make_unique<tensorflow::profiler::XPlane>());
  }
  err = tracer.Enable(tracer_options, collector.get(), xplanes);

  vec = SimpleAddSubWithProfiler(kNumElements);

  tracer.Disable();

  // Validate functional correctness - ie, the kernel ran
  EXPECT_EQ(vec.size(), kNumElements);
  EXPECT_THAT(vec, Each(DistanceFrom(0, Lt(0.001))));

  // Expect 4 * elems / (32 elemn / warp) +- 5% double instructions
  // (if they were sampled)
  // Double this as two kernels are sampled (middle kernel is not)
  if (records_fp64 > 0) {
    LOG(INFO) << "Sampled " << total_fp64 << " fp64 instructions";
    double target = kNumElements * 4 * 2 / 32;
    EXPECT_THAT(total_fp64, DistanceFrom(target, Lt(target * 5 / 100)));
  }
  if (records_cycles > 0) {
    LOG(INFO) << "Sampled " << total_cycles << " cycles";
  }
}

}  // namespace
}  // namespace test
}  // namespace profiler
}  // namespace xla
