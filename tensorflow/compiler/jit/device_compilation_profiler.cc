/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/device_compilation_profiler.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/jit/xla_activity.pb.h"
#include "tensorflow/compiler/jit/xla_activity_listener.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/tsl/platform/mutex.h"

namespace tensorflow {
namespace {
bool ShouldBeMegamorphic(int64_t compile_count, int64_t execution_count) {
  const int64_t kCompileThreshold = 10;
  const int64_t kMinExecutionsPerCompile = 50;

  // This heuristic is trying to capture the following property: have we sunk a
  // certain minimum amount of compile time into the cluster that didn't quite
  // "pay off"?
  return compile_count > kCompileThreshold &&
         execution_count < kMinExecutionsPerCompile * compile_count;
}

void RegisterExecutionForCluster(
    const NameAttrList& function,
    DeviceCompilationProfiler::ClusterCompileStats* stats) {
  ++stats->execution_count;

  // The is_megamorphic bit is "sticky".  We assume clusters that have been
  // observed to be megamorphic once stay megamorphic forever.
  if (!stats->is_megamorphic &&
      ShouldBeMegamorphic(stats->compile_count, stats->execution_count)) {
    VLOG(1) << "Marking " << function.name()
            << " as megamorphic, compile_count=" << stats->compile_count
            << " execution_count=" << stats->execution_count;
    stats->is_megamorphic = true;
  }
}

// The number of times a lazy compilation must be requested for a specific
// signature before  we attempt to compile it.
constexpr int64_t kDefaultCompilationThreshold = 2;

// Maximum number of ongoing compilations.
constexpr int64_t kMaxNumOngoingCompilations = kNumAsyncDeviceCompilerThreads;

}  // namespace

StatusOr<DeviceCompilationProfiler::ClusterCompileStats>
DeviceCompilationProfiler::GetCompileStats(const NameAttrList& function) const {
  mutex_lock lock(cluster_compile_stats_mu_);

  if (auto it = cluster_compile_stats_.find(function.name());
      it != cluster_compile_stats_.end()) {
    return it->second;
  }

  return errors::NotFound("Couldn't find compilation stats for cluster: ",
                          function.name());
}

void DeviceCompilationProfiler::RegisterExecution(
    const NameAttrList& function) {
  mutex_lock lock(cluster_compile_stats_mu_);
  auto it =
      cluster_compile_stats_.emplace(function.name(), ClusterCompileStats{})
          .first;
  RegisterExecutionForCluster(function, &it->second);
}

Status DeviceCompilationProfiler::RegisterCompilation(
    const NameAttrList& function, int64_t compile_time_us,
    bool used_persistent_cache) {
  metrics::UpdateXlaCompilationTime(compile_time_us);

  const std::string& function_name = function.name();

  mutex_lock lock(cluster_compile_stats_mu_);
  // Create a stats entry if it doesn't already exist.
  auto it =
      cluster_compile_stats_.emplace(function.name(), ClusterCompileStats{})
          .first;

  const uint64 compile_time_s = compile_time_us / 1.0e6;
  it->second.compile_count++;
  it->second.cumulative_compile_time_us += compile_time_us;
  VLOG(1) << "Compiled " << function_name << " " << it->second.compile_count
          << " times, compile time: " << compile_time_us
          << " us, cumulative: " << it->second.cumulative_compile_time_us
          << " us ("
          << tensorflow::strings::HumanReadableElapsedTime(compile_time_s)
          << " / "
          << tensorflow::strings::HumanReadableElapsedTime(
                 it->second.cumulative_compile_time_us / 1.0e6)
          << ")";

  XlaJitCompilationActivity jit_compilation_activity;
  jit_compilation_activity.set_cluster_name(function_name);
  jit_compilation_activity.set_compile_count(it->second.compile_count);
  jit_compilation_activity.set_compile_time_us(compile_time_us);
  jit_compilation_activity.set_cumulative_compile_time_us(
      it->second.cumulative_compile_time_us);
  jit_compilation_activity.set_used_persistent_cache(used_persistent_cache);
  return BroadcastXlaActivity(std::move(jit_compilation_activity));
}

bool DeviceCompilationProfiler::ShouldCompileCluster(
    const NameAttrList& function, DeviceCompileMode compile_mode,
    int64_t current_request_count) {
  std::optional<int64_t> compile_threshold;
  if (compile_mode == DeviceCompileMode::kLazy) {
    compile_threshold = kDefaultCompilationThreshold;
  } else if (compile_mode == DeviceCompileMode::kAsync) {
    compile_threshold = 0;  // for now, always compile right away.
  }

  if (compile_mode == DeviceCompileMode::kStrict) {
    // Lazy compilation is disabled.
    return true;
  }

  mutex_lock lock(cluster_compile_stats_mu_);
  // Create a stats entry if one isn't found and register an execution.
  // Determine eligibility assuming this is the first execution of the cluster
  // and this cluster has never been compiled before.
  auto [it, cluster_not_found] =
      cluster_compile_stats_.emplace(function.name(), ClusterCompileStats{});
  if (cluster_not_found) {
    RegisterExecutionForCluster(function, &it->second);
  }

  // We avoid compiling clusters that have "gone megamorphic" i.e. have an
  // excessive amount of shape dynamism.
  if (it->second.is_megamorphic) {
    BroadcastOptimizationRemark(XlaOptimizationRemark::MEGAMORPHIC_FUNCTION,
                                function.name())
        .IgnoreError();
    VLOG(2) << "Not compiling cluster " << function.name()
            << " because it is megamorphic.";
    return false;
  }

  // We always compile a cluster the very first time it is executed.  This is an
  // optimistic guess that pays off for statically shaped TensorFlow graphs
  // (since they get the benefit of XLA right away without waiting for warmup)
  // and doesn't hurt much for dynamically shaped TensorFlow graphs (we "pay" at
  // most one cluster-compilation's worth of compile time).
  if (it->second.execution_count == 1) {
    return true;
  }

  if (compile_mode == DeviceCompileMode::kAsync) {
    // Asynchronous compilation is enabled.
    mutex_lock lock(async_compilation_state_.async_compilation_state_mu);
    if (async_compilation_state_.num_ongoing_compilations >=
        kMaxNumOngoingCompilations) {
      VLOG(2) << "Not asynchronously compiling cluster " << function.name()
              << " because of too many ongoing compilations.";
      return false;
    }
  }

  bool reached_compile_threshold = current_request_count >= *compile_threshold;
  if (!reached_compile_threshold) {
    VLOG(2) << "Not compiling cluster " << function.name()
            << " because it has not reached compile threshold; threshold is "
            << *compile_threshold << " execution count "
            << current_request_count << ".";
  }
  return reached_compile_threshold;
}

void DeviceCompilationProfiler::IncrementOngoingAsyncCompilations() {
  mutex_lock lock(async_compilation_state_.async_compilation_state_mu);
  async_compilation_state_.num_ongoing_compilations++;
}

void DeviceCompilationProfiler::DecrementOngoingAsyncCompilations() {
  mutex_lock lock(async_compilation_state_.async_compilation_state_mu);
  async_compilation_state_.num_ongoing_compilations--;
}

int64_t DeviceCompilationProfiler::GetNumOngoingAsyncCompilations() const {
  mutex_lock lock(async_compilation_state_.async_compilation_state_mu);
  return async_compilation_state_.num_ongoing_compilations;
}

std::string DeviceCompilationProfiler::DebugString() const {
  std::string debug_string =
      "DeviceCompilationProfiler {\ncluster_compile_stats: {\n";
  {
    mutex_lock lock(cluster_compile_stats_mu_);

    for (const auto& [key, stats] : cluster_compile_stats_) {
      absl::StrAppend(&debug_string, key, ": ", stats.DebugString(), "\n");
    }
  }

  absl::StrAppend(&debug_string, "}\nnum_ongoing_compilations=",
                  GetNumOngoingAsyncCompilations(), "\n}\n");

  return debug_string;
}

}  // namespace tensorflow
