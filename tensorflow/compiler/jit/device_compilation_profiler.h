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

#ifndef TENSORFLOW_COMPILER_JIT_DEVICE_COMPILATION_PROFILER_H_
#define TENSORFLOW_COMPILER_JIT_DEVICE_COMPILATION_PROFILER_H_

#include <cstdint>
#include <string>

#include "tensorflow/compiler/jit/xla_compile_util.h"
#include "tensorflow/core/framework/attr_value.pb.h"

namespace tensorflow {

// Tracks statistics for device compilation and uses these to determine whether
// the given cluster should be compiled or not.
class DeviceCompilationProfiler : public ResourceBase {
 public:
  DeviceCompilationProfiler() = default;
  ~DeviceCompilationProfiler() override;

  struct ClusterCompileStats {
    // Number of times the cluster has been (re-)compiled.
    int64_t compile_count = 0;

    // The number of times this cluster has been executed.
    int64_t execution_count = 0;

    // Cumulative time spent compiling the cluster.
    int64_t cumulative_compile_time_us = 0;

    // True if we have decided that this cluster is too dynamic (i.e. its shapes
    // change too frequently) to profitably JIT compile.  Once a cluster is
    // tagged megamorphic, it stays megamorphic forever.
    bool is_megamorphic = false;

    std::string DebugString() const {
      return absl::StrCat(
          "DeviceCompilationProfiler::ClusterCompileStats {compile_count=",
          compile_count, ", execution_count=", execution_count,
          ", cumulative_compile_time_us=", cumulative_compile_time_us,
          ", is_megamorphic=", is_megamorphic, "}");
    }
  };

  // Returns the compilation statistics for the given cluster.
  absl::StatusOr<ClusterCompileStats> GetCompileStats(
      const NameAttrList& function) const;

  // Determines whether the cluster should be compiled. Creates and inserts an
  // entry into stats (also calls `RegisterExecution`) for `function` if it
  // doesn't already exist.
  virtual bool ShouldCompileCluster(const NameAttrList& function,
                                    DeviceCompileMode compile_mode,
                                    int64_t current_request_count);

  // Registers a cluster execution. Increments the execution count for the given
  // cluster and also determines whether the cluster has gone megamorphic (and
  // sets the megamorphic bit accordingly).
  void RegisterExecution(const NameAttrList& function);

  // Registers a cluster compilation. Increments the compilation count and
  // accumulates the compile time for the given cluster. Also broadcasts an
  // XlaJitCompilationActivity.
  virtual absl::Status RegisterCompilation(const NameAttrList& function,
                                           int64_t compile_time_us,
                                           bool used_persistent_cache);

  void IncrementOngoingAsyncCompilations();
  void DecrementOngoingAsyncCompilations();
  int64_t GetNumOngoingAsyncCompilations() const;
  std::string DebugString() const override;

 private:
  mutable mutex mu_;

  // Maps cluster names to compilation statistics for said cluster.
  absl::flat_hash_map<std::string, ClusterCompileStats> cluster_compile_stats_
      TF_GUARDED_BY(mu_);

  int64_t num_ongoing_compilations_ TF_GUARDED_BY(mu_) = 0;

  DeviceCompilationProfiler(const DeviceCompilationProfiler&) = delete;
  void operator=(const DeviceCompilationProfiler&) = delete;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_DEVICE_COMPILATION_PROFILER_H_
