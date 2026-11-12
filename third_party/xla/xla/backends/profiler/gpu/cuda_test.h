/* Copyright 2021 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_PROFILER_GPU_CUDA_TEST_H_
#define XLA_BACKENDS_PROFILER_GPU_CUDA_TEST_H_

#include <functional>

#include "absl/status/status.h"
#include "absl/time/time.h"
#include "third_party/gpus/cuda/include/driver_types.h"

namespace xla {
namespace profiler {
namespace test {
// Calls a function on the device to print a string as many times as indicated
// by iters.
void PrintfKernel(int iters = 1);

// Calls an empty kernel (named "empty") on the device as many times as
// indicated by iters.
void EmptyKernel(int iters = 1);

// Waits for device activity to complete.
void Synchronize();

// Copies a few bytes of memory from host to device.
void MemCopyH2D();

// Copies a few bytes of memory from device to host, asynchronously.
void MemCopyH2D_Async();

// Copies a few bytes of memory from device to host.
void MemCopyD2H();

// Returns true if it s possible to copy bytes from device 0 to device 1.
bool MemCopyP2PAvailable();

// Copies a few bytes of memory from device 0 to device 1.
void MemCopyP2PImplicit();

// Copies a few bytes of memory from device 0 to device 1.
void MemCopyP2PExplicit();

// Creates a simple cuda graph, clones it, instantiates it and executes it.
void CudaGraphCreateAndExecute();

// Return true if the cuda version is new enough so cuda graph activity trace
// is supported by its CUPTI.
bool IsCudaNewEnoughForGraphTraceTest();

// Captures and instantiates a simple CUDA graph before profiling, launches and
// synchronizes it once before start_profiler(), once between start_profiler()
// and stop_profiler(), and once after stop_profiler(), returning the
// cudaError_t from the post-profiler graph launch and stream synchronization.
cudaError_t ExecuteGraphBeforeDuringAndAfterProfiler(
    const std::function<void()>& start_profiler,
    const std::function<void()>& stop_profiler);

// Builds a parent CUDA graph containing a conditional IF node whose body
// contains a child graph node, then spawns a worker thread that continuously
// recaptures (cudaStreamBeginCaptureToGraph) and updates
// (cudaGraphExecChildGraphNodeSetParams) the child node and launches the parent
// graph while the calling thread executes up to max_profiler_sessions profiling
// cycles via run_single_profiler_session(). Returns the first non-cudaSuccess
// error observed by the worker thread, or cudaSuccess if none occurred.
cudaError_t ExecuteConcurrentConditionalGraphUpdateDuringProfiler(
    int max_profiler_sessions,
    const std::function<bool()>& run_single_profiler_session);

enum class CrossContextWorkloadMode {
  // Worker thread captures and instantiates a CUDA graph on peer context B
  // during each iteration while a peer-waiting kernel runs on context A.
  kStreamCaptureAndGraphLaunch,
  // Worker thread pre-instantiates CUDA graphs on both contexts during warmup
  // and only dispatches cuCtxSetCurrent + cuGraphLaunch during steady-state
  // execution.
  kPreinstantiatedGraphLaunchOnly,
};

// Finalizes/resets profiler state via `reset_profiler_before_context_creation`
// on each attempt so CUPTI has not yet seen the newly created contexts, creates
// two independent CUDA contexts sharing a host-mapped synchronization flag,
// spawns a worker thread that continuously executes a cross-context step
// according to `workload_mode`, and invokes `run_single_profiler_session()` up
// to `max_profiler_sessions` times with a per-session watchdog
// `session_timeout`. Returns `absl::DeadlineExceededError` if a profiler
// session deadlocks with zero worker progress, or `absl::OkStatus()` if all
// sessions finish without deadlocking.
absl::Status ExecuteCrossContextWorkloadDuringProfilerSubscribe(
    int max_profiler_sessions, absl::Duration session_timeout,
    CrossContextWorkloadMode workload_mode,
    const std::function<void()>& reset_profiler_before_context_creation,
    const std::function<bool()>& run_single_profiler_session);

}  // namespace test
}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_CUDA_TEST_H_
