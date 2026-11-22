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

// Creates some GPU activity to test functionalities of gpuperfcounter/gputrace.
#include "third_party/gpus/cuda/include/cuda.h"

#include <stdio.h>

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"
#include "absl/debugging/leak_check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "third_party/gpus/cuda/include/driver_types.h"
#include "xla/backends/profiler/gpu/cuda_test.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "tsl/profiler/lib/scoped_annotation.h"

namespace xla {
namespace profiler {
namespace test {

using tsl::profiler::ScopedAnnotation;

namespace {

// Simple printf kernel.
__global__ void simple_print() { printf("hello, world!\n"); }

// Empty kernel.
__global__ void empty() {}

// Simple kernel accesses memory.
__global__ void access(int* addr) { *addr = *addr * 2; }

unsigned* g_device_copy;

unsigned *gpu0_buf, *gpu1_buf;

}  // namespace

void PrintfKernel(int iters) {
  for (int i = 0; i < iters; ++i) {
    simple_print<<<1, 1>>>();
  }
}

void EmptyKernel(int iters) {
  for (int i = 0; i < iters; ++i) {
    empty<<<1, 1>>>();
  }
}

void AccessKernel(int* addr) { access<<<1, 1>>>(addr); }

void Synchronize() { cudaDeviceSynchronize(); }

void UnifiedMemoryHtoDAndDtoH() {
  int* addr = nullptr;
  cudaMallocManaged(reinterpret_cast<void**>(&addr), sizeof(int));
  // The page is now in host memory.
  *addr = 1;
  // The kernel wants to access the page. HtoD transfer happens.
  AccessKernel(addr);
  Synchronize();
  // The page is now in device memory. CPU wants to access the page. DtoH
  // transfer happens.
  EXPECT_EQ(*addr, 2);
}

void MemCopyH2D() {
  unsigned host_val = 0x12345678;
  cudaMalloc(reinterpret_cast<void**>(&g_device_copy), sizeof(unsigned));
  cudaMemcpy(g_device_copy, &host_val, sizeof(unsigned),
             cudaMemcpyHostToDevice);
}

void MemCopyH2D_Async() {
  unsigned host_val = 0x12345678;
  cudaMalloc(reinterpret_cast<void**>(&g_device_copy), sizeof(unsigned));
  cudaMemcpyAsync(g_device_copy, &host_val, sizeof(unsigned),
                  cudaMemcpyHostToDevice);
}

void MemCopyD2H() {
  unsigned host_val = 0;
  cudaMalloc(reinterpret_cast<void**>(&g_device_copy), sizeof(unsigned));
  cudaMemcpy(&host_val, g_device_copy, sizeof(unsigned),
             cudaMemcpyDeviceToHost);
}

namespace {

// Helper function to set up memory buffers on two devices.
void P2PMemcpyHelper() {
  cudaSetDevice(0);
  cudaMalloc(reinterpret_cast<void**>(&gpu0_buf), sizeof(unsigned));
  cudaDeviceEnablePeerAccess(/*peerDevice=*/1, /*flags=*/0);
  cudaSetDevice(1);
  cudaMalloc(reinterpret_cast<void**>(&gpu1_buf), sizeof(unsigned));
  cudaDeviceEnablePeerAccess(/*peerDevice=*/0, /*flags=*/0);
}

}  // namespace

bool MemCopyP2PAvailable() {
  int can_access_01 = 0;
  cudaDeviceCanAccessPeer(&can_access_01, /*device=*/0, /*peerDevice=*/1);
  int can_access_10 = 0;
  cudaDeviceCanAccessPeer(&can_access_01, /*device=*/1, /*peerDevice=*/0);
  return can_access_01 && can_access_10;
}

void MemCopyP2PImplicit() {
  P2PMemcpyHelper();
  cudaMemcpy(gpu1_buf, gpu0_buf, sizeof(unsigned), cudaMemcpyDefault);
}

void MemCopyP2PExplicit() {
  P2PMemcpyHelper();
  cudaMemcpyPeer(gpu1_buf, 1 /* device */, gpu0_buf, 0 /* device */,
                 sizeof(unsigned));
}

// The test about cuda graph is based on Nvidia's CUPTI sample code
// under extras/CUPTI/samples/cuda_graphs_trace/ dir of CUDA distribution.
__global__ void VecAdd(const int* a, const int* b, int* c, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

__global__ void VecSub(const int* a, const int* b, int* c, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) c[i] = a[i] - b[i];
}

// Adds a graph node through the polymorphic cudaGraphAddNode API instead of
// the type-specific cudaGraphAdd*Node entry points. CUDA 13 folded the
// dependencyData argument of cudaGraphAddNode_v2 into cudaGraphAddNode.
static cudaError_t AddGraphNode(cudaGraphNode_t* node, cudaGraph_t graph,
                                const cudaGraphNode_t* dependencies,
                                size_t num_dependencies,
                                cudaGraphNodeParams* node_params) {
#if CUDART_VERSION >= 13000
  return cudaGraphAddNode(node, graph, dependencies,
                          /*dependencyData=*/nullptr, num_dependencies,
                          node_params);
#else
  return cudaGraphAddNode_v2(node, graph, dependencies,
                             /*dependencyData=*/nullptr, num_dependencies,
                             node_params);
#endif
}

void CudaGraphCreateAndExecute() {
  constexpr size_t kNumElements = 2048;
  constexpr size_t kNumBytes = kNumElements * sizeof(int);
  constexpr int kThreadsPerBlock = 256;
  int blocks_per_grid = 0;

  cudaStream_t stream = nullptr;
  cudaGraphNodeParams kernel_params{};
  kernel_params.type = cudaGraphNodeTypeKernel;
  cudaGraphNodeParams memcpy_node_params{};
  memcpy_node_params.type = cudaGraphNodeTypeMemcpy;
  cudaMemcpy3DParms& memcpy_params = memcpy_node_params.memcpy.copyParams;
  cudaGraph_t graph;
  cudaGraph_t cloned_graph;
  cudaGraphExec_t graph_exec;
  cudaGraphNode_t nodes[5];

  // Allocates input/output vectors in host memory.
  std::vector<int> vec_a(kNumElements);
  std::vector<int> vec_b(kNumElements);
  std::vector<int> vec_c(kNumElements);

  // Allocates vectors in device memory.
  int *d_a, *d_b, *d_c;
  cudaMalloc((void**)&d_a, kNumBytes);
  cudaMalloc((void**)&d_b, kNumBytes);
  cudaMalloc((void**)&d_c, kNumBytes);

  cudaGraphCreate(&graph, 0);

  // Init memcpy params.
  memcpy_params.kind = cudaMemcpyHostToDevice;
  memcpy_params.srcPtr.ptr = vec_a.data();
  memcpy_params.dstPtr.ptr = d_a;
  memcpy_params.extent.width = kNumBytes;
  memcpy_params.extent.height = 1;
  memcpy_params.extent.depth = 1;
  {
    ScopedAnnotation memcpy_1_annotation(
        "Thunk:#name=my_module/prep,hlo_op=memcpy.1#");
    AddGraphNode(&nodes[0], graph, nullptr, 0, &memcpy_node_params);
  }

  memcpy_params.srcPtr.ptr = vec_b.data();
  memcpy_params.dstPtr.ptr = d_b;
  {
    ScopedAnnotation memcpy_2_annotation(
        "Thunk:#name=my_module/prep,hlo_op=memcpy.2#");
    AddGraphNode(&nodes[1], graph, nullptr, 0, &memcpy_node_params);
  }

  // Init kernel params.
  int num = kNumElements;
  void* kernelArgs[] = {(void*)&d_a, (void*)&d_b, (void*)&d_c, (void*)&num};
  blocks_per_grid = (kNumElements + kThreadsPerBlock - 1) / kThreadsPerBlock;
  kernel_params.kernel.func = (void*)VecAdd;
  kernel_params.kernel.gridDim = dim3(blocks_per_grid, 1, 1);
  kernel_params.kernel.blockDim = dim3(kThreadsPerBlock, 1, 1);
  kernel_params.kernel.sharedMemBytes = 0;
  kernel_params.kernel.kernelParams = (void**)kernelArgs;
  kernel_params.kernel.extra = nullptr;
  {
    ScopedAnnotation add_1_annotation(
        "Thunk:#name=my_module/body,hlo_op=add.1#");
    AddGraphNode(&nodes[2], graph, &nodes[0], 2, &kernel_params);
  }

  kernel_params.kernel.func = (void*)VecSub;
  {
    ScopedAnnotation sub_1_annotation(
        "Thunk:#name=my_module/body,hlo_op=sub.1#");
    AddGraphNode(&nodes[3], graph, &nodes[2], 1, &kernel_params);
  }
  memcpy_params.kind = cudaMemcpyDeviceToHost;
  memcpy_params.srcPtr.ptr = d_c;
  memcpy_params.dstPtr.ptr = vec_c.data();
  memcpy_params.extent.width = kNumBytes;
  memcpy_params.extent.height = 1;
  memcpy_params.extent.depth = 1;
  {
    ScopedAnnotation memcpy_3_annotation(
        "Thunk:#name=my_module/post,hlo_op=memcpy.3#");
    AddGraphNode(&nodes[4], graph, &nodes[3], 1, &memcpy_node_params);
  }
  cudaGraphClone(&cloned_graph, graph);

  cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);

  {
    ScopedAnnotation module_annotation("Thunk:#name=my_module,hlo_op=call.1#");
    cudaGraphLaunch(graph_exec, stream);
  }

  cudaStreamSynchronize(stream);

  cudaFree(d_c);
  cudaFree(d_b);
  cudaFree(d_a);
}

bool IsCudaNewEnoughForGraphTraceTest() { return true; }

namespace {

__global__ void SetCondKernel(cudaGraphConditionalHandle handle) {
  cudaGraphSetConditional(handle, 1);
}

__global__ void ChildBodyKernel(int* out) {
  for (int i = 0; i < 256; ++i) {
    out[i] += 1;
  }
}

}  // namespace

cudaError_t ExecuteGraphBeforeDuringAndAfterProfiler(
    const std::function<void()>& start_profiler,
    const std::function<void()>& stop_profiler) {
  cudaStream_t stream = nullptr;
  cudaError_t err = cudaStreamCreate(&stream);
  if (err != cudaSuccess) return err;

  cudaGraph_t graph = nullptr;
  err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  if (err != cudaSuccess) return err;
  empty<<<1, 1, 0, stream>>>();
  err = cudaStreamEndCapture(stream, &graph);
  if (err != cudaSuccess) return err;

  cudaGraphExec_t graph_exec = nullptr;
  err = cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);
  if (err != cudaSuccess) return err;

  // Launch before profiling starts.
  err = cudaGraphLaunch(graph_exec, stream);
  if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) return err;

  start_profiler();

  // Launch while profiling is active (CUPTI instruments the graph exec).
  err = cudaGraphLaunch(graph_exec, stream);
  if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    stop_profiler();
    return err;
  }

  stop_profiler();

  // Launch after profiling stops (if cuptiFinalize() was called, the
  // CUPTI-instrumented graph_exec faults with cudaErrorIllegalAddress = 700).
  cudaError_t post_err = cudaGraphLaunch(graph_exec, stream);
  if (post_err == cudaSuccess) {
    post_err = cudaStreamSynchronize(stream);
  }
  if (post_err == cudaSuccess) {
    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
  }
  return post_err;
}

cudaError_t ExecuteConcurrentConditionalGraphUpdateDuringProfiler(
    int max_profiler_sessions,
    const std::function<bool()>& run_single_profiler_session) {
  int* d_out = nullptr;
  cudaError_t err =
      cudaMalloc(reinterpret_cast<void**>(&d_out), 256 * sizeof(int));
  if (err != cudaSuccess) return err;
  err = cudaMemset(d_out, 0, 256 * sizeof(int));
  if (err != cudaSuccess) return err;

  cudaGraph_t parent_graph = nullptr;
  err = cudaGraphCreate(&parent_graph, 0);
  if (err != cudaSuccess) return err;

  cudaGraphConditionalHandle handle{};
  err = cudaGraphConditionalHandleCreate(&handle, parent_graph, 1,
                                         cudaGraphCondAssignDefault);
  if (err != cudaSuccess) return err;

  cudaGraphNodeParams setter_params{};
  setter_params.type = cudaGraphNodeTypeKernel;
  void* setter_args[] = {&handle};
  setter_params.kernel.func = reinterpret_cast<void*>(SetCondKernel);
  setter_params.kernel.gridDim = dim3(1, 1, 1);
  setter_params.kernel.blockDim = dim3(1, 1, 1);
  setter_params.kernel.kernelParams = setter_args;
  cudaGraphNode_t setter_node = nullptr;
  err = AddGraphNode(&setter_node, parent_graph, nullptr, 0, &setter_params);
  if (err != cudaSuccess) return err;

  cudaGraphNodeParams cond_params{};
  cond_params.type = cudaGraphNodeTypeConditional;
  cond_params.conditional.handle = handle;
  cond_params.conditional.type = cudaGraphCondTypeIf;
  cond_params.conditional.size = 1;
  cudaGraphNode_t cond_node = nullptr;
  err = AddGraphNode(&cond_node, parent_graph, &setter_node, 1, &cond_params);
  if (err != cudaSuccess) return err;
  cudaGraph_t body_graph = cond_params.conditional.phGraph_out[0];

  cudaStream_t stream = nullptr;
  err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (err != cudaSuccess) return err;

  cudaGraph_t init_child = nullptr;
  err = cudaGraphCreate(&init_child, 0);
  if (err != cudaSuccess) return err;
  err = cudaStreamBeginCaptureToGraph(stream, init_child, nullptr, nullptr, 0,
                                      cudaStreamCaptureModeThreadLocal);
  if (err != cudaSuccess) return err;
  ChildBodyKernel<<<1, 32, 0, stream>>>(d_out);
  err = cudaStreamEndCapture(stream, &init_child);
  if (err != cudaSuccess) return err;

  cudaGraphNode_t body_child_node = nullptr;
  err = cudaGraphAddChildGraphNode(&body_child_node, body_graph, nullptr, 0,
                                   init_child);
  if (err != cudaSuccess) return err;
  cudaGraphDestroy(init_child);

  cudaGraphExec_t parent_exec = nullptr;
  err = cudaGraphInstantiate(&parent_exec, parent_graph, nullptr, nullptr, 0);
  if (err != cudaSuccess) return err;

  std::atomic<bool> running{true};
  std::atomic<cudaError_t> worker_err{cudaSuccess};

  std::thread worker([&]() {
    cudaSetDevice(0);
    while (running.load(std::memory_order_relaxed)) {
      cudaGraph_t updated_child = nullptr;
      cudaError_t step_err = cudaGraphCreate(&updated_child, 0);
      if (step_err != cudaSuccess) {
        worker_err.store(step_err, std::memory_order_relaxed);
        break;
      }
      step_err =
          cudaStreamBeginCaptureToGraph(stream, updated_child, nullptr, nullptr,
                                        0, cudaStreamCaptureModeThreadLocal);
      if (step_err != cudaSuccess) {
        cudaGraphDestroy(updated_child);
        worker_err.store(step_err, std::memory_order_relaxed);
        break;
      }
      ChildBodyKernel<<<1, 32, 0, stream>>>(d_out);
      step_err = cudaStreamEndCapture(stream, &updated_child);
      if (step_err != cudaSuccess) {
        cudaGraphDestroy(updated_child);
        worker_err.store(step_err, std::memory_order_relaxed);
        break;
      }
      step_err = cudaGraphExecChildGraphNodeSetParams(
          parent_exec, body_child_node, updated_child);
      cudaGraphDestroy(updated_child);
      if (step_err != cudaSuccess) {
        worker_err.store(step_err, std::memory_order_relaxed);
        break;
      }
      step_err = cudaGraphLaunch(parent_exec, stream);
      if (step_err != cudaSuccess) {
        worker_err.store(step_err, std::memory_order_relaxed);
        break;
      }
      step_err = cudaStreamSynchronize(stream);
      if (step_err != cudaSuccess) {
        worker_err.store(step_err, std::memory_order_relaxed);
        break;
      }
    }
  });

  for (int i = 0; i < max_profiler_sessions &&
                  worker_err.load(std::memory_order_relaxed) == cudaSuccess;
       ++i) {
    if (!run_single_profiler_session()) {
      break;
    }
  }

  running.store(false, std::memory_order_relaxed);
  worker.join();

  cudaError_t final_err = worker_err.load(std::memory_order_relaxed);
  if (final_err == cudaSuccess) {
    cudaGraphExecDestroy(parent_exec);
    cudaGraphDestroy(parent_graph);
    cudaStreamDestroy(stream);
    cudaFree(d_out);
  }
  return final_err;
}

namespace {

// Models two ranks of a cross-context / multi-GPU step:
//   - WaitPeerRankKernel: runs on context A, polling mapped flag *flag != 0 up
//     to a clock64() safety cap (safety_limit_cycles).
//   - SignalPeerRankKernel: runs on context B, writing *flag = 1.
__global__ void WaitPeerRankKernel(volatile uint32_t* flag,
                                   uint64_t safety_limit_cycles) {
  uint64_t start_cycles = static_cast<uint64_t>(clock64());
  while (*flag == 0 && static_cast<uint64_t>(clock64()) - start_cycles <
                           safety_limit_cycles) {
  }
}

__global__ void SignalPeerRankKernel(volatile uint32_t* flag) { *flag = 1; }

absl::string_view CrossContextWorkerStepName(int step) {
  switch (step) {
    case 0:
      return "init";
    case 1:
      return "ctx_a_kernel_running_and_ctx_b_cuStreamBeginCapture";
    case 2:
      return "ctx_b_cuLaunchKernel(signal_peer_rank_on_capturing_stream)";
    case 3:
      return "ctx_b_cuStreamEndCapture_and_cuGraphLaunch";
    case 4:
      return "cuStreamSynchronize";
    case 5:
      return "ctx_a_cuGraphLaunch_running_before_ctx_b_cuGraphLaunch";
    case 6:
      return "ctx_b_cuCtxSetCurrent_and_cuGraphLaunch(preinstantiated_exec)";
    default:
      return "unknown";
  }
}

CUresult FinishCaptureAndInstantiateSingleKernelGraph(CUstream stream,
                                                      CUfunction fn,
                                                      void** kernel_args,
                                                      CUgraphExec* out_exec) {
  *out_exec = nullptr;
  CUresult launch_err =
      cuLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, stream, kernel_args, nullptr);
  CUgraph graph = nullptr;
  CUresult end_err = cuStreamEndCapture(stream, &graph);
  CUresult err = (launch_err != CUDA_SUCCESS) ? launch_err : end_err;
  if (err == CUDA_SUCCESS) {
    err = cuGraphInstantiateWithFlags(out_exec, graph, 0);
  }
  if (graph != nullptr) {
    cuGraphDestroy(graph);
  }
  return err;
}

CUresult CaptureAndInstantiateSingleKernelGraph(CUstream stream, CUfunction fn,
                                                void** kernel_args,
                                                CUgraphExec* out_exec) {
  *out_exec = nullptr;
  CUresult err =
      cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL);
  if (err != CUDA_SUCCESS) {
    return err;
  }
  return FinishCaptureAndInstantiateSingleKernelGraph(stream, fn, kernel_args,
                                                      out_exec);
}

struct CrossContextAttemptState {
  CUcontext ctx_a = nullptr;
  CUcontext ctx_b = nullptr;
  void* host_flag = nullptr;
  CUdeviceptr dev_flag_a = 0;
  CUdeviceptr dev_flag_b = 0;
  CUfunction fn_wait_a = nullptr;
  CUfunction fn_signal_b = nullptr;
  CUstream stream_a = nullptr;
  uint64_t safety_limit_cycles = 0;

  std::atomic<bool> stop{false};
  std::atomic<int> worker_step{0};
  std::atomic<uint64_t> completed_iters{0};
  std::atomic<CUresult> worker_cu_err{CUDA_SUCCESS};
  std::atomic<bool> keep_going{true};

  absl::Notification warmup_done;
  absl::Notification worker_done;
  absl::Notification session_done;
};

void RunCrossContextWorker(std::shared_ptr<CrossContextAttemptState> state,
                           CrossContextWorkloadMode workload_mode) {
  absl::LeakCheckDisabler leak_disabler;
  auto fail_worker = [&](CUresult err) {
    state->worker_cu_err.store(err, std::memory_order_release);
    if (!state->warmup_done.HasBeenNotified()) {
      state->warmup_done.Notify();
    }
    state->worker_done.Notify();
  };

  CUstream stream_cap_b = nullptr;
  CUstream stream_run_b = nullptr;
  CUresult err = cuCtxSetCurrent(state->ctx_b);
  if (err != CUDA_SUCCESS) {
    fail_worker(err);
    return;
  }
  err = cuStreamCreate(&stream_cap_b, CU_STREAM_NON_BLOCKING);
  if (err != CUDA_SUCCESS) {
    fail_worker(err);
    return;
  }
  err = cuStreamCreate(&stream_run_b, CU_STREAM_NON_BLOCKING);
  if (err != CUDA_SUCCESS) {
    cuStreamDestroy(stream_cap_b);
    fail_worker(err);
    return;
  }

  void* wait_args[] = {&state->dev_flag_a, &state->safety_limit_cycles};
  void* sig_args[] = {&state->dev_flag_b};
  const bool graph_only =
      (workload_mode ==
       CrossContextWorkloadMode::kPreinstantiatedGraphLaunchOnly);

  CUgraphExec exec_wait_a = nullptr;
  CUgraphExec exec_sig_b = nullptr;
  absl::Cleanup worker_resources_cleanup = [&]() {
    if (exec_sig_b != nullptr) {
      cuCtxSetCurrent(state->ctx_b);
      cuGraphExecDestroy(exec_sig_b);
    }
    if (exec_wait_a != nullptr) {
      cuCtxSetCurrent(state->ctx_a);
      cuGraphExecDestroy(exec_wait_a);
    }
    cuCtxSetCurrent(state->ctx_b);
    cuStreamDestroy(stream_run_b);
    cuStreamDestroy(stream_cap_b);
  };

  if (graph_only) {
    err = cuCtxSetCurrent(state->ctx_a);
    if (err == CUDA_SUCCESS) {
      err = CaptureAndInstantiateSingleKernelGraph(
          state->stream_a, state->fn_wait_a, wait_args, &exec_wait_a);
    }
    if (err == CUDA_SUCCESS) {
      err = cuCtxSetCurrent(state->ctx_b);
    }
    if (err == CUDA_SUCCESS) {
      err = CaptureAndInstantiateSingleKernelGraph(
          stream_cap_b, state->fn_signal_b, sig_args, &exec_sig_b);
    }
    if (err != CUDA_SUCCESS) {
      fail_worker(err);
      return;
    }
  }

  while (!state->stop.load(std::memory_order_acquire)) {
    *static_cast<volatile uint32_t*>(state->host_flag) = 0;

    err = cuCtxSetCurrent(state->ctx_a);
    if (err == CUDA_SUCCESS) {
      err = graph_only ? cuGraphLaunch(exec_wait_a, state->stream_a)
                       : cuLaunchKernel(state->fn_wait_a, 1, 1, 1, 1, 1, 1, 0,
                                        state->stream_a, wait_args, nullptr);
    }
    if (!graph_only) {
      if (err == CUDA_SUCCESS) {
        err = cuCtxSetCurrent(state->ctx_b);
      }
      if (err == CUDA_SUCCESS) {
        err = cuStreamBeginCapture(stream_cap_b,
                                   CU_STREAM_CAPTURE_MODE_THREAD_LOCAL);
      }
    }
    if (err != CUDA_SUCCESS) {
      state->worker_cu_err.store(err, std::memory_order_release);
      break;
    }
    state->worker_step.store(graph_only ? 5 : 1, std::memory_order_release);

    absl::SleepFor(absl::Microseconds(200));

    CUgraphExec iter_exec = nullptr;
    if (graph_only) {
      state->worker_step.store(6, std::memory_order_release);
      err = cuCtxSetCurrent(state->ctx_b);
      if (err == CUDA_SUCCESS) {
        err = cuGraphLaunch(exec_sig_b, stream_run_b);
      }
    } else {
      state->worker_step.store(2, std::memory_order_release);
      err = FinishCaptureAndInstantiateSingleKernelGraph(
          stream_cap_b, state->fn_signal_b, sig_args, &iter_exec);
      if (err == CUDA_SUCCESS) {
        state->worker_step.store(3, std::memory_order_release);
        err = cuGraphLaunch(iter_exec, stream_run_b);
      }
    }
    if (err != CUDA_SUCCESS) {
      if (iter_exec != nullptr) {
        cuGraphExecDestroy(iter_exec);
      }
      state->worker_cu_err.store(err, std::memory_order_release);
      break;
    }

    state->worker_step.store(4, std::memory_order_release);
    err = cuStreamSynchronize(stream_run_b);
    if (err == CUDA_SUCCESS) {
      err = cuCtxSetCurrent(state->ctx_a);
    }
    if (err == CUDA_SUCCESS) {
      err = cuStreamSynchronize(state->stream_a);
    }
    if (iter_exec != nullptr) {
      cuCtxSetCurrent(state->ctx_b);
      cuGraphExecDestroy(iter_exec);
    }
    if (err != CUDA_SUCCESS) {
      state->worker_cu_err.store(err, std::memory_order_release);
      break;
    }

    uint64_t completed =
        state->completed_iters.fetch_add(1, std::memory_order_acq_rel) + 1;
    if (completed == 2 && !state->warmup_done.HasBeenNotified()) {
      state->warmup_done.Notify();
    }
    if (graph_only) {
      absl::SleepFor(absl::Microseconds(250));
    }
  }

  if (!state->warmup_done.HasBeenNotified()) {
    state->warmup_done.Notify();
  }
  state->worker_done.Notify();
}

}  // namespace

absl::Status ExecuteCrossContextWorkloadDuringProfilerSubscribe(
    int max_profiler_sessions, absl::Duration session_timeout,
    CrossContextWorkloadMode workload_mode,
    const std::function<void()>& reset_profiler_before_context_creation,
    const std::function<bool()>& run_single_profiler_session) {
  absl::LeakCheckDisabler leak_disabler;
  ABSL_RETURN_IF_ERROR(stream_executor::cuda::ToStatus(cuInit(0), "cuInit(0)"));

  int driver_version = 0;
  ABSL_RETURN_IF_ERROR(stream_executor::cuda::ToStatus(
      cuDriverGetVersion(&driver_version), "cuDriverGetVersion"));
  if (driver_version < 12050) {
    return absl::CancelledError(absl::StrFormat(
        "cuCtxCreate_v4 requires CUDA driver 12.5+, found %d", driver_version));
  }

  int num_devs = 0;
  ABSL_RETURN_IF_ERROR(stream_executor::cuda::ToStatus(cuDeviceGetCount(&num_devs),
                                                  "cuDeviceGetCount"));
  if (num_devs < 1) {
    return absl::CancelledError("No CUDA devices available");
  }

  CUcontext saved_ctx = nullptr;
  cuCtxGetCurrent(&saved_ctx);
  absl::Cleanup restore_ctx = [saved_ctx]() { cuCtxSetCurrent(saved_ctx); };

  CUdevice dev_a = 0;
  CUdevice dev_b = 0;
  ABSL_RETURN_IF_ERROR(stream_executor::cuda::ToStatus(cuDeviceGet(&dev_a, 0),
                                                  "cuDeviceGet(0)"));
  ABSL_RETURN_IF_ERROR(stream_executor::cuda::ToStatus(
      cuDeviceGet(&dev_b, (num_devs > 1) ? 1 : 0), "cuDeviceGet(peer)"));

  int clock_rate_khz = 0;
  absl::Status clock_status = stream_executor::cuda::ToStatus(
      cuDeviceGetAttribute(&clock_rate_khz, CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
                           dev_a),
      "cuDeviceGetAttribute(CLOCK_RATE)");
  if (!clock_status.ok() || clock_rate_khz <= 0) {
    clock_rate_khz = 2000000;
  }

  cudaKernel_t wait_kernel = nullptr;
  ABSL_RETURN_IF_ERROR(stream_executor::cuda::ToStatus(
      cudaGetKernel(&wait_kernel,
                    reinterpret_cast<const void*>(WaitPeerRankKernel)),
      "cudaGetKernel(WaitPeerRankKernel)"));
  cudaKernel_t signal_kernel = nullptr;
  ABSL_RETURN_IF_ERROR(stream_executor::cuda::ToStatus(
      cudaGetKernel(&signal_kernel,
                    reinterpret_cast<const void*>(SignalPeerRankKernel)),
      "cudaGetKernel(SignalPeerRankKernel)"));

  for (int attempt = 1; attempt <= max_profiler_sessions; ++attempt) {
    if (reset_profiler_before_context_creation) {
      cuCtxSetCurrent(nullptr);
      reset_profiler_before_context_creation();
    }

    auto state = std::make_shared<CrossContextAttemptState>();
    state->safety_limit_cycles =
        static_cast<uint64_t>(
            absl::ToInt64Milliseconds(session_timeout + absl::Seconds(3))) *
        static_cast<uint64_t>(clock_rate_khz);

    absl::Cleanup state_cleanup = [state]() {
      if (state->ctx_b != nullptr) {
        cuCtxSetCurrent(state->ctx_b);
        cuCtxDestroy(state->ctx_b);
      }
      if (state->ctx_a != nullptr) {
        cuCtxSetCurrent(state->ctx_a);
        if (state->stream_a != nullptr) {
          cuStreamDestroy(state->stream_a);
        }
        if (state->host_flag != nullptr) {
          cuMemFreeHost(state->host_flag);
        }
        cuCtxDestroy(state->ctx_a);
      }
    };

    ABSL_RETURN_IF_ERROR(stream_executor::cuda::ToStatus(
        cuCtxCreate_v4(&state->ctx_a, /*ctxCreateParams=*/nullptr, /*flags=*/0,
                       dev_a),
        "cuCtxCreate_v4(ctx_a)"));
    ABSL_RETURN_IF_ERROR(stream_executor::cuda::ToStatus(
        cuCtxCreate_v4(&state->ctx_b, /*ctxCreateParams=*/nullptr, /*flags=*/0,
                       dev_b),
        "cuCtxCreate_v4(ctx_b)"));
    ABSL_RETURN_IF_ERROR(stream_executor::cuda::ToStatus(
        cuCtxSetCurrent(state->ctx_a), "cuCtxSetCurrent(ctx_a)"));
    ABSL_RETURN_IF_ERROR(stream_executor::cuda::ToStatus(
        cuMemHostAlloc(&state->host_flag, sizeof(uint32_t),
                       CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP),
        "cuMemHostAlloc"));
    ABSL_RETURN_IF_ERROR(stream_executor::cuda::ToStatus(
        cuMemHostGetDevicePointer(&state->dev_flag_a, state->host_flag, 0),
        "cuMemHostGetDevicePointer(ctx_a)"));
    ABSL_RETURN_IF_ERROR(stream_executor::cuda::ToStatus(
        cuKernelGetFunction(&state->fn_wait_a, wait_kernel),
        "cuKernelGetFunction(wait_kernel)"));
    ABSL_RETURN_IF_ERROR(stream_executor::cuda::ToStatus(
        cuStreamCreate(&state->stream_a, CU_STREAM_NON_BLOCKING),
        "cuStreamCreate(stream_a)"));
    ABSL_RETURN_IF_ERROR(stream_executor::cuda::ToStatus(
        cuCtxSetCurrent(state->ctx_b), "cuCtxSetCurrent(ctx_b)"));
    ABSL_RETURN_IF_ERROR(stream_executor::cuda::ToStatus(
        cuMemHostGetDevicePointer(&state->dev_flag_b, state->host_flag, 0),
        "cuMemHostGetDevicePointer(ctx_b)"));
    ABSL_RETURN_IF_ERROR(stream_executor::cuda::ToStatus(
        cuKernelGetFunction(&state->fn_signal_b, signal_kernel),
        "cuKernelGetFunction(signal_kernel)"));

    std::thread worker(RunCrossContextWorker, state, workload_mode);
    if (!state->warmup_done.WaitForNotificationWithTimeout(absl::Seconds(10))) {
      state->stop.store(true, std::memory_order_release);
      if (state->worker_done.WaitForNotificationWithTimeout(absl::Seconds(5))) {
        worker.join();
      } else {
        worker.detach();
        std::move(state_cleanup).Cancel();
      }
      return absl::DeadlineExceededError(
          absl::StrFormat("Worker warmup timed out on attempt %d", attempt));
    }

    CUresult warmup_err = state->worker_cu_err.load(std::memory_order_acquire);
    if (warmup_err != CUDA_SUCCESS) {
      state->stop.store(true, std::memory_order_release);
      worker.join();
      return stream_executor::cuda::ToStatus(warmup_err, "worker warmup");
    }

    absl::SleepFor(absl::Microseconds((attempt * 137) % 500));

    std::thread profiler_thread(
        [state, session_fn = run_single_profiler_session]() {
          absl::LeakCheckDisabler thread_leak_disabler;
          state->keep_going.store(session_fn(), std::memory_order_release);
          state->session_done.Notify();
        });

    bool finished =
        state->session_done.WaitForNotificationWithTimeout(session_timeout);
    bool deadlocked = false;
    int deadlocked_step = 0;
    uint64_t deadlocked_iters = 0;

    for (int extra_window = 0; !finished && extra_window < 2; ++extra_window) {
      uint64_t it0 = state->completed_iters.load(std::memory_order_acquire);
      if (state->session_done.WaitForNotificationWithTimeout(
              absl::Milliseconds(500))) {
        finished = true;
        break;
      }
      uint64_t it1 = state->completed_iters.load(std::memory_order_acquire);
      if (it1 == it0) {
        deadlocked = true;
        deadlocked_step = state->worker_step.load(std::memory_order_acquire);
        deadlocked_iters = it1;
        break;
      }
    }
    if (!finished && !deadlocked) {
      deadlocked = true;
      deadlocked_step = state->worker_step.load(std::memory_order_acquire);
      deadlocked_iters = state->completed_iters.load(std::memory_order_acquire);
    }

    state->stop.store(true, std::memory_order_release);
    bool prof_joined =
        state->session_done.WaitForNotificationWithTimeout(absl::Seconds(6));
    bool worker_joined =
        state->worker_done.WaitForNotificationWithTimeout(absl::Seconds(6));
    if (prof_joined) {
      profiler_thread.join();
    } else {
      profiler_thread.detach();
    }
    if (worker_joined) {
      worker.join();
    } else {
      worker.detach();
    }
    if (!(prof_joined && worker_joined)) {
      std::move(state_cleanup).Cancel();
    }

    if (deadlocked) {
      return absl::DeadlineExceededError(absl::StrFormat(
          "Profiler session deadlocked on attempt %d/%d "
          "(worker_step=%s, completed_iters=%d)",
          attempt, max_profiler_sessions,
          CrossContextWorkerStepName(deadlocked_step), deadlocked_iters));
    }

    CUresult iter_err = state->worker_cu_err.load(std::memory_order_acquire);
    if (iter_err != CUDA_SUCCESS) {
      return stream_executor::cuda::ToStatus(iter_err, "worker loop");
    }
    if (!state->keep_going.load(std::memory_order_acquire)) {
      break;
    }
  }

  return absl::OkStatus();
}

}  // namespace test
}  // namespace profiler
}  // namespace xla
