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
#include <functional>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include <gtest/gtest.h>
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "third_party/gpus/cuda/include/driver_types.h"
#include "xla/backends/profiler/gpu/cuda_test.h"
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

}  // namespace test
}  // namespace profiler
}  // namespace xla
