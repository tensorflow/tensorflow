/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#if TENSORFLOW_USE_ROCM || \
    (GOOGLE_CUDA && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)))

//#if TENSORFLOW_USE_ROCM

#include "tensorflow/core/kernels/cwise_op_fma.h"
#include "tensorflow/core/kernels/cwise_ops_gpu_common.cu.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

#if TENSORFLOW_USE_ROCM
#include "rocm/include/hip/hip_fp16.h"
#else
#include "third_party/gpus/cuda/include/cuda_fp16.hpp"
#endif

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

template <FMAType Type, typename T>
__device__ T fma_op(T m1, T m2) {
  if (Type == FMAType_Add)
    return m1 + m2;
  else if (Type == FMAType_Sub)
    return m1 - m2;
  else
    return m2 - m1;
}

template <class T>
__device__ T load_broadcast(const T* x) {
  return *x;
}

template <>
__device__ __half2 load_broadcast(const __half2* x) {
  return __half2half2(*(const __half*)x);
}

//
// Fast pathway: each tensor is either full-size or 1-element.
//

template <typename T, int N, FMAType Type>
__global__ void CwiseFusedMulAddKernel(GpuLaunchConfig cfg, T* out, const T* x1,
                                       const T* y1, const T* x2) {
  constexpr bool broadcast_x1 = (N & 1);
  constexpr bool broadcast_y1 = (N & 2);
  constexpr bool broadcast_x2 = (N & 4);
  T cx1 = load_broadcast(x1);
  T cx2 = load_broadcast(x2);
  T cy1 = load_broadcast(y1);
  GPU_1D_KERNEL_LOOP(i, cfg.virtual_thread_count) {
    out[i] = fma_op<Type>(
        (broadcast_x1 ? cx1 : x1[i]) * (broadcast_y1 ? cy1 : y1[i]),
        broadcast_x2 ? cx2 : x2[i]);
  }
}

template <typename T, int N, FMAType Type>
__global__ void CwiseFusedMulAdd2Kernel(GpuLaunchConfig cfg, T* out,
                                        const T* x1, const T* y1, const T* x2,
                                        const T* y2) {
  constexpr bool broadcast_x1 = (N & 1);
  constexpr bool broadcast_y1 = (N & 2);
  constexpr bool broadcast_x2 = (N & 4);
  constexpr bool broadcast_y2 = (N & 8);
  GPU_1D_KERNEL_LOOP(i, cfg.virtual_thread_count) {
    out[i] = fma_op<Type>(x1[broadcast_x1 ? 0 : i] * y1[broadcast_y1 ? 0 : i],
                          x2[broadcast_x2 ? 0 : i] * y2[broadcast_y2 ? 0 : i]);
  }
}

template <typename T, FMAType Type>
template <int N>
void LaunchFusedMulAddOp<GPUDevice, T, Type>::execute(const GPUDevice& device,
                                                      T* out, const T* x1,
                                                      const T* y1, const T* x2,
                                                      uint64 elements) {
  if (std::is_same<T, Eigen::half>::value && !(elements & 1)) {
    auto config = GetGpuLaunchConfig(elements / 2, device);
    TF_CHECK_OK(GpuLaunchKernel(
        CwiseFusedMulAddKernel<__half2, N, Type>, config.block_count,
        config.thread_per_block, 0, device.stream(), config, (__half2*)out,
        (const __half2*)x1, (const __half2*)y1, (const __half2*)x2));
  } else {
    auto config = GetGpuLaunchConfig(elements, device);
    TF_CHECK_OK(GpuLaunchKernel(CwiseFusedMulAddKernel<T, N, Type>,
                                config.block_count, config.thread_per_block, 0,
                                device.stream(), config, out, x1, y1, x2));
  }
}

template <typename T, FMAType Type>
void LaunchFusedMulAddOp<GPUDevice, T, Type>::operator()(
    const GPUDevice& device, T* out, const T* x1, const T* y1, const T* x2,
    uint64 elements, bool broadcast_x1, bool broadcast_y1, bool broadcast_x2) {
  int index =
      (broadcast_x1 ? 1 : 0) + (broadcast_y1 ? 2 : 0) + (broadcast_x2 ? 4 : 0);
  exec_fun* execs[8] = {&execute<0>, &execute<1>, &execute<2>, &execute<3>,
                        &execute<4>, &execute<5>, &execute<6>, &execute<7>};
  execs[index](device, out, x1, y1, x2, elements);
}

template <typename T, FMAType Type>
template <int N>
void LaunchFusedMulAdd2Op<GPUDevice, T, Type>::execute(const GPUDevice& device,
                                                       T* out, const T* x1,
                                                       const T* y1, const T* x2,
                                                       const T* y2,
                                                       uint64 elements) {
  auto config = GetGpuLaunchConfig(elements, device);
  TF_CHECK_OK(GpuLaunchKernel(CwiseFusedMulAdd2Kernel<T, N, Type>,
                              config.block_count, config.thread_per_block, 0,
                              device.stream(), config, out, x1, y1, x2, y2));
}

template <typename T, FMAType Type>
void LaunchFusedMulAdd2Op<GPUDevice, T, Type>::operator()(
    const GPUDevice& device, T* out, const T* x1, const T* y1, const T* x2,
    const T* y2, uint64 elements, bool broadcast_x1, bool broadcast_y1,
    bool broadcast_x2, bool broadcast_y2) {
  int index = (broadcast_x1 ? 1 : 0) + (broadcast_y1 ? 2 : 0) +
              (broadcast_x2 ? 4 : 0) + (broadcast_y2 ? 8 : 0);
  exec_fun* execs[16] = {
      &execute<0>,  &execute<1>,  &execute<2>,  &execute<3>,
      &execute<4>,  &execute<5>,  &execute<6>,  &execute<7>,
      &execute<8>,  &execute<9>,  &execute<10>, &execute<11>,
      &execute<12>, &execute<13>, &execute<14>, &execute<15>,
  };
  execs[index](device, out, x1, y1, x2, y2, elements);
}

//
// Fallback pathway: we need to broadcast some tensors along some dimensions
// but not others. Necessary because, at the time grappler does the fusion,
// we don't have full input shapes.
//
template <typename T, FMAType type, int N>
struct Fallback_FMA_Arg {
  T* out;
  const T* ptrs[N - 1];
  // uint64 has high overhead
  int32 shifts[N][5];
  int32 dims[6];
  // uint32 broadcast_mask;
  uint32 x_mask[N - 1];
  __device__ void shift(int dim, int delta) {
    out += shifts[0][dim - 1] * delta;
    for (int i = 0; i < N - 1; i++) ptrs[i] += shifts[i + 1][dim - 1] * delta;
  }
  void construct(const int64 dims[6], const uint8 broadcast_masks[6]);

  // todo: __half2 optimizations
  __device__ void x_loop(int start, int x_step) {
    for (int x = start; x < dims[0]; x += x_step) {
      T m1 = ptrs[0][x & x_mask[0]] * ptrs[1][x & x_mask[1]];
      T m2 = ptrs[2][x & x_mask[2]];
      if (N == 5) m2 *= ptrs[3][x & x_mask[3]];
      out[x] = fma_op<type>(m1, m2);
    }
  }

  __device__ void x_loop_2(int start, int span) {
    int x_cnt = 0;
    for (int x = start; x_cnt < span && x < dims[0]; x += blockDim.x) {
      T m1 = ptrs[0][x & x_mask[0]] * ptrs[1][x & x_mask[1]];
      T m2 = ptrs[2][x & x_mask[2]];
      if (N == 5) m2 *= ptrs[3][x & x_mask[3]];
      out[x] = fma_op<type>(m1, m2);
      x_cnt++;
    }
  }
};

template <typename T, FMAType type, int N>
void Fallback_FMA_Arg<T, type, N>::construct(const int64 _dims[6],
                                             const uint8 broadcast_masks[6]) {
  for (int y = 0; y < 6; y++) dims[y] = (int32)_dims[y];
  for (int y = 0; y < N; y++) {
    int b0, b1, b2;
    if (y == 0)
      b0 = b1 = b2 = 1;
    else {
      b0 = broadcast_masks[0] & (1 << (y - 1));
      b1 = broadcast_masks[1] & (1 << (y - 1));
      b2 = broadcast_masks[2] & (1 << (y - 1));
    }
    int stride = (b0 ? _dims[0] : 1) * (b1 ? _dims[1] : 1);
    shifts[y][1] = b2 ? stride : 0;
    for (int x = 1; x < 4; x++) {
      int b = (y == 0) ? 1 : (broadcast_masks[x + 1] & (1 << (y - 1)));
      int bn = (y == 0) ? 1 : (broadcast_masks[x + 2] & (1 << (y - 1)));
      if (b) stride *= _dims[x + 1];
      shifts[y][x + 1] = bn ? stride : 0;
    }
  }

  for (int y = 0; y < N - 1; y++)
    x_mask[y] = (broadcast_masks[0] & (1 << y)) ? -1 : 0;
  // broadcast_mask = broadcast_masks[0];

  shifts[0][0] = dims[0];
  for (int i = 1; i < N; i++)
    shifts[i][0] = (broadcast_masks[1] & (1 << (i - 1)))
                       ? ((broadcast_masks[0] & (1 << (i - 1))) ? dims[0] : 1)
                       : 0;
}

// "Fallback fallback" case
template <typename T, FMAType Type, int N>
__global__ void FallbackLaunchFusedMulAddKernel6D(
    Fallback_FMA_Arg<T, Type, N> arg) {
  arg.shift(4, blockIdx.z);
  arg.shift(3, blockIdx.y);
  int32 z = threadIdx.z + blockIdx.x * blockDim.z;
  if (z >= arg.dims[2]) return;
  arg.shift(2, z);
  arg.shift(1, threadIdx.y);
  for (int32_t t = 0; t < arg.dims[5]; t++) {
    int y_count = 0;
    for (int32 y = threadIdx.y; y < arg.dims[1]; y += blockDim.y) {
      arg.x_loop(threadIdx.x, blockDim.x);
      arg.shift(1, blockDim.y);
      y_count++;
    }
    arg.shift(1, -blockDim.y * y_count);
    arg.shift(5, 1);
  }
}

struct FallbackFMAGrid {
  // TODO: enforce repeats_x,y < 65536
  uint16 tdims[4];
  uint32 gdims[2];
};

template <typename T, FMAType Type, int N, int loops>
__global__ void FallbackLaunchFusedMulAddKernel4D(
    Fallback_FMA_Arg<T, Type, N> arg, FallbackFMAGrid grid) {
  int tid0 = threadIdx.x;
  int tid1 = threadIdx.y;
  int tid2 = threadIdx.z % grid.tdims[0];
  int tid3 = threadIdx.z / grid.tdims[0];
  int gid0 = blockIdx.x;
  int gid1 = blockIdx.y;
  int gid2 = blockIdx.z % grid.gdims[0];
  int gid3 = blockIdx.z / grid.gdims[0];

  int32 px = blockDim.x;
  int32 py = blockDim.y;
  int32 pz = grid.tdims[0];
  int32 pt = grid.tdims[1];

  int32 repeats_x = grid.tdims[2];
  int32 repeats_y = grid.tdims[3];
  int32 x0 = tid0 + gid0 * px * repeats_x;
  int32 y0 = tid1 + gid1 * py * repeats_y;
  int32 z0 = tid2 + gid2 * pz;
  int32 t0 = tid3 + gid3 * pt;
  pz *= grid.gdims[0];
  pt *= grid.gdims[1];
  if (x0 >= arg.dims[0]) return;
  if (y0 >= arg.dims[1]) return;
  if (z0 >= arg.dims[2]) return;
  if (t0 >= arg.dims[3]) return;
  arg.shift(3, t0);
  arg.shift(2, z0);
  arg.shift(1, y0);
  if (loops == 1) {
    arg.x_loop_2(x0, repeats_x);
  } else if (loops == 2) {
    int y_cnt = 0;
    for (int32 y = y0; y < arg.dims[1] && y_cnt < repeats_y; y += py) {
      arg.x_loop_2(x0, repeats_x);
      arg.shift(1, py);
      y_cnt++;
    }
  } else {
    for (int32 t = t0; t < arg.dims[3]; t += pt) {
      int z_cnt = 0;
      for (int32 z = z0; z < arg.dims[2]; z += pz) {
        int y_cnt = 0;
        for (int32 y = y0; y < arg.dims[1] && y_cnt < repeats_y; y += py) {
          arg.x_loop_2(x0, repeats_x);
          arg.shift(1, py);
          y_cnt++;
        }
        arg.shift(1, -py * y_cnt);
        arg.shift(2, pz);
        z_cnt++;
      }
      arg.shift(2, -pz * z_cnt);
      arg.shift(3, pt);
    }
  }
}

template <typename T, FMAType Type, int N>
__global__ void FallbackLaunchFusedMulAddKernel2D(
    Fallback_FMA_Arg<T, Type, N> arg) {
  arg.shift(1, blockIdx.x);
  arg.x_loop(threadIdx.x, blockDim.x);
}

template <typename T, FMAType type, int N>
void Fallback_FMA_execute(const GPUDevice& device, int64 dims[6],
                          Fallback_FMA_Arg<T, type, N>& arg) {
  const uint64 kThreads = 256;
  if (dims[2] == 1 && dims[3] == 1 && dims[4] == 1 && dims[5] == 1) {
    TF_CHECK_OK(GpuLaunchKernel(
        FallbackLaunchFusedMulAddKernel2D<T, type, N>, dim3(dims[1], 1, 1),
        dim3(dims[0] > 256 ? 256 : dims[0], 1, 1), 0, device.stream(), arg));
  } else if (dims[4] == 1 && dims[5] == 1) {
    FallbackFMAGrid grid_conf;
    dim3 block, grid;
    int64 min_blocks = 0, min_repeats = 0;

    (void)ReadInt64FromEnvVar("TF_FMA_MIN_BLOCKS", 0, &min_blocks);
    (void)ReadInt64FromEnvVar("TF_FMA_MIN_REPEATS", 16, &min_repeats);

    int repeats = 1;
    uint32 true_grid[2][4];
    int axis_repeats[4] = {1, 1, 1, 1};
    uint32 threads = 256, blocks = 1;
    bool y_loop = false, z_loop = false, t_loop = false;
    // We have 4 dimensions to loop over, but only 3 thread dimensions
    // Therefore, we squeeze dims[2] and dims[3] into threadIdx.z
    for (int i = 0; i < 4; i++) {
      true_grid[0][i] = min(threads, dims[i]);
      true_grid[1][i] = (dims[i] + true_grid[0][i] - 1) / true_grid[0][i];
      threads /= true_grid[0][i];
      blocks *= true_grid[1][i];
    }

    // It is more efficient for each thread to process >1 location
    if (min_blocks != 0 || min_repeats != 0) {
      for (int i = 0; i < 4; i++) {
        int factor =
            (min_blocks != 0) ? (blocks / min_blocks) : (min_repeats / repeats);
        if (factor <= 1) break;
        auto& g = true_grid[1][i];
        uint32 og = g;
        g = (g + factor - 1) / factor;
        if (g == 0) g = 1;
        if (g != og) {
          if (i == 1) y_loop = true;
          if (i == 2) z_loop = true;
          if (i == 3) t_loop = true;
        }
        axis_repeats[i] = (og + g - 1) / g;
        repeats *= axis_repeats[i];
        blocks /= og;
        blocks *= g;
      }
    }

    block.x = true_grid[0][0];
    block.y = true_grid[0][1];
    block.z = true_grid[0][2] * true_grid[0][3];
    grid.x = true_grid[1][0];
    grid.y = true_grid[1][1];
    grid.z = true_grid[1][2] * true_grid[1][3];
    grid_conf.tdims[0] = true_grid[0][2];
    grid_conf.tdims[1] = true_grid[0][3];
    grid_conf.gdims[0] = true_grid[1][2];
    grid_conf.gdims[1] = true_grid[1][3];
    grid_conf.tdims[2] = axis_repeats[0];
    grid_conf.tdims[3] = axis_repeats[1];
    TF_CHECK_OK(GpuLaunchKernel(
        (z_loop || t_loop)
            ? FallbackLaunchFusedMulAddKernel4D<T, type, N, 4>
            : (y_loop ? FallbackLaunchFusedMulAddKernel4D<T, type, N, 2>
                      : FallbackLaunchFusedMulAddKernel4D<T, type, N, 1>),
        grid, block, 0, device.stream(), arg, grid_conf));
  } else {
    int block_x = min(kThreads, dims[0]);
    int block_y = min(kThreads / block_x, dims[1]);
    int block_z = min(kThreads / (block_x * block_y), dims[2]);
    int grid_x = (dims[2] + block_z - 1) / block_z;
    int grid_y = dims[3];
    int grid_z = dims[4];
    TF_CHECK_OK(GpuLaunchKernel(FallbackLaunchFusedMulAddKernel6D<T, type, N>,
                                dim3(grid_x, grid_y, grid_z),
                                dim3(block_x, block_y, block_z), 0,
                                device.stream(), arg));
  }
}

template <typename T, FMAType Type>
void FallbackLaunchFusedMulAddOp<GPUDevice, T, Type>::operator()(
    const GPUDevice& device, T* out, const T* x1, const T* y1, const T* x2,
    int64 dims[6], uint8 broadcast_masks[6]) {
  Fallback_FMA_Arg<T, Type, 4> arg;
  arg.out = out;
  arg.ptrs[0] = x1;
  arg.ptrs[1] = y1;
  arg.ptrs[2] = x2;
  arg.construct(dims, broadcast_masks);
  Fallback_FMA_execute(device, dims, arg);
}

template <typename T, FMAType Type>
void FallbackLaunchFusedMulAdd2Op<GPUDevice, T, Type>::operator()(
    const GPUDevice& device, T* out, const T* x1, const T* y1, const T* x2,
    const T* y2, int64 dims[6], uint8 broadcast_masks[6]) {
  Fallback_FMA_Arg<T, Type, 5> arg;
  arg.out = out;
  arg.ptrs[0] = x1;
  arg.ptrs[1] = y1;
  arg.ptrs[2] = x2;
  arg.ptrs[3] = y2;
  arg.construct(dims, broadcast_masks);
  Fallback_FMA_execute(device, dims, arg);
}

#define INSTANTIATE_FMA(X, S)                                                 \
  template void LaunchFusedMulAddOp<GPUDevice, X, S>::operator()(             \
      const GPUDevice& device, X* out, const X* x1, const X* y1, const X* x2, \
      uint64 elements, bool bc1, bool bc2, bool bc3);                         \
  template void LaunchFusedMulAdd2Op<GPUDevice, X, S>::operator()(            \
      const GPUDevice& device, X* out, const X* x1, const X* y1, const X* x2, \
      const X* y2, uint64 elements, bool bc1, bool bc2, bool bc3, bool bc4);  \
  template void FallbackLaunchFusedMulAddOp<GPUDevice, X, S>::operator()(     \
      const GPUDevice& device, X* out, const X* x1, const X* y1, const X* x2, \
      int64 dims[5], uint8 broadcast_masks[5]);                               \
  template void FallbackLaunchFusedMulAdd2Op<GPUDevice, X, S>::operator()(    \
      const GPUDevice& device, X* out, const X* x1, const X* y1, const X* x2, \
      const X* y2, int64 dims[5], uint8 broadcast_masks[5]);

INSTANTIATE_FMA(Eigen::half, FMAType_Add);
INSTANTIATE_FMA(float, FMAType_Add);
INSTANTIATE_FMA(double, FMAType_Add);
INSTANTIATE_FMA(Eigen::half, FMAType_SubRev);
INSTANTIATE_FMA(float, FMAType_SubRev);
INSTANTIATE_FMA(double, FMAType_SubRev);
INSTANTIATE_FMA(Eigen::half, FMAType_Sub);
INSTANTIATE_FMA(float, FMAType_Sub);
INSTANTIATE_FMA(double, FMAType_Sub);

};  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
