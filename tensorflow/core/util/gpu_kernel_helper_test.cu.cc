/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU

#include <time.h>

#include <numeric>

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"

#define CUDA_EXPECT_SUCCESS                                 \
  {                                                         \
    gpuDeviceSynchronize();                                 \
    cudaError_t err = cudaGetLastError();                   \
    EXPECT_EQ(cudaSuccess, err) << cudaGetErrorString(err); \
  }

#define CUDA_ASSERT_SUCCESS                                 \
  {                                                         \
    gpuDeviceSynchronize();                                 \
    cudaError_t err = cudaGetLastError();                   \
    ASSERT_EQ(cudaSuccess, err) << cudaGetErrorString(err); \
  }

namespace tensorflow {

namespace {

__global__ void SetOutbufZero(GpuLaunchConfig config,
                              int* __restrict__ outbuf) {
  GPU_1D_KERNEL_LOOP(x, config.virtual_thread_count) { outbuf[x] = 0; }
}

// counting number of jobs by using atomic +1
__global__ void Count1D(GpuLaunchConfig config, int bufsize,
                        int* __restrict__ outbuf) {
  GPU_1D_KERNEL_LOOP(x, config.virtual_thread_count) {
    if (x < 0) {  // x might overflow when testing extreme case
      break;
    }
    atomicAdd(&outbuf[x % bufsize], 1);
  }
}
__global__ void Count2D(Gpu2DLaunchConfig config, int bufsize,
                        int* __restrict__ outbuf) {
  GPU_AXIS_KERNEL_LOOP(x, config.virtual_thread_count.x, X) {
    if (x < 0) {  // x might overflow when testing extreme case
      break;
    }
    GPU_AXIS_KERNEL_LOOP(y, config.virtual_thread_count.y, Y) {
      if (y < 0) {  // y might overflow when testing extreme case
        break;
      }
      int idx = x * config.virtual_thread_count.y + y;
      atomicAdd(&outbuf[idx % bufsize], 1);
    }
  }
}
__global__ void Count3D(Gpu3DLaunchConfig config, int bufsize,
                        int* __restrict__ outbuf) {
  GPU_AXIS_KERNEL_LOOP(x, config.virtual_thread_count.x, X) {
    if (x < 0) {  // x might overflow when testing extreme case
      break;
    }
    GPU_AXIS_KERNEL_LOOP(y, config.virtual_thread_count.y, Y) {
      if (y < 0) {  // y might overflow when testing extreme case
        break;
      }
      GPU_AXIS_KERNEL_LOOP(z, config.virtual_thread_count.z, Z) {
        if (z < 0) {  // z might overflow when testing extreme case
          break;
        }
        int idx =
            x * config.virtual_thread_count.y * config.virtual_thread_count.z +
            y * config.virtual_thread_count.z + z;
        atomicAdd(&outbuf[idx % bufsize], 1);
      }
    }
  }
}

__global__ void GpuShuffleGetSrcLaneTest(unsigned* __restrict__ failure_count) {
  unsigned lane_id = GpuLaneId();
  for (int width = warpSize; width > 1; width /= 2) {
    auto check_result = [&](const char* op_name, int param, unsigned actual,
                            unsigned expected) {
      if (actual != expected) {
        printf("Cuda%sGetSrcLane(%d, %d) for lane %d returned %d, not %d\n",
               op_name, param, width, lane_id, actual, expected);
        GpuAtomicAdd(failure_count, 1);
      }
    };

    for (int src_lane = -warpSize; src_lane <= warpSize; ++src_lane) {
#if TENSORFLOW_USE_ROCM
      if (src_lane < 0 || src_lane >= width) continue;
#endif
      unsigned actual_lane = detail::GpuShuffleGetSrcLane(src_lane, width);
      unsigned expect_lane =
          GpuShuffleSync(kCudaWarpAll, lane_id, src_lane, width);
      check_result("Shuffle", src_lane, actual_lane, expect_lane);
    }

    for (unsigned delta = 0; delta <= warpSize; ++delta) {
      unsigned actual_lane = detail::GpuShuffleUpGetSrcLane(delta, width);
      unsigned expect_lane =
          GpuShuffleUpSync(kCudaWarpAll, lane_id, delta, width);
      check_result("ShuffleUp", delta, actual_lane, expect_lane);
    }

    for (unsigned delta = 0; delta <= warpSize; ++delta) {
      unsigned actual_lane = detail::GpuShuffleDownGetSrcLane(delta, width);
      unsigned expect_lane =
          GpuShuffleDownSync(kCudaWarpAll, lane_id, delta, width);
      check_result("ShuffleDown", delta, actual_lane, expect_lane);
    }

    for (int lane_lane = warpSize; lane_lane > 0; lane_lane /= 2) {
      unsigned actual_lane = detail::GpuShuffleXorGetSrcLane(lane_lane, width);
      unsigned expect_lane =
          GpuShuffleXorSync(kCudaWarpAll, lane_id, lane_lane, width);
      check_result("ShuffleXor", lane_lane, actual_lane, expect_lane);
    }
  }
}

}  // namespace

class GpuLaunchConfigTest : public ::testing::Test {
 protected:
  static const int bufsize = 1024;
  int* outbuf = nullptr;
  int* outbuf_host = nullptr;
  int hostbuf[bufsize];
  Eigen::GpuStreamDevice stream;
  Eigen::GpuDevice d = Eigen::GpuDevice(&stream);

  void copyToHost() {
#if TENSORFLOW_USE_ROCM
    hipMemcpy(hostbuf, outbuf, sizeof(int) * bufsize, hipMemcpyDeviceToHost);
#endif
  }
  virtual void SetUp() {
#if GOOGLE_CUDA
    cudaError_t err = cudaMallocManaged(&outbuf, sizeof(int) * bufsize);
    outbuf_host = outbuf;
#else
    cudaError_t err = hipMalloc(&outbuf, sizeof(int) * bufsize);
    outbuf_host = hostbuf;
#endif
    ASSERT_EQ(cudaSuccess, err) << cudaGetErrorString(err);
  }

  virtual void TearDown() {
    gpuDeviceSynchronize();
    gpuFree(outbuf);
    outbuf = nullptr;
  }
};

TEST_F(GpuLaunchConfigTest, GetGpuLaunchConfig) {
  GpuLaunchConfig cfg;

// test valid inputs
#define TEST_LAUNCH_PARAMETER(work_element_count)                             \
  cfg = GetGpuLaunchConfig(bufsize, d);                                       \
  TF_CHECK_OK(GpuLaunchKernel(SetOutbufZero, cfg.block_count,                 \
                              cfg.thread_per_block, 0, d.stream(), cfg,       \
                              outbuf));                                       \
  CUDA_ASSERT_SUCCESS                                                         \
  cfg = GetGpuLaunchConfig(work_element_count, d);                            \
  TF_CHECK_OK(GpuLaunchKernel(Count1D, cfg.block_count, cfg.thread_per_block, \
                              0, d.stream(), cfg, bufsize, outbuf));          \
  CUDA_EXPECT_SUCCESS                                                         \
  copyToHost();                                                               \
  EXPECT_EQ(work_element_count,                                               \
            std::accumulate(outbuf_host, outbuf_host + bufsize, 0));          \
                                                                              \
  cfg = GetGpuLaunchConfig(bufsize, d, SetOutbufZero, 0, 0);                  \
  TF_CHECK_OK(GpuLaunchKernel(SetOutbufZero, cfg.block_count,                 \
                              cfg.thread_per_block, 0, d.stream(), cfg,       \
                              outbuf));                                       \
  CUDA_ASSERT_SUCCESS                                                         \
  cfg = GetGpuLaunchConfig(work_element_count, d, Count1D, 0, 0);             \
  TF_CHECK_OK(GpuLaunchKernel(Count1D, cfg.block_count, cfg.thread_per_block, \
                              0, d.stream(), cfg, bufsize, outbuf));          \
  CUDA_EXPECT_SUCCESS                                                         \
  copyToHost();                                                               \
  EXPECT_EQ(work_element_count,                                               \
            std::accumulate(outbuf_host, outbuf_host + bufsize, 0));

  TEST_LAUNCH_PARAMETER(128);
  TEST_LAUNCH_PARAMETER(129);
  TEST_LAUNCH_PARAMETER(511);
  TEST_LAUNCH_PARAMETER(512);
  TEST_LAUNCH_PARAMETER(2048);
  TEST_LAUNCH_PARAMETER(2049);
  TEST_LAUNCH_PARAMETER(8191);
  TEST_LAUNCH_PARAMETER(8192);
  TEST_LAUNCH_PARAMETER(123456);
  TEST_LAUNCH_PARAMETER(1 << 30);
#undef TEST_LAUNCH_PARAMETER
}

bool operator==(const Gpu2DLaunchConfig& a, const Gpu2DLaunchConfig& b) {
  return a.thread_per_block.x == b.thread_per_block.x &&
         a.thread_per_block.y == b.thread_per_block.y &&
         a.thread_per_block.z == b.thread_per_block.z &&
         a.block_count.x == b.block_count.x &&
         a.block_count.y == b.block_count.y &&
         a.block_count.z == b.block_count.z &&
         a.thread_per_block.x == b.thread_per_block.x &&
         a.thread_per_block.y == b.thread_per_block.y &&
         a.thread_per_block.z == b.thread_per_block.z;
}

TEST_F(GpuLaunchConfigTest, GetGpu2DLaunchConfig) {
  Gpu2DLaunchConfig cfg;
  GpuLaunchConfig cfg1d;

// test valid inputs
#define TEST_LAUNCH_PARAMETER(dimx, dimy)                                      \
  cfg1d = GetGpuLaunchConfig(bufsize, d);                                      \
  TF_EXPECT_OK(GpuLaunchKernel(SetOutbufZero, cfg1d.block_count,               \
                               cfg1d.thread_per_block, 0, d.stream(), cfg1d,   \
                               outbuf));                                       \
  CUDA_ASSERT_SUCCESS                                                          \
  cfg = GetGpu2DLaunchConfig(dimx, dimy, d);                                   \
  TF_EXPECT_OK(GpuLaunchKernel(Count2D, cfg.block_count, cfg.thread_per_block, \
                               0, d.stream(), cfg, bufsize, outbuf));          \
  CUDA_EXPECT_SUCCESS                                                          \
  copyToHost();                                                                \
  EXPECT_EQ(dimx* dimy,                                                        \
            std::accumulate(outbuf_host, outbuf_host + bufsize, 0));           \
                                                                               \
  cfg1d = GetGpuLaunchConfig(bufsize, d, SetOutbufZero, 0, 0);                 \
  TF_EXPECT_OK(GpuLaunchKernel(SetOutbufZero, cfg1d.block_count,               \
                               cfg1d.thread_per_block, 0, d.stream(), cfg1d,   \
                               outbuf));                                       \
  CUDA_ASSERT_SUCCESS                                                          \
  cfg = GetGpu2DLaunchConfig(dimx, dimy, d, Count2D, 0, 0);                    \
  TF_EXPECT_OK(GpuLaunchKernel(Count2D, cfg.block_count, cfg.thread_per_block, \
                               0, d.stream(), cfg, bufsize, outbuf));          \
  CUDA_EXPECT_SUCCESS                                                          \
  copyToHost();                                                                \
  EXPECT_EQ(dimx* dimy, std::accumulate(outbuf_host, outbuf_host + bufsize, 0))

  TEST_LAUNCH_PARAMETER(128, 128);
  TEST_LAUNCH_PARAMETER(129, 64);
  TEST_LAUNCH_PARAMETER(511, 2048);
  TEST_LAUNCH_PARAMETER(512, 512);
  TEST_LAUNCH_PARAMETER(2048, 1024);
  TEST_LAUNCH_PARAMETER(2049, 32);
  TEST_LAUNCH_PARAMETER(8191, 1);
  TEST_LAUNCH_PARAMETER(8192, 10);
  TEST_LAUNCH_PARAMETER(123456, 12);
  TEST_LAUNCH_PARAMETER(1, 1 << 30);
  TEST_LAUNCH_PARAMETER(1 << 30, 1);
#undef TEST_LAUNCH_PARAMETER
}

TEST_F(GpuLaunchConfigTest, GetGpu3DLaunchConfig) {
  Gpu3DLaunchConfig cfg;
  GpuLaunchConfig cfg1d;

// test valid inputs
#define TEST_LAUNCH_PARAMETER(dimx, dimy, dimz)                                \
  cfg1d = GetGpuLaunchConfig(bufsize, d, SetOutbufZero, 0, 0);                 \
  TF_EXPECT_OK(GpuLaunchKernel(SetOutbufZero, cfg1d.block_count,               \
                               cfg1d.thread_per_block, 0, d.stream(), cfg1d,   \
                               outbuf));                                       \
  CUDA_ASSERT_SUCCESS                                                          \
  cfg = GetGpu3DLaunchConfig(dimx, dimy, dimz, d, Count3D, 0, 0);              \
  TF_EXPECT_OK(GpuLaunchKernel(Count3D, cfg.block_count, cfg.thread_per_block, \
                               0, d.stream(), cfg, bufsize, outbuf));          \
  CUDA_EXPECT_SUCCESS                                                          \
  copyToHost();                                                                \
  EXPECT_EQ(dimx* dimy* dimz,                                                  \
            std::accumulate(outbuf_host, outbuf_host + bufsize, 0))

  TEST_LAUNCH_PARAMETER(128, 128, 128);
  TEST_LAUNCH_PARAMETER(129, 64, 1024);
  TEST_LAUNCH_PARAMETER(511, 2048, 128);
  TEST_LAUNCH_PARAMETER(512, 512, 64);
  TEST_LAUNCH_PARAMETER(2048, 1024, 128);
  TEST_LAUNCH_PARAMETER(2049, 32, 1024);
  TEST_LAUNCH_PARAMETER(8191, 1, 1024);
  TEST_LAUNCH_PARAMETER(8192, 10, 32);
  TEST_LAUNCH_PARAMETER(123456, 12, 21);
  TEST_LAUNCH_PARAMETER(1, 1, 1 << 30);
  TEST_LAUNCH_PARAMETER(1, 1 << 30, 1);
  TEST_LAUNCH_PARAMETER(1 << 30, 1, 1);
#undef TEST_LAUNCH_PARAMETER
}

TEST(CudaDeviceFunctionsTest, ShuffleGetSrcLane) {
  unsigned* failure_count;
#if GOOGLE_CUDA
  ASSERT_EQ(cudaMallocManaged(&failure_count, sizeof(unsigned)), cudaSuccess);
#else
  ASSERT_EQ(hipHostMalloc(&failure_count, sizeof(unsigned), 0), cudaSuccess);
#endif
  *failure_count = 0;
  TF_EXPECT_OK(GpuLaunchKernel(GpuShuffleGetSrcLaneTest, 1, TF_RED_WARPSIZE, 0,
                               nullptr, failure_count));
  ASSERT_EQ(gpuDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(*failure_count, 0);
  gpuFree(failure_count);
}

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
