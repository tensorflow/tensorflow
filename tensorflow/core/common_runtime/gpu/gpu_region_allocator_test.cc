/* Copyright 2015 Google Inc. All Rights Reserved.

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

#if GOOGLE_CUDA

#include "tensorflow/core/common_runtime/gpu/gpu_region_allocator.h"

#include <algorithm>
#include <vector>

#include "tensorflow/stream_executor/stream_executor.h"
#include <gtest/gtest.h>
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/port.h"

namespace gpu = ::perftools::gputools;

namespace tensorflow {
namespace {

TEST(GPURegionAllocatorTest, Simple) {
  GPURegionAllocator a(0, 1 << 26);
  std::vector<void*> ptrs;
  for (int s = 1; s < 1024; s++) {
    void* raw = a.AllocateRaw(1, s);
    ptrs.push_back(raw);
  }
  std::sort(ptrs.begin(), ptrs.end());
  for (int i = 0; i < ptrs.size(); i++) {
    if (i > 0) {
      CHECK_NE(ptrs[i], ptrs[i - 1]);  // No dups
    }
    a.DeallocateRaw(ptrs[i]);
  }
  float* t1 = a.Allocate<float>(1024);
  double* t2 = a.Allocate<double>(1048576);
  a.Deallocate(t1);
  a.Deallocate(t2);
}

TEST(GPURegionAllocatorTest, CheckMemLeak) {
  EXPECT_DEATH(
      {
        GPURegionAllocator a(0, 1 << 26);
        float* t1 = a.Allocate<float>(1024);
        if (t1) {
          LOG(INFO) << "Not deallocating";
        }
      },
      "");
}

TEST(GPURegionAllocatorTest, TracksSizes) {
  GPURegionAllocator a(0, 1 << 26);
  EXPECT_EQ(true, a.TracksAllocationSizes());
}

TEST(GPURegionAllocatorTest, AllocatedVsRequested) {
  GPURegionAllocator a(0, 1 << 26);
  float* t1 = a.Allocate<float>(1);
  EXPECT_EQ(sizeof(float), a.RequestedSize(t1));

  // Minimum allocation size if 256
  EXPECT_EQ(256, a.AllocatedSize(t1));

  a.Deallocate(t1);
}

}  // namespace
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
