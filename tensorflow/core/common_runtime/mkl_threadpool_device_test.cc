/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifdef INTEL_MKL

#include "tensorflow/core/common_runtime/threadpool_device.h"

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

#ifdef _OPENMP
TEST(MKLThreadPoolDeviceTest, TestOmpDefaults) {
  SessionOptions options;
  unsetenv("OMP_NUM_THREADS");

  ThreadPoolDevice* tp = new ThreadPoolDevice(
      options, "/device:CPU:0", Bytes(256), DeviceLocality(), cpu_allocator());

  const int ht = port::NumHyperthreadsPerCore();
  EXPECT_EQ(omp_get_max_threads(), (port::NumSchedulableCPUs() + ht - 1) / ht);
}

TEST(MKLThreadPoolDeviceTest, TestOmpPreSets) {
  SessionOptions options;
  setenv("OMP_NUM_THREADS", "314", 1);

  ThreadPoolDevice* tp = new ThreadPoolDevice(
      options, "/device:CPU:0", Bytes(256), DeviceLocality(), cpu_allocator());

  EXPECT_EQ(omp_get_max_threads(), 314);
}
#endif  // _OPENMP

}  // namespace tensorflow

#endif  // INTEL_MKL
