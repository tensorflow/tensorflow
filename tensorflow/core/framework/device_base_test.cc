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

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/device_base.h"

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

TEST(DeviceBaseTest, CpuDevice) {
  DeviceBase dbase(Env::Default());
  thread::ThreadPool pool(Env::Default(), "test", 16);
  Eigen::ThreadPoolDevice eigen_device(pool.AsEigenThreadPool(),
                                       pool.NumThreads());
  ASSERT_FALSE(dbase.has_eigen_cpu_device());
  dbase.set_eigen_cpu_device(&eigen_device);
  ASSERT_TRUE(dbase.has_eigen_cpu_device());

  {
    auto d = dbase.eigen_cpu_device();
    EXPECT_EQ(d->numThreads(), 16);
  }

  {
    ScopedPerThreadMaxParallelism maxp(4);
    auto d = dbase.eigen_cpu_device();
    EXPECT_EQ(d->numThreads(), 4);
  }

  {
    ScopedPerThreadMaxParallelism maxp(1);
    auto d = dbase.eigen_cpu_device();
    EXPECT_EQ(d->numThreads(), 1);
  }

  {
    ScopedPerThreadMaxParallelism maxp(1000);
    auto d = dbase.eigen_cpu_device();
    EXPECT_EQ(d->numThreads(), 16);
  }
}

}  // namespace tensorflow
