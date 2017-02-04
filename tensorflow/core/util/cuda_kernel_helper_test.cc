/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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


#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "tensorflow/core/platform/test.h"

#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {

TEST(SetConstantTest, RandomValue) {
  std::random_device r;
  std::default_random_engine engine(r());
  std::uniform_int_distribution<float> uniform_dist(0.0f, 1.0f);
  const float random_value = uniform_dist(engine);

  const int element_num = 4096;
  thrust::device_vector<float> vec_d(element_num);

  GPUDevice d;
  const int32 count = element_num;
  CudaLaunchConfig config = GetCudaLaunchConfig(count, d);
  SetConstant<float><<<config.block_count, config.thread_per_block, 0, d.stream()>>>
    (config.virtual_thread_count, thrust::raw_pointer_cast(vec_d.data()), random_value);

  thrust::host_vector<float> vec_h = vec_d;
  for (int i = 0; i < element_num; ++ i) {
    EXPECT_EQ(vec_h[i], random_value);
  }
}

TEST(ReplaceValueTest, RandomValue) {
}

TEST(CudaAtomicAddTest, DoubleType) {
}

TEST(CudaAtomicMaxTest, FloatType) {
}

TEST(CudaAtomicMaxTest, DoubleType) {
}

TEST(CudaAtomicMaxTest, HalfType) {
}

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
