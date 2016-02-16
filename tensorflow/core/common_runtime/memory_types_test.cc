/* Copyright 2016 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/memory_types.h"

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

TEST(MemoryTypeChecker, Int32OK) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor v(DT_INT32, {});
  v.scalar<int32>().setZero();
  auto in0 = test::graph::Constant(g, v);
  auto in1 = test::graph::Constant(g, v);
  test::graph::Add(g, in0, in1);
  TF_EXPECT_OK(ValidateMemoryTypes(DEVICE_CPU, g));
#if GOOGLE_CUDA
  // There is a kernel for adding two int32s on host memory.
  TF_EXPECT_OK(ValidateMemoryTypes(DEVICE_GPU, g));
#endif  // GOOGLE_CUDA
  delete g;
}

TEST(MemoryTypeChecker, Int32NotOk) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor v(DT_INT32, {});
  v.scalar<int32>().setZero();
  auto x = test::graph::Constant(g, v);
  test::graph::Cast(g, x, DT_FLOAT);
  TF_EXPECT_OK(ValidateMemoryTypes(DEVICE_CPU, g));
#if GOOGLE_CUDA
  // There is no kernel for casting int32/host memory to float/device
  // memory.
  EXPECT_TRUE(errors::IsInternal(ValidateMemoryTypes(DEVICE_GPU, g)));
#endif  // GOOGLE_CUDA
  delete g;
}

}  // namespace
