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

#ifdef INTEL_MKL

#include "tensorflow/core/util/mkl_util.h"

#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

#ifdef INTEL_MKL_DNN

TEST(MklUtilTest, MklDnnBlockedFormatTest) {
  // Let's create 2D tensor of shape {3, 4} with 3 being innermost dimension
  // first (case 1) and then it being outermost dimension (case 2).
  auto cpu_engine = engine(engine::cpu, 0);

  // Setting for case 1
  MklDnnData<float> a(&cpu_engine);
  memory::dims dim1 = {3, 4};
  memory::dims strides1 = {1, 3};
  a.SetUsrMem(dim1, strides1);

  memory::desc a_md1 = a.GetUsrMemDesc();
  EXPECT_EQ(a_md1.data.ndims, 2);
  EXPECT_EQ(a_md1.data.dims[0], 3);
  EXPECT_EQ(a_md1.data.dims[1], 4);
  EXPECT_EQ(a_md1.data.format, mkldnn_blocked);

  // Setting for case 2
  MklDnnData<float> b(&cpu_engine);
  memory::dims dim2 = {3, 4};
  memory::dims strides2 = {4, 1};
  b.SetUsrMem(dim2, strides2);

  memory::desc b_md2 = b.GetUsrMemDesc();
  EXPECT_EQ(b_md2.data.ndims, 2);
  EXPECT_EQ(b_md2.data.dims[0], 3);
  EXPECT_EQ(b_md2.data.dims[1], 4);
  EXPECT_EQ(b_md2.data.format, mkldnn_blocked);
}

#endif  // INTEL_MKL_DNN
}  // namespace
}  // namespace tensorflow

#endif  // INTEL_MKL
