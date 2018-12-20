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

#ifndef INTEL_MKL_ML_ONLY

TEST(MklUtilTest, MklDnnTfShape) {
  auto cpu_engine = engine(engine::cpu, 0);
  MklDnnData<float> a(&cpu_engine);

  const int N = 1, C = 2, H = 3, W = 4;
  memory::dims a_dims = {N, C, H, W};
  MklDnnShape a_mkldnn_shape;
  a_mkldnn_shape.SetMklTensor(true);
  // Create TF layout in NCHW.
  a_mkldnn_shape.SetTfLayout(a_dims.size(), a_dims, memory::format::nchw);
  TensorShape a_tf_shape_nchw({N, C, H, W});
  TensorShape a_tf_shape_nhwc({N, H, W, C});
  TensorShape a_mkldnn_tf_shape = a_mkldnn_shape.GetTfShape();
  // Check that returned shape is in NCHW format.
  EXPECT_EQ(a_tf_shape_nchw, a_mkldnn_tf_shape);
  EXPECT_NE(a_tf_shape_nhwc, a_mkldnn_tf_shape);

  memory::dims b_dims = {N, C, H, W};
  MklDnnShape b_mkldnn_shape;
  b_mkldnn_shape.SetMklTensor(true);
  // Create TF layout in NHWC.
  b_mkldnn_shape.SetTfLayout(b_dims.size(), b_dims, memory::format::nhwc);
  TensorShape b_tf_shape_nhwc({N, H, W, C});
  TensorShape b_tf_shape_nchw({N, C, H, W});
  TensorShape b_mkldnn_tf_shape = b_mkldnn_shape.GetTfShape();
  // Check that returned shape is in NHWC format.
  EXPECT_EQ(b_tf_shape_nhwc, b_mkldnn_tf_shape);
  EXPECT_NE(b_tf_shape_nchw, b_mkldnn_tf_shape);
}

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

TEST(MklUtilTest, LRUCacheTest) {
  // The cached objects are of type int*
  size_t capacity = 100;
  size_t num_objects = capacity + 10;
  LRUCache<int> lru_cache(capacity);

  // Test SetOp: be able to set more ops than the capacity
  for (int k = 0; k < num_objects; k++) {
    lru_cache.SetOp(std::to_string(k), new int(k));
  }

  // Test GetOp and capacity:
  //   -- Less recently accessed objects should not be
  //      in cache any more.
  for (int k = 0; k < num_objects - capacity; k++) {
    EXPECT_EQ(nullptr, lru_cache.GetOp(std::to_string(k)));
  }

  // Test GetOp and capacity:
  //   -- most recently accessed objects should still
  //      be in cache.
  for (int k = num_objects - capacity; k < num_objects; k++) {
    int* pint = lru_cache.GetOp(std::to_string(k));
    EXPECT_NE(nullptr, pint);
    EXPECT_EQ(*pint, k);
  }

  // Clean up the cache
  lru_cache.Clear();

  // After clean up, there should be no cached object.
  for (int k = 0; k < num_objects; k++) {
    EXPECT_EQ(nullptr, lru_cache.GetOp(std::to_string(k)));
  }
}

#endif  // INTEL_MKL_ML_ONLY
}  // namespace
}  // namespace tensorflow

#endif  // INTEL_MKL
