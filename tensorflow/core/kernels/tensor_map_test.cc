/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/kernels/tensor_map.h"
#include "tensorflow/core/framework/tensor.h"
#include "absl/container/flat_hash_map.h"

#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

namespace {

TEST(TensorMapTest, Empty) {
  TensorMap tm;
  EXPECT_EQ(tm.tensors().size(), 0);
  EXPECT_EQ(tm.tensors().begin(), tm.tensors().end());
}

TEST(TensorMap, Copy) {
  TensorMap tm;
  TensorMap tmc = tm.Copy();
  EXPECT_EQ(tm.dtype(),tmc.dtype());
  EXPECT_EQ(tm.tensors(),tmc.tensors());
}

TEST(TensorMap, Insert) {
  EXPECT_EQ(1,1);
  TensorMap tm;
  Tensor k = Tensor(DT_INT64, TensorShape({1,1}));
  Tensor v = Tensor(DT_INT64, TensorShape({2,3}));
  tm.insert(k,v);
  absl::flat_hash_map<Tensor,Tensor> am;
  am.try_emplace(k,v);
  EXPECT_EQ(tm.tensors(), am);
}

//TODO(kattian): test Lookup, Erase

}  // namespace

}  // namespace tensorflow
