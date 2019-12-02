/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/eager/tensor_handle.h"

#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(TensorHandle_ShapeTest, AsyncShape) {
  Tensor t(DT_UINT16, TensorShape({2, 2}));
  EXPECT_TRUE(t.shape().IsSameSize(TensorShape({2, 2})));
  for (int64 a = 0; a < t.shape().dim_size(0); a++) {
    for (int64 b = 0; b < t.shape().dim_size(1); b++) {
      t.matrix<uint16>()(a, b) = uint16(a * b);
    }
  }

  TensorHandle* sync_th;
  EXPECT_TRUE(TensorHandle::CreateLocalHandle(t, &sync_th).ok());
  TensorHandle* async_th;
  EXPECT_TRUE(TensorHandle::CreateEmptyLocalHandle(true, nullptr, nullptr,
                                                   nullptr, DataType::DT_UINT16,
                                                   nullptr, &async_th)
                  .ok());

  EXPECT_TRUE(async_th->CopyInferenceShape(sync_th).ok());
  EXPECT_FALSE(async_th->IsReady());

  TensorShape sync_shape;
  TensorShape async_shape;
  EXPECT_TRUE(sync_th->Shape(&sync_shape).ok());
  EXPECT_TRUE(async_th->Shape(&async_shape).ok());
  EXPECT_EQ(sync_shape, async_shape);

  int num_dims = -1;
  EXPECT_TRUE(async_th->NumDims(&num_dims).ok());
  EXPECT_EQ(num_dims, 2);

  int64 num_elements = -1;
  EXPECT_TRUE(async_th->NumElements(&num_elements).ok());
  EXPECT_EQ(num_elements, 4);

  sync_th->Unref();
  async_th->Unref();
}

}  // namespace
}  // namespace tensorflow
