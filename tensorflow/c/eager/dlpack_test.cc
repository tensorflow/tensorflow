/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/c/eager/dlpack.h"

#include <vector>

#include "absl/strings/str_join.h"
#include "include/dlpack/dlpack.h"  // from @dlpack
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

void TestHandleFromDLPack(TF_Status* status, TFE_Context* ctx,
                          std::vector<int64_t> shape,
                          std::vector<int64_t> strides) {
  size_t num_elements = 1;
  for (int i = 0; i < static_cast<int32_t>(shape.size()); ++i) {
    num_elements *= shape[i];
  }
  std::vector<float> data(num_elements);
  for (size_t j = 0; j < num_elements; ++j) {
    data[j] = j;
  }
  DLManagedTensor dlm_in = {};
  DLTensor* dltensor_in = &dlm_in.dl_tensor;
  dltensor_in->data = data.data();
  dltensor_in->device = {kDLCPU, 0};
  dltensor_in->ndim = static_cast<int32_t>(shape.size());
  dltensor_in->dtype = {kDLFloat, 32, 1};
  dltensor_in->shape = shape.data();
  dltensor_in->strides = strides.data();
  TFE_TensorHandle* handle = TFE_HandleFromDLPack(&dlm_in, status, ctx);
  ASSERT_NE(handle, nullptr)
      << TF_Message(status) << " (shape=[" << absl::StrJoin(shape, ",")
      << "], strides=[" << absl::StrJoin(strides, ",") << "])";

  auto* dlm_out =
      static_cast<DLManagedTensor*>(TFE_HandleToDLPack(handle, status));
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  const DLTensor* dltensor_out = &dlm_out->dl_tensor;
  EXPECT_EQ(dltensor_out->device.device_type, dltensor_in->device.device_type);
  EXPECT_EQ(dltensor_out->device.device_id, dltensor_in->device.device_id);
  EXPECT_EQ(dltensor_out->ndim, dltensor_in->ndim);
  EXPECT_EQ(dltensor_out->dtype.code, dltensor_in->dtype.code);
  EXPECT_EQ(dltensor_out->dtype.bits, dltensor_in->dtype.bits);
  EXPECT_EQ(dltensor_out->dtype.lanes, dltensor_in->dtype.lanes);
  for (int i = 0; i < dltensor_in->ndim; ++i) {
    EXPECT_EQ(dltensor_out->shape[i], dltensor_in->shape[i]);
    if (dltensor_out->strides) {
      if (i == dltensor_in->ndim - 1) {
        EXPECT_EQ(dltensor_out->strides[i], 1);
      } else {
        EXPECT_EQ(dltensor_out->strides[i],
                  dltensor_out->shape[i + 1] * dltensor_out->strides[i + 1]);
      }
    }
  }
  const float* data_in = static_cast<const float*>(dltensor_in->data);
  const float* data_out = static_cast<const float*>(dltensor_out->data);
  for (size_t j = 0; j < num_elements; ++j) {
    EXPECT_EQ(data_out[j], data_in[j]);
  }

  TFE_CallDLManagedTensorDeleter(dlm_out);
  TFE_DeleteTensorHandle(handle);
}

TEST(DLPack, HandleFromDLPackStrides) {
  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_Context* ctx = TFE_NewContext(opts, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  TestHandleFromDLPack(status, ctx, {}, {});
  TestHandleFromDLPack(status, ctx, {4}, {});
  TestHandleFromDLPack(status, ctx, {4}, {1});
  TestHandleFromDLPack(status, ctx, {4, 3, 2}, {});
  TestHandleFromDLPack(status, ctx, {4, 3, 2}, {6, 2, 1});
  // Test that dims with size=1 can have any stride.
  TestHandleFromDLPack(status, ctx, {1}, {1});
  TestHandleFromDLPack(status, ctx, {1}, {0});
  TestHandleFromDLPack(status, ctx, {4, 1, 2}, {2, 1, 1});
  TestHandleFromDLPack(status, ctx, {4, 1, 2}, {2, 0, 1});
  TestHandleFromDLPack(status, ctx, {4, 3, 1}, {3, 1, 1});
  TestHandleFromDLPack(status, ctx, {4, 3, 1}, {3, 1, 0});
  // Test that empty tensors can have any strides.
  TestHandleFromDLPack(status, ctx, {4, 0, 2}, {0, 2, 1});
  TestHandleFromDLPack(status, ctx, {4, 0, 2}, {0, 1, 1});
  TestHandleFromDLPack(status, ctx, {4, 0, 2}, {0, 0, 1});
  TestHandleFromDLPack(status, ctx, {4, 0, 2}, {0, 2, 0});

  TFE_DeleteContext(ctx);
  TF_DeleteStatus(status);
}

}  // namespace
}  // namespace tensorflow
