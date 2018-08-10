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

#include "tensorflow/c/eager/c_api.h"

#include <string.h>
#include "tensorflow/c/eager/c_api_test_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

TEST(CApiDebug, ScalarCPU) {
  TFE_TensorHandle* h = TestScalarTensorHandle();
  TF_Status* status = TF_NewStatus();
  TFE_TensorDebugInfo* debug_info = TFE_TensorHandleTensorDebugInfo(h, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  ASSERT_EQ(0, TFE_TensorDebugInfoOnDeviceNumDims(debug_info));

  TFE_DeleteTensorDebugInfo(debug_info);
  TFE_DeleteTensorHandle(h);
  TF_DeleteStatus(status);
}

TEST(CApiDebug, 2DCPU) {
  TFE_TensorHandle* h = TestMatrixTensorHandle3X2();
  TF_Status* status = TF_NewStatus();
  TFE_TensorDebugInfo* debug_info = TFE_TensorHandleTensorDebugInfo(h, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  ASSERT_EQ(2, TFE_TensorDebugInfoOnDeviceNumDims(debug_info));
  // Shape is the same for CPU tensors.
  EXPECT_EQ(3, TFE_TensorDebugInfoOnDeviceDim(debug_info, 0));
  EXPECT_EQ(2, TFE_TensorDebugInfoOnDeviceDim(debug_info, 1));

  TFE_DeleteTensorDebugInfo(debug_info);
  TFE_DeleteTensorHandle(h);
  TF_DeleteStatus(status);
}
