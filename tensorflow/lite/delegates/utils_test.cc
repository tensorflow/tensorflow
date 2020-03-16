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
#include "tensorflow/lite/delegates/utils.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace delegates {
namespace {

TEST(UtilsTest, CreateNewTensorWithDifferentTypeTest) {
  std::vector<TfLiteTensor> tensors(2);
  // Data about original tensor.
  // The same shape should be reflected in tensors[1] later.
  tensors[0].dims = TfLiteIntArrayCreate(2);
  tensors[0].dims->data[0] = 2;
  tensors[0].dims->data[1] = 3;
  tensors[0].type = kTfLiteFloat32;
  // To simulate a valid TFLite Context.
  TfLiteContext context;
  context.AddTensors = [](struct TfLiteContext*, int tensors_to_add,
                          int* first_new_tensor_index) {
    // The util should be adding exactly one tensor to the graph.
    if (tensors_to_add != 1) {
      return kTfLiteError;
    }
    // This ensures that the 'new tensor' is the second tensor in the vector
    // above.
    *first_new_tensor_index = 1;
    return kTfLiteOk;
  };
  context.ResizeTensor = [](struct TfLiteContext*, TfLiteTensor* tensor,
                            TfLiteIntArray* new_size) {
    // Ensure dimensions are the same as the original tensor.
    if (new_size->size != 2 || new_size->data[0] != 2 || new_size->data[1] != 3)
      return kTfLiteError;
    tensor->dims = new_size;
    return kTfLiteOk;
  };
  context.tensors = tensors.data();

  TfLiteTensor* new_tensor = nullptr;
  int new_tensor_index = -1;
  EXPECT_EQ(CreateNewTensorWithDifferentType(
                &context, /**original_tensor_index**/ 0,
                /**new_type**/ kTfLiteUInt8, &new_tensor, &new_tensor_index),
            kTfLiteOk);
  EXPECT_EQ(new_tensor_index, 1);
  EXPECT_NE(new_tensor, nullptr);
  EXPECT_NE(new_tensor->dims, nullptr);
  EXPECT_EQ(new_tensor->type, kTfLiteUInt8);
  EXPECT_EQ(new_tensor->allocation_type, kTfLiteArenaRw);

  // Cleanup.
  TfLiteIntArrayFree(tensors[0].dims);
  TfLiteIntArrayFree(tensors[1].dims);
}

}  // namespace
}  // namespace delegates
}  // namespace tflite
