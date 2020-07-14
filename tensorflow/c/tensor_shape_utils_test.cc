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

#include "tensorflow/c/tensor_shape_utils.h"
#include "tensorflow/c/tf_tensor_internal.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"

namespace tensorflow { 

template <typename T> 
void TestShapeMatch(T shape) {
	Tensor tensor(DT_FLOAT, shape); 
	Status status; 
	TF_Tensor* tf_tensor = TF_TensorFromTensor(tensor, &status); 
	ASSERT_TRUE(status.ok()) << status.ToString();
	ASSERT_EQ(tensor.shape().DebugString(), TF_ShapeDebugString(tf_tensor)); 
}

TEST(ShapeDebugString, RegularShape) { 
	TestShapeMatch(TensorShape({5, 4, 7})); 
} 

TEST(ShapeDebugString, ScalarShape) { 
	TestShapeMatch(TensorShape({})); 
}

} // namespace tensorflow
