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
#ifndef TENSORFLOW_C_EAGER_C_API_TEST_UTIL_H_
#define TENSORFLOW_C_EAGER_C_API_TEST_UTIL_H_

#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/core/platform/types.h"

// Return a tensor handle containing a float scalar
TFE_TensorHandle* TestScalarTensorHandle(float value);

// Return a tensor handle containing a int scalar
TFE_TensorHandle* TestScalarTensorHandle(int value);

// Return a tensor handle containing a bool scalar
TFE_TensorHandle* TestScalarTensorHandle(bool value);

// Return a tensor handle containing a 2x2 matrix of doubles
TFE_TensorHandle* DoubleTestMatrixTensorHandle();

// Return a tensor handle containing a 2x2 matrix of floats
TFE_TensorHandle* TestMatrixTensorHandle();

// Return a tensor handle containing a 100x100 matrix of floats
TFE_TensorHandle* TestMatrixTensorHandle100x100();

// Return a tensor handle containing a 3x2 matrix of doubles
TFE_TensorHandle* DoubleTestMatrixTensorHandle3X2();

// Return a tensor handle containing a 3x2 matrix of floats
TFE_TensorHandle* TestMatrixTensorHandle3X2();

// Return an add op multiplying `a` by `b`.
TFE_Op* AddOp(TFE_Context* ctx, TFE_TensorHandle* a, TFE_TensorHandle* b);

// Return a matmul op multiplying `a` by `b`.
TFE_Op* MatMulOp(TFE_Context* ctx, TFE_TensorHandle* a, TFE_TensorHandle* b);

// Return an identity op.
TFE_Op* IdentityOp(TFE_Context* ctx, TFE_TensorHandle* a);

// Return a shape op fetching the shape of `a`.
TFE_Op* ShapeOp(TFE_Context* ctx, TFE_TensorHandle* a);

// Return an 1-D INT32 tensor containing a single value 1.
TFE_TensorHandle* TestAxisTensorHandle();

// Return an op taking minimum of `input` long `axis` dimension.
TFE_Op* MinOp(TFE_Context* ctx, TFE_TensorHandle* input,
              TFE_TensorHandle* axis);

// If there is a device of type `device_type`, returns true
// and sets 'device_name' accordingly.
// `device_type` must be either "GPU" or "TPU".
bool GetDeviceName(TFE_Context* ctx, tensorflow::string* device_name,
                   const char* device_type);

#endif  // TENSORFLOW_C_EAGER_C_API_TEST_UTIL_H_
