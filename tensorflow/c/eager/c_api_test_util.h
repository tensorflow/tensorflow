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
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/tensorflow_server.pb.h"

// Return a tensor handle containing a float scalar
TFE_TensorHandle* TestScalarTensorHandle(TFE_Context* ctx, float value);

// Return a tensor handle containing a int scalar
TFE_TensorHandle* TestScalarTensorHandle(TFE_Context* ctx, int value);

// Return a tensor handle containing a bool scalar
TFE_TensorHandle* TestScalarTensorHandle(TFE_Context* ctx, bool value);

// Return a tensor handle containing a tstring scalar
TFE_TensorHandle* TestScalarTensorHandle(TFE_Context* ctx,
                                         const tensorflow::tstring& value);

// Return a tensor handle containing a 2x2 matrix of doubles
TFE_TensorHandle* DoubleTestMatrixTensorHandle(TFE_Context* ctx);

// Return a tensor handle containing a 2x2 matrix of floats
TFE_TensorHandle* TestMatrixTensorHandle(TFE_Context* ctx);

// Return a tensor handle containing 2D matrix containing given data and
// dimensions
TFE_TensorHandle* TestMatrixTensorHandleWithInput(TFE_Context* ctx,
                                                  float data[], int64_t dims[],
                                                  int num_dims);

// Get a Matrix TensorHandle with given float values and dimensions
TFE_TensorHandle* TestTensorHandleWithDimsFloat(TFE_Context* ctx, float data[],
                                                int64_t dims[], int num_dims);

// Get a Matrix TensorHandle with given int values and dimensions
TFE_TensorHandle* TestTensorHandleWithDimsInt(TFE_Context* ctx, int data[],
                                              int64_t dims[], int num_dims);

// Return a tensor handle with given type, values and dimensions.
template <class T, TF_DataType datatype>
TFE_TensorHandle* TestTensorHandleWithDims(TFE_Context* ctx, const T* data,
                                           const int64_t* dims, int num_dims) {
  TF_Status* status = TF_NewStatus();
  TF_Tensor* t = TFE_AllocateHostTensor(ctx, datatype, dims, num_dims, status);
  memcpy(TF_TensorData(t), data, TF_TensorByteSize(t));
  TFE_TensorHandle* th = TFE_NewTensorHandleFromTensor(ctx, t, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteTensor(t);
  TF_DeleteStatus(status);
  return th;
}

// Return a scalar tensor handle with given values.
template <class T, TF_DataType datatype>
TFE_TensorHandle* TestScalarTensorHandle(TFE_Context* ctx, const T value) {
  T data[] = {value};
  return TestTensorHandleWithDims<T, datatype>(ctx, data, nullptr, 0);
}

// Return a tensor handle containing a 100x100 matrix of floats
TFE_TensorHandle* TestMatrixTensorHandle100x100(TFE_Context* ctx);

// Return a tensor handle containing a 3x2 matrix of doubles
TFE_TensorHandle* DoubleTestMatrixTensorHandle3X2(TFE_Context* ctx);

// Return a tensor handle containing a 3x2 matrix of floats
TFE_TensorHandle* TestMatrixTensorHandle3X2(TFE_Context* ctx);

// Return a variable handle referring to a variable with the given initial value
// on the given device.
TFE_TensorHandle* TestVariable(TFE_Context* ctx, float value,
                               const tensorflow::string& device_name = "");

// Return an add op multiplying `a` by `b`.
TFE_Op* AddOp(TFE_Context* ctx, TFE_TensorHandle* a, TFE_TensorHandle* b);

// Return a matmul op multiplying `a` by `b`.
TFE_Op* MatMulOp(TFE_Context* ctx, TFE_TensorHandle* a, TFE_TensorHandle* b);

// Return an identity op.
TFE_Op* IdentityOp(TFE_Context* ctx, TFE_TensorHandle* a);

// Return a shape op fetching the shape of `a`.
TFE_Op* ShapeOp(TFE_Context* ctx, TFE_TensorHandle* a);

// Return an 1-D INT32 tensor containing a single value 1.
TFE_TensorHandle* TestAxisTensorHandle(TFE_Context* ctx);

// Return an op taking minimum of `input` long `axis` dimension.
TFE_Op* MinOp(TFE_Context* ctx, TFE_TensorHandle* input,
              TFE_TensorHandle* axis);

// If there is a device of type `device_type`, returns true
// and sets 'device_name' accordingly.
// `device_type` must be either "GPU" or "TPU".
bool GetDeviceName(TFE_Context* ctx, tensorflow::string* device_name,
                   const char* device_type);

// Create a ServerDef with the given `job_name` and add `num_tasks` tasks in it.
tensorflow::ServerDef GetServerDef(const tensorflow::string& job_name,
                                   int num_tasks);

// Create a ServerDef with job name "localhost" and add `num_tasks` tasks in it.
tensorflow::ServerDef GetServerDef(int num_tasks);

#endif  // TENSORFLOW_C_EAGER_C_API_TEST_UTIL_H_
