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

#include <limits>

#define EIGEN_USE_THREADS
#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/list_kernels.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/kernels/concat_lib.h"
#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/util.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

#define REGISTER_TENSOR_LIST_STACK_GPU(T)                         \
  REGISTER_KERNEL_BUILDER(Name("TensorListStack")                 \
                              .TypeConstraint<T>("element_dtype") \
                              .Device(DEVICE_GPU),                \
                          TensorListStack<GPUDevice, T>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_TENSOR_LIST_STACK_GPU);
REGISTER_TENSOR_LIST_STACK_GPU(bfloat16);
TF_CALL_complex64(REGISTER_TENSOR_LIST_STACK_GPU);
TF_CALL_complex128(REGISTER_TENSOR_LIST_STACK_GPU);
TF_CALL_int64(REGISTER_TENSOR_LIST_STACK_GPU);
REGISTER_TENSOR_LIST_STACK_GPU(bool);

#undef REGISTER_TENSOR_LIST_STACK_GPU

#define REGISTER_TENSOR_LIST_PUSH_BACK_BATCH_GPU(T)               \
  REGISTER_KERNEL_BUILDER(Name("TensorListPushBackBatch")         \
                              .TypeConstraint<T>("element_dtype") \
                              .Device(DEVICE_GPU),                \
                          TensorListPushBackBatch<GPUDevice, T>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_TENSOR_LIST_PUSH_BACK_BATCH_GPU);
REGISTER_TENSOR_LIST_PUSH_BACK_BATCH_GPU(bfloat16);
TF_CALL_complex64(REGISTER_TENSOR_LIST_PUSH_BACK_BATCH_GPU);
TF_CALL_complex128(REGISTER_TENSOR_LIST_PUSH_BACK_BATCH_GPU);
TF_CALL_int64(REGISTER_TENSOR_LIST_PUSH_BACK_BATCH_GPU);
REGISTER_TENSOR_LIST_PUSH_BACK_BATCH_GPU(bool);

#undef REGISTER_TENSOR_LIST_PUSH_BACK_BATCH_GPU

#define REGISTER_TENSOR_LIST_FROM_TENSOR_GPU(T)                   \
  REGISTER_KERNEL_BUILDER(Name("TensorListFromTensor")            \
                              .TypeConstraint<T>("element_dtype") \
                              .Device(DEVICE_GPU)                 \
                              .HostMemory("element_shape"),       \
                          TensorListFromTensor<GPUDevice, T>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_TENSOR_LIST_FROM_TENSOR_GPU);
REGISTER_TENSOR_LIST_FROM_TENSOR_GPU(bfloat16);
TF_CALL_complex64(REGISTER_TENSOR_LIST_FROM_TENSOR_GPU);
TF_CALL_complex128(REGISTER_TENSOR_LIST_FROM_TENSOR_GPU);
TF_CALL_int64(REGISTER_TENSOR_LIST_FROM_TENSOR_GPU);
REGISTER_TENSOR_LIST_FROM_TENSOR_GPU(bool);

#undef REGISTER_TENSOR_LIST_FROM_TENSOR_GPU

REGISTER_UNARY_VARIANT_BINARY_OP_FUNCTION(ADD_VARIANT_BINARY_OP, DEVICE_GPU,
                                          TensorList, TensorList::kTypeName,
                                          TensorListBinaryAdd<GPUDevice>);
REGISTER_UNARY_VARIANT_UNARY_OP_FUNCTION(ZEROS_LIKE_VARIANT_UNARY_OP,
                                         DEVICE_GPU, TensorList,
                                         TensorList::kTypeName,
                                         TensorListZerosLike<GPUDevice>);

}  // namespace tensorflow
#endif  // GOOGLE_CUDA
