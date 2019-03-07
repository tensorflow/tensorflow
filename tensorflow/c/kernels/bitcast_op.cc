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

#include <sstream>

#include "tensorflow/c/kernels.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/selective_registration.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/types.h"

// BitcastOp implements a bitcast kernel, creating an output tensor that shares
// the same data buffer as the input but with a different shape and/or data
// type. Its inputs are:
//
//   * the input tensor
//   * an attribute named "T" containing the TF_DataType of the input tensor
//   * an attribute named "type" containing the TF_DataType of the output tensor
//
// Given an input tensor of shape [...], if the input DataType "T" is larger
// than the output DataType "type", then the shape changes from [...]
// to [..., sizeof(T)/sizeof(type)].
//
// If "T" is smaller than "type", the operator requires that the rightmost
// dimension be equal to sizeof(type)/sizeof(T). The shape then goes from
// [..., sizeof(type)/sizeof(T)] to [...].
//
// Bitcast is implemented as a low-level cast, so machines with different endian
// orderings will give different results.
typedef struct BitcastOp {
  TF_DataType input_data_type;
  TF_DataType output_data_type;
  size_t in_size;
  size_t out_size;
} BitcastOp;

static void* BitcastOp_Create(TF_OpKernelConstruction* ctx) {
  auto* kernel = new BitcastOp;

  TF_Status* s = TF_NewStatus();
  TF_OpKernelConstruction_GetAttrType(ctx, "T", &kernel->input_data_type, s);

  if (TF_GetCode(s) == TF_OK) {
    TF_OpKernelConstruction_GetAttrType(ctx, "type", &kernel->output_data_type,
                                        s);
  }

  if (TF_GetCode(s) == TF_OK) {
    kernel->in_size = TF_DataTypeSize(kernel->input_data_type);
    kernel->out_size = TF_DataTypeSize(kernel->output_data_type);

    size_t check_size = std::max(kernel->in_size, kernel->out_size) %
                        std::min(kernel->in_size, kernel->out_size);
    if (check_size != 0) {
      std::ostringstream err;
      err << "cannot convert between datatype " << kernel->input_data_type
          << " and " << kernel->output_data_type;
      TF_SetStatus(s, TF_INVALID_ARGUMENT, err.str().c_str());
    }
  }

  if (TF_GetCode(s) != TF_OK) {
    TF_OpKernelConstruction_Failure(ctx, s);
    delete kernel;
    kernel = nullptr;
  }

  TF_DeleteStatus(s);
  return kernel;
}

static void BitcastOp_Delete(void* kernel) {
  delete static_cast<BitcastOp*>(kernel);
}

static void BitcastOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* k = static_cast<BitcastOp*>(kernel);
  int dim_count = 0;

  TF_Tensor* tensor;
  TF_Status* status = TF_NewStatus();
  TF_GetInput(ctx, 0, &tensor, status);
  if (TF_GetCode(status) == TF_OK) {
    dim_count = TF_NumDims(tensor);
    if (!(k->in_size >= k->out_size ||
          (dim_count > 0 &&
           TF_Dim(tensor, dim_count - 1) == k->out_size / k->in_size))) {
      std::ostringstream err;
      err << "Cannot bitcast from " << k->input_data_type << " to "
          << k->output_data_type;
      TF_SetStatus(status, TF_INVALID_ARGUMENT, err.str().c_str());
    }
  }

  if (TF_GetCode(status) == TF_OK) {
    auto* dims = new int64_t[dim_count + 1];
    int new_dim_count = dim_count;
    for (int dim = 0; dim < dim_count; ++dim) {
      dims[dim] = TF_Dim(tensor, dim);
    }
    if (k->out_size < k->in_size) {
      dims[new_dim_count++] = static_cast<int64_t>(k->in_size / k->out_size);
    } else if (k->out_size > k->in_size) {
      --new_dim_count;
    }

    TF_Tensor* output = TF_AllocateTensor(k->output_data_type, dims, 0,
                                          TF_DataTypeSize(k->output_data_type));
    TF_TensorBitcastFrom(tensor, k->output_data_type, output, dims,
                         new_dim_count, status);
    if (TF_GetCode(status) == TF_OK) {
      TF_SetOutput(ctx, 0, output, status);
    }
    delete[] dims;
    TF_DeleteTensor(output);
  }

  if (TF_GetCode(status) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status);
  }
  TF_DeleteStatus(status);
  TF_DeleteTensor(tensor);
}

static void RegisterBitcastOp() {
  TF_Status* status = TF_NewStatus();

  {
    auto* builder = TF_NewKernelBuilder("Bitcast", tensorflow::DEVICE_CPU,
                                        &BitcastOp_Create, &BitcastOp_Compute,
                                        &BitcastOp_Delete);
    TF_RegisterKernelBuilder("BitcastOp", builder, status);
    CHECK_EQ(TF_OK, TF_GetCode(status))
        << "Error while registering bitcast kernel";
  }

#if GOOGLE_CUDA
  {
    auto* builder = TF_NewKernelBuilder("Bitcast", tensorflow::DEVICE_GPU,
                                        &BitcastOp_Create, &BitcastOp_Compute,
                                        &BitcastOp_Delete);
    TF_RegisterKernelBuilder("BitcastOp", builder, status);
    CHECK_EQ(TF_OK, TF_GetCode(status))
        << "Error while registering CUDA bitcast kernel";
  }
#endif

  TF_DeleteStatus(status);
}

// A dummy static variable initialized by a lambda whose side-effect is to
// register the bitcast kernel.
static bool BitcastOpIsRegistered = []() {
  if (SHOULD_REGISTER_OP_KERNEL("BitcastOp")) {
    RegisterBitcastOp();
  }
  return true;
}();
