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

#ifndef TENSORFLOW_CORE_KERNELS_MLIR_GENERATED_UNRANKED_OP_GPU_ABS_H_
#define TENSORFLOW_CORE_KERNELS_MLIR_GENERATED_UNRANKED_OP_GPU_ABS_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"  // from @llvm-project
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {

// A type-erased version of the UnrankedMemRefType to allow it to be used
// as the return type of an extern "C" function on windows.
struct UntypedUnrankedMemRefType {
  int64_t rank;
  void* descriptor;
};

template <typename ElemType>
UnrankedMemRefType<ElemType> ConvertToTyped(UntypedUnrankedMemRefType desc) {
  return {desc.rank, desc.descriptor};
}

// Returns a pointer to an allocated MlirTensorBuffer that takes ownership of
// pre-allocated memory.
TensorBuffer* GetMlirTensorBuffer(const void* ptr, size_t size,
                                  Allocator* allocator);

template <typename ElemType>
::UnrankedMemRefType<ElemType> ConvertTensorToDescriptor(const Tensor& tensor) {
  ::UnrankedMemRefType<ElemType> result;
  result.rank = tensor.dims();
  result.descriptor = malloc(sizeof(void*) * (2 * result.rank + 3));

  // Fill the descriptor.
  void** pointers = static_cast<void**>(result.descriptor);
  pointers[0] = tensor.data();
  pointers[1] = tensor.data();
  intptr_t* int_pointers = static_cast<intptr_t*>(result.descriptor);
  int_pointers[2] = 0;
  // Fill size.
  for (int i = 0; i < result.rank; ++i) {
    int_pointers[3 + i] = tensor.dim_size(i);
  }
  // Fill strides.
  int64_t stride = 1;
  for (int i = result.rank - 1; i >= 0; --i) {
    int_pointers[i + result.rank + 3] = stride;
    stride *= tensor.dim_size(i);
  }
  return result;
}

template <typename ElemType>
TensorShape ExtractShapeFromDescriptor(
    ::UnrankedMemRefType<ElemType> unranked_descriptor) {
  TensorShape shape;
  intptr_t* pointers = static_cast<intptr_t*>(unranked_descriptor.descriptor);
  for (int i = 0; i < unranked_descriptor.rank; ++i) {
    shape.AddDim(pointers[3 + i]);
  }
  return shape;
}

template <typename ElemType>
Tensor ConvertDescriptorToTensor(
    ::UnrankedMemRefType<ElemType> unranked_descriptor, DataType TfDataType,
    Allocator* allocator) {
  void* base_ptr = static_cast<void**>(unranked_descriptor.descriptor)[0];
  TensorShape result_shape = ExtractShapeFromDescriptor(unranked_descriptor);
  TensorBuffer* buffer = GetMlirTensorBuffer(
      base_ptr, sizeof(ElemType) * result_shape.num_elements(), allocator);

  // Tensor takes ownership of the buffer.
  Tensor tensor{TfDataType, result_shape, buffer};
  // When Tensor is constructed, its ref-counter is incremented. We need to
  // decrement it back.
  buffer->Unref();
  return tensor;
}

template <DataType TfDataType, typename OutputDataType, typename Kernel,
          typename InputDataType = OutputDataType>
class MlirUnrankedOp : public OpKernel {
 public:
  explicit MlirUnrankedOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    llvm::SmallVector<::UnrankedMemRefType<InputDataType>, 2> input_descs;
    for (int i = 0, end = ctx->num_inputs(); i < end; ++i) {
      input_descs.push_back(
          std::move(ConvertTensorToDescriptor<InputDataType>(ctx->input(i))));
    }
    auto result_desc = Kernel::Invoke(ctx, input_descs);
    for (const auto& input_desc : input_descs) {
      free(input_desc.descriptor);
    }
    if (!ctx->status().ok()) {
      free(result_desc.descriptor);
      return;
    }
    void* result_data_ptr = static_cast<void**>(result_desc.descriptor)[0];

    // Detect input buffer reuse.
    for (int i = 0, end = ctx->num_inputs(); i < end; ++i) {
      const Tensor& input = ctx->input(i);
      if (input.data() == result_data_ptr) {
        // Run a bitcast in case the output type is different.
        Tensor output;
        TensorShape result_shape = ExtractShapeFromDescriptor(result_desc);
        OP_REQUIRES_OK(ctx,
                       output.BitcastFrom(input, TfDataType, result_shape));

        ctx->set_output(0, output);
        free(result_desc.descriptor);
        return;
      }
    }

    tensorflow::AllocatorAttributes attrs;
    auto* allocator = ctx->get_allocator(attrs);
    Tensor result_tensor = ConvertDescriptorToTensor<OutputDataType>(
        result_desc, TfDataType, allocator);
    free(result_desc.descriptor);
    ctx->set_output(0, result_tensor);
  }
};

#define MLIR_FUNCTION(tf_op, mlir_type) _mlir_ciface_##tf_op##_##mlir_type

#define REGISTER_KERNEL(tf_op, mlir_type, data_type)                  \
  REGISTER_KERNEL_BUILDER(                                            \
      Name(#tf_op).Device(DEVICE_GPU).TypeConstraint<data_type>("T"), \
      MlirUnranked##tf_op##mlir_type##Op);

#define REGISTER_COMPLEX_KERNEL(tf_op, mlir_type, data_type, input_data_type) \
  REGISTER_KERNEL_BUILDER(Name(#tf_op)                                        \
                              .Device(DEVICE_GPU)                             \
                              .TypeConstraint<input_data_type>("T")           \
                              .TypeConstraint<data_type>("Tout"),             \
                          MlirUnranked##tf_op##mlir_type##Op);

#define REGISTER_KERNEL_NO_TYPE_CONSTRAINT(tf_op, mlir_type) \
  REGISTER_KERNEL_BUILDER(Name(#tf_op).Device(DEVICE_GPU),   \
                          MlirUnranked##tf_op##mlir_type##Op);

// OpKernel with Compute function that converts input tensors to unranked
// memref descriptors and calls mlir-generated unranked kernel. The outputs
// are converted back to tensors using MlirTensorBuffer to take ownership of
// pre-allocated memory.
#define GENERATE_AND_REGISTER_BINARY_KERNEL(tf_op, mlir_type, tf_data_type, \
                                            data_type)                      \
  GENERATE_BINARY_KERNEL(tf_op, mlir_type, tf_data_type, data_type)         \
  REGISTER_KERNEL(tf_op, mlir_type, data_type)

#define GENERATE_BINARY_KERNEL(tf_op, mlir_type, tf_data_type, data_type) \
  extern "C" UntypedUnrankedMemRefType MLIR_FUNCTION(tf_op, mlir_type)(   \
      tensorflow::OpKernelContext * ctx,                                  \
      const ::UnrankedMemRefType<data_type>* arg1,                        \
      const ::UnrankedMemRefType<data_type>* arg2);                       \
                                                                          \
  namespace {                                                             \
  class MlirUnranked##tf_op##mlir_type##Op                                \
      : public MlirUnrankedOp<tf_data_type, data_type,                    \
                              MlirUnranked##tf_op##mlir_type##Op> {       \
   public:                                                                \
    using MlirUnrankedOp::MlirUnrankedOp;                                 \
                                                                          \
    static ::UnrankedMemRefType<data_type> Invoke(                        \
        OpKernelContext* ctx,                                             \
        llvm::ArrayRef<::UnrankedMemRefType<data_type>> args) {           \
      return ConvertToTyped<data_type>(                                   \
          MLIR_FUNCTION(tf_op, mlir_type)(ctx, &args[0], &args[1]));      \
    }                                                                     \
  };                                                                      \
  }

#define GENERATE_AND_REGISTER_UNARY_KERNEL(tf_op, mlir_type, tf_data_type, \
                                           data_type)                      \
  GENERATE_UNARY_KERNEL(tf_op, mlir_type, tf_data_type, data_type)         \
  REGISTER_KERNEL(tf_op, mlir_type, data_type)

#define GENERATE_UNARY_KERNEL(tf_op, mlir_type, tf_data_type, data_type) \
  GENERATE_UNARY_KERNEL2(tf_op, mlir_type, tf_data_type, data_type, data_type)

#define GENERATE_UNARY_KERNEL2(tf_op, mlir_type, tf_data_type, data_type, \
                               input_data_type)                           \
  extern "C" UntypedUnrankedMemRefType MLIR_FUNCTION(tf_op, mlir_type)(   \
      tensorflow::OpKernelContext * ctx,                                  \
      const ::UnrankedMemRefType<input_data_type>* arg);                  \
                                                                          \
  namespace {                                                             \
  class MlirUnranked##tf_op##mlir_type##Op                                \
      : public MlirUnrankedOp<tf_data_type, data_type,                    \
                              MlirUnranked##tf_op##mlir_type##Op,         \
                              input_data_type> {                          \
   public:                                                                \
    using MlirUnrankedOp::MlirUnrankedOp;                                 \
                                                                          \
    static ::UnrankedMemRefType<data_type> Invoke(                        \
        OpKernelContext* ctx,                                             \
        llvm::ArrayRef<::UnrankedMemRefType<input_data_type>> args) {     \
      return ConvertToTyped<data_type>(                                   \
          MLIR_FUNCTION(tf_op, mlir_type)(ctx, &args[0]));                \
    }                                                                     \
  };                                                                      \
  }

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_MLIR_GENERATED_UNRANKED_OP_GPU_ABS_H_
