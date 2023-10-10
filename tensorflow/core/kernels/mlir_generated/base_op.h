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

#ifndef TENSORFLOW_CORE_KERNELS_MLIR_GENERATED_BASE_OP_H_
#define TENSORFLOW_CORE_KERNELS_MLIR_GENERATED_BASE_OP_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"  // from @llvm-project
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {

// Unranked memref descriptor as it is expected and returned by the external
// MLIR-generated "C" function.
struct UnrankedMemRef {
  int64_t rank;
  void* descriptor;
};

// Returns a pointer to an allocated MlirTensorBuffer that takes ownership of
// pre-allocated memory.
TensorBuffer* GetMlirTensorBuffer(const void* ptr, size_t size,
                                  Allocator* allocator);

/// Used to allocate descriptors on stack when they are small.

constexpr int kMaxRankForOnStackDescriptors = 10;

static constexpr size_t GetSizeOfDescriptor(int rank) {
  return sizeof(void*) * (2 * rank + 3);
}

using DescriptorBuffer =
    llvm::SmallVector<unsigned char,
                      GetSizeOfDescriptor(kMaxRankForOnStackDescriptors)>;

/// Converts tensors to memory descriptors and back.

UnrankedMemRef ConvertTensorToDescriptor(const Tensor& tensor,
                                         DescriptorBuffer& buffer);

TensorShape ExtractShapeFromDescriptor(UnrankedMemRef unranked_descriptor);

template <typename ElemType>
Tensor ConvertDescriptorToTensor(UnrankedMemRef unranked_descriptor,
                                 DataType TfDataType, Allocator* allocator) {
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

// OpKernel with Compute function that converts input tensors to unranked
// memref descriptors and calls the MLIR-generated unranked kernel. The outputs
// are converted back to tensors using MlirTensorBuffer to take ownership of
// pre-allocated memory.
template <DataType TfDataType, typename OutputDataType,
          DataType CastedTfDataType = TfDataType>
class MLIROpKernel : public OpKernel {
 public:
  explicit MLIROpKernel(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    VLOG(4) << ctx->op_kernel().TraceString(*ctx, true);

    // Convert tensor arguments to unranked memory descriptors.
    llvm::SmallVector<DescriptorBuffer, 4> buffers(ctx->num_inputs());
    llvm::SmallVector<UnrankedMemRef, 4> args;
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      args.push_back(ConvertTensorToDescriptor(ctx->input(i), buffers[i]));
    }

    UnrankedMemRef result_desc = Invoke(ctx, args);
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
        OP_REQUIRES_OK(
            ctx, output.BitcastFrom(input, CastedTfDataType, result_shape));

        ctx->set_output(0, output);
        free(result_desc.descriptor);
        return;
      }
    }

    tensorflow::AllocatorAttributes attrs;
    auto* allocator = ctx->get_allocator(attrs);
    Tensor result_tensor = ConvertDescriptorToTensor<OutputDataType>(
        result_desc, TfDataType, allocator);
    if (TfDataType != CastedTfDataType) {
      Tensor casted_result_tensor;
      OP_REQUIRES_OK(
          ctx, casted_result_tensor.BitcastFrom(result_tensor, CastedTfDataType,
                                                result_tensor.shape()));
      result_tensor = casted_result_tensor;
    }
    free(result_desc.descriptor);
    ctx->set_output(0, result_tensor);
  }

 protected:
  virtual UnrankedMemRef Invoke(
      OpKernelContext* ctx, llvm::SmallVectorImpl<UnrankedMemRef>& args) = 0;
};

/// Generate C function and kernel names.

#define MLIR_FUNCTION(tf_op, platform, input_type, output_type) \
  _mlir_ciface_##tf_op##_##platform##_##input_type##_##output_type

#define MLIR_OP(tf_op, platform, input_type, output_type) \
  Mlir##tf_op##platform##input_type##output_type##Op

/// Register kernels.

#define REGISTER_ALIASED_KERNEL(tf_op, mlir_op, platform, input_type,     \
                                output_type, additional_cstrs)            \
  REGISTER_KERNEL_BUILDER(                                                \
      Name(#tf_op)                                                        \
          .Device(DEVICE_##platform)                                      \
          .TypeConstraint<typename EnumToDataType<input_type>::Type>("T") \
              additional_cstrs,                                           \
      MLIR_OP(mlir_op, platform, input_type, output_type));

#define REGISTER_KERNEL(tf_op, platform, input_type, output_type,          \
                        additional_cstrs)                                  \
  REGISTER_ALIASED_KERNEL(tf_op, tf_op, platform, input_type, output_type, \
                          additional_cstrs)

#define REGISTER_COMPLEX_KERNEL(tf_op, platform, input_type, output_type)      \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name(#tf_op)                                                             \
          .Device(DEVICE_##platform)                                           \
          .TypeConstraint<typename EnumToDataType<input_type>::Type>("T")      \
          .TypeConstraint<typename EnumToDataType<output_type>::Type>("Tout"), \
      MLIR_OP(tf_op, platform, input_type, output_type));

#define REGISTER_KERNEL_NO_TYPE_CONSTRAINT(tf_op, platform, input_type) \
  REGISTER_KERNEL_BUILDER(Name(#tf_op).Device(DEVICE_##platform),       \
                          MLIR_OP(tf_op, platform, input_type, input_type));

/// Unary kernels.

#define GENERATE_AND_REGISTER_UNARY_KERNEL(tf_op, platform, input_type, \
                                           additional_cstrs)            \
  GENERATE_UNARY_KERNEL(tf_op, platform, input_type)                    \
  REGISTER_KERNEL(tf_op, platform, input_type, input_type, additional_cstrs)

#define GENERATE_AND_REGISTER_UNARY_KERNEL2(tf_op, platform, input_type,   \
                                            output_type, additional_cstrs) \
  GENERATE_UNARY_KERNEL(tf_op, platform, input_type, output_type)          \
  REGISTER_KERNEL(tf_op, platform, input_type, output_type, additional_cstrs)

#define GENERATE_AND_REGISTER_UNARY_KERNEL3(                              \
    tf_op, platform, input_type, output_type, casted_input_type,          \
    casted_output_type, additional_cstrs)                                 \
  GENERATE_UNARY_KERNEL3(tf_op, platform, input_type, output_type,        \
                         casted_input_type, casted_output_type)           \
  REGISTER_KERNEL(tf_op, platform, casted_input_type, casted_output_type, \
                  additional_cstrs)

#define GENERATE_AND_REGISTER_UNARY_JIT_KERNEL(tf_op, platform, input_type, \
                                               additional_cstrs)            \
  GENERATE_AND_REGISTER_UNARY_KERNEL(tf_op, platform, input_type,           \
                                     .Label(kJitKernelLabel) additional_cstrs)

#define GENERATE_UNARY_KERNEL(tf_op, platform, input_type) \
  GENERATE_UNARY_KERNEL2(tf_op, platform, input_type, input_type)

#define GENERATE_UNARY_KERNEL2(tf_op, platform, input_type, output_type)       \
  GENERATE_UNARY_KERNEL3(tf_op, platform, input_type, output_type, input_type, \
                         output_type)

#define GENERATE_UNARY_KERNEL3(tf_op, platform, input_type, output_type,     \
                               casted_input_type, casted_output_type)        \
  extern "C" void MLIR_FUNCTION(tf_op, platform, input_type, output_type)(   \
      UnrankedMemRef * result, OpKernelContext * ctx, UnrankedMemRef * arg); \
                                                                             \
  namespace {                                                                \
  class MLIR_OP(tf_op, platform, casted_input_type, casted_output_type)      \
      : public MLIROpKernel<output_type,                                     \
                            typename EnumToDataType<output_type>::Type,      \
                            casted_output_type> {                            \
   public:                                                                   \
    using MLIROpKernel::MLIROpKernel;                                        \
                                                                             \
    UnrankedMemRef Invoke(                                                   \
        OpKernelContext* ctx,                                                \
        llvm::SmallVectorImpl<UnrankedMemRef>& args) override {              \
      UnrankedMemRef result;                                                 \
      MLIR_FUNCTION(tf_op, platform, input_type, output_type)                \
      (&result, ctx, &args[0]);                                              \
      return result;                                                         \
    }                                                                        \
  };                                                                         \
  }

/// Binary kernels.

#define GENERATE_AND_REGISTER_BINARY_KERNEL(tf_op, platform, input_type, \
                                            additional_cstrs)            \
  GENERATE_BINARY_KERNEL(tf_op, platform, input_type)                    \
  REGISTER_KERNEL(tf_op, platform, input_type, input_type, additional_cstrs)

#define GENERATE_AND_REGISTER_BINARY_KERNEL2(tf_op, platform, input_type,   \
                                             output_type, additional_cstrs) \
  GENERATE_BINARY_KERNEL2(tf_op, platform, input_type, output_type)         \
  REGISTER_KERNEL(tf_op, platform, input_type, output_type, additional_cstrs)

#define GENERATE_AND_REGISTER_BINARY_KERNEL3(                             \
    tf_op, platform, input_type, output_type, casted_input_type,          \
    casted_output_type, additional_cstrs)                                 \
  GENERATE_BINARY_KERNEL3(tf_op, platform, input_type, output_type,       \
                          casted_input_type, casted_output_type)          \
  REGISTER_KERNEL(tf_op, platform, casted_input_type, casted_output_type, \
                  additional_cstrs)

#define GENERATE_AND_REGISTER_BINARY_JIT_KERNEL(tf_op, platform, input_type, \
                                                additional_cstrs)            \
  GENERATE_AND_REGISTER_BINARY_KERNEL(                                       \
      tf_op, platform, input_type, .Label(kJitKernelLabel) additional_cstrs)

#define GENERATE_BINARY_KERNEL(tf_op, platform, input_type) \
  GENERATE_BINARY_KERNEL2(tf_op, platform, input_type, input_type)

#define GENERATE_BINARY_KERNEL2(tf_op, platform, input_type, output_type) \
  GENERATE_BINARY_KERNEL3(tf_op, platform, input_type, output_type,       \
                          input_type, output_type)

#define GENERATE_BINARY_KERNEL3(tf_op, platform, input_type, output_type,    \
                                casted_input_type, casted_output_type)       \
  extern "C" void MLIR_FUNCTION(tf_op, platform, input_type, output_type)(   \
      UnrankedMemRef * result, OpKernelContext * ctx, UnrankedMemRef * arg0, \
      UnrankedMemRef * arg1);                                                \
                                                                             \
  namespace {                                                                \
  class MLIR_OP(tf_op, platform, casted_input_type, casted_output_type)      \
      : public MLIROpKernel<output_type,                                     \
                            typename EnumToDataType<output_type>::Type,      \
                            casted_output_type> {                            \
   public:                                                                   \
    using MLIROpKernel::MLIROpKernel;                                        \
                                                                             \
    UnrankedMemRef Invoke(                                                   \
        OpKernelContext* ctx,                                                \
        llvm::SmallVectorImpl<UnrankedMemRef>& args) override {              \
      UnrankedMemRef result;                                                 \
      MLIR_FUNCTION(tf_op, platform, input_type, output_type)                \
      (&result, ctx, &args[0], &args[1]);                                    \
      return result;                                                         \
    }                                                                        \
  };                                                                         \
  }

/// Ternary kernels.

#define GENERATE_AND_REGISTER_TERNARY_KERNEL(tf_op, platform, input_type, \
                                             additional_cstrs)            \
  GENERATE_TERNARY_KERNEL(tf_op, platform, input_type)                    \
  REGISTER_KERNEL(tf_op, platform, input_type, input_type, additional_cstrs)

#define GENERATE_AND_REGISTER_TERNARY_KERNEL2(tf_op, platform, input_type,   \
                                              output_type, additional_cstrs) \
  GENERATE_TERNARY_KERNEL2(tf_op, platform, input_type, output_type)         \
  REGISTER_KERNEL(tf_op, platform, input_type, output_type, additional_cstrs)

#define GENERATE_AND_REGISTER_TERNARY_KERNEL3(                            \
    tf_op, platform, input_type, output_type, casted_input_type,          \
    casted_output_type, additional_cstrs)                                 \
  GENERATE_TERNARY_KERNEL3(tf_op, platform, input_type, output_type,      \
                           casted_input_type, casted_output_type)         \
  REGISTER_KERNEL(tf_op, platform, casted_input_type, casted_output_type, \
                  additional_cstrs)

#define GENERATE_AND_REGISTER_TERNARY_JIT_KERNEL(tf_op, platform, input_type, \
                                                 additional_cstrs)            \
  GENERATE_AND_REGISTER_TERNARY_KERNEL(                                       \
      tf_op, platform, input_type, .Label(kJitKernelLabel) additional_cstrs)

#define GENERATE_TERNARY_KERNEL(tf_op, platform, input_type) \
  GENERATE_TERNARY_KERNEL2(tf_op, platform, input_type, input_type)

#define GENERATE_TERNARY_KERNEL2(tf_op, platform, input_type, output_type) \
  GENERATE_TERNARY_KERNEL3(tf_op, platform, input_type, output_type,       \
                           input_type, output_type)

#define GENERATE_TERNARY_KERNEL3(tf_op, platform, input_type, output_type,   \
                                 casted_input_type, casted_output_type)      \
  extern "C" void MLIR_FUNCTION(tf_op, platform, input_type, output_type)(   \
      UnrankedMemRef * result, OpKernelContext * ctx, UnrankedMemRef * arg0, \
      UnrankedMemRef * arg1, UnrankedMemRef * arg2);                         \
                                                                             \
  namespace {                                                                \
  class MLIR_OP(tf_op, platform, casted_input_type, casted_output_type)      \
      : public MLIROpKernel<output_type,                                     \
                            typename EnumToDataType<output_type>::Type,      \
                            casted_output_type> {                            \
   public:                                                                   \
    using MLIROpKernel::MLIROpKernel;                                        \
                                                                             \
    UnrankedMemRef Invoke(                                                   \
        OpKernelContext* ctx,                                                \
        llvm::SmallVectorImpl<UnrankedMemRef>& args) override {              \
      UnrankedMemRef result;                                                 \
      MLIR_FUNCTION(tf_op, platform, input_type, output_type)                \
      (&result, ctx, &args[0], &args[1], &args[2]);                          \
      return result;                                                         \
    }                                                                        \
  };                                                                         \
  }

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_MLIR_GENERATED_BASE_OP_H_
