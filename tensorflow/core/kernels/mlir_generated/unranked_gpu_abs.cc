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

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "mlir/ExecutionEngine/CRunnerUtils.h"  // from @llvm-project
#include "mlir/ExecutionEngine/RunnerUtils.h"  // from @llvm-project
#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor.h"

// TODO(b/169547730): Automatically generate these declarations.
extern "C" ::UnrankedMemRefType<Eigen::half> _mlir_ciface_abs_unranked_f16(
    tensorflow::OpKernelContext* ctx, ::UnrankedMemRefType<Eigen::half>* arg);
extern "C" ::UnrankedMemRefType<float> _mlir_ciface_abs_unranked_f32(
    tensorflow::OpKernelContext* ctx, ::UnrankedMemRefType<float>* arg);
extern "C" ::UnrankedMemRefType<double> _mlir_ciface_abs_unranked_f64(
    tensorflow::OpKernelContext* ctx, ::UnrankedMemRefType<double>* arg);
extern "C" ::UnrankedMemRefType<int32> _mlir_ciface_abs_unranked_i32(
    tensorflow::OpKernelContext* ctx, ::UnrankedMemRefType<int32>* arg);
extern "C" ::UnrankedMemRefType<int64> _mlir_ciface_abs_unranked_i64(
    tensorflow::OpKernelContext* ctx, ::UnrankedMemRefType<int64>* arg);

namespace tensorflow {
namespace {

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

// A simple TensorBuffer implementation that allows us to create Tensors that
// take ownership of pre-allocated memory.
template <typename ElemType>
class MlirTensorBuffer : public TensorBuffer {
 public:
  MlirTensorBuffer(const void* ptr, TensorShape shape, Allocator* allocator)
      : TensorBuffer(const_cast<void*>(ptr)),
        size_(sizeof(ElemType) * shape.num_elements()),
        allocator_(allocator) {}

  ~MlirTensorBuffer() override {
    if (data()) {
      allocator_->DeallocateRaw(data());
    }
  }

  size_t size() const override { return size_; }

  TensorBuffer* root_buffer() override { return this; }

  void FillAllocationDescription(AllocationDescription* proto) const override {
    proto->set_allocated_bytes(size_);
  }

 private:
  size_t size_;
  Allocator* allocator_;
};

template <typename ElemType>
Tensor ConvertDescriptorToTensor(
    ::UnrankedMemRefType<ElemType> unranked_descriptor, DataType tf_data_type,
    Allocator* allocator) {
  void* base_ptr = static_cast<void**>(unranked_descriptor.descriptor)[0];
  TensorShape result_shape;
  intptr_t* pointers = static_cast<intptr_t*>(unranked_descriptor.descriptor);
  for (int i = 0; i < unranked_descriptor.rank; ++i) {
    result_shape.AddDim(pointers[3 + i]);
  }
  auto* buffer =
      new MlirTensorBuffer<ElemType>(base_ptr, result_shape, allocator);
  // Tensor takes ownership of the buffer.
  Tensor tensor{tf_data_type, result_shape, buffer};
  // When Tensor is constructed, its ref-counter is incremented. We need to
  // decrement it back.
  buffer->Unref();
  return tensor;
}

}  // namespace

#define MLIR_FUNCTION(data_type) _mlir_ciface_abs_unranked_##data_type

// Generates a class derived from OpKernel with Compute function that converts
// input tensors to unranked memref descriptors and calls mlir-generated
// unranked kernel. The outputs are converted back to tensors using
// MlirTensorBuffer to take ownership of pre-allocated memory.
#define REGISTER_AND_GENERATE_KERNEL(kernel_name, type_name, data_type,     \
                                     tf_data_type)                          \
  namespace {                                                               \
  class MlirUnranked##kernel_name##type_name##Op : public OpKernel {        \
   public:                                                                  \
    MlirUnranked##kernel_name##type_name##Op(OpKernelConstruction* ctx)     \
        : OpKernel(ctx) {}                                                  \
                                                                            \
    void Compute(OpKernelContext* ctx) override {                           \
      const Tensor& input = ctx->input(0);                                  \
                                                                            \
      auto input_desc = ConvertTensorToDescriptor<data_type>(input);        \
      auto result_desc = MLIR_FUNCTION(type_name)(ctx, &input_desc);        \
      free(input_desc.descriptor);                                          \
                                                                            \
      tensorflow::AllocatorAttributes attrs;                                \
      auto* allocator = ctx->get_allocator(attrs);                          \
                                                                            \
      Tensor result_tensor = ConvertDescriptorToTensor<data_type>(          \
          result_desc, tf_data_type, allocator);                            \
      free(result_desc.descriptor);                                         \
      ctx->set_output(0, result_tensor);                                    \
    }                                                                       \
  };                                                                        \
  }                                                                         \
                                                                            \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name(#kernel_name).Device(DEVICE_GPU).TypeConstraint<data_type>("T"), \
      MlirUnranked##kernel_name##type_name##Op);

REGISTER_AND_GENERATE_KERNEL(Abs, f16, Eigen::half, DT_HALF);
REGISTER_AND_GENERATE_KERNEL(Abs, f32, float, DT_FLOAT);
REGISTER_AND_GENERATE_KERNEL(Abs, f64, double, DT_DOUBLE);
REGISTER_AND_GENERATE_KERNEL(Abs, i32, int32, DT_INT32);
REGISTER_AND_GENERATE_KERNEL(Abs, i64, int64, DT_INT64);

}  // namespace tensorflow
