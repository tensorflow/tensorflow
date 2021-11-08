/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_JIT_TF_CPURT_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_JIT_TF_CPURT_H_

#include <utility>

#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/runtime_fallback/util/type_util.h"
#include "tensorflow/core/tfrt/utils/fallback_tensor.h"
#include "tfrt/cpu/jit/cpurt.h"  // from @tf_runtime
#include "tfrt/dtype/dtype.h"  // from @tf_runtime

namespace tensorflow {

// A set of helper classes to convert results returned from the compiled
// functions (memrefs or async memrefs) to the Tensorflow Tensors that can be
// seamlessly passed to the Tensorflow fallback kernels.

// MemrefTensorBuffer wraps a memref returned from the compiled kernel into
// the Tensorflow `TensorBuffer` that can be used to construct a `Tensor`.
class MemrefTensorBuffer : public TensorBuffer {
 public:
  MemrefTensorBuffer(void* base_ptr, void* data, size_t size, bool owner)
      : TensorBuffer(data), base_ptr_(base_ptr), size_(size), owner_(owner) {}

  ~MemrefTensorBuffer() override {
    if (owner_) free(base_ptr_);
  }

  void FillAllocationDescription(AllocationDescription* proto) const override {
    proto->set_requested_bytes(size());
    proto->set_allocator_name("tf_cpurt");
  }

  size_t size() const override { return size_; }
  bool OwnsMemory() const override { return owner_; }
  TensorBuffer* root_buffer() override { return this; }

 private:
  void* base_ptr_;
  size_t size_;
  bool owner_;
};

// Keep track of compiled kernel operands to detect input to output forwarding.
//
// Reuse conversion context as a kernel context for convenience, can be a
// separate allocation if needed.
struct TensorflowConversionContext
    : public tfrt::cpu::jit::Executable::KernelContext {
  explicit TensorflowConversionContext(size_t num_operands) {
    tensor_operands.reserve(num_operands);
  }

  // Ensure that the context is always moved around instead of copying.
  TensorflowConversionContext(const TensorflowConversionContext&) = delete;
  TensorflowConversionContext(TensorflowConversionContext&&) = default;

  llvm::SmallDenseMap<const void*, const Tensor*> tensor_operands;

  void* forward(size_t size, size_t alignment,
                llvm::ArrayRef<unsigned> candidates) override {
    // TODO(ecg): Do the real buffer forwarding here.
    return nullptr;
  }
};

namespace internal {
// The returned memref can point into statically allocated memory that we can't
// pass to `free` (memref.global). The LLVM lowering of `memref.global` sets the
// allocated pointer to the magic value 0xDEADBEEF.
template <typename T, int rank>
inline bool IsStaticStorageDuration(StridedMemRefType<T, rank>* memref) {
  return reinterpret_cast<std::intptr_t>(memref->basePtr) == 0xDEADBEEF;
}
}  // namespace internal

// Converts StridedMemrefType to the tensorflow::Tensor. This struct satisfies
// ReturnStridedMemref's concept (see cpurt.h).
struct ConvertTensor {
  using ResultType = tensorflow::tfrt_stub::FallbackTensor;
  using ConversionContext = TensorflowConversionContext;

  template <typename T, int rank>
  static llvm::ArrayRef<int64_t> Sizes(StridedMemRefType<T, rank>* memref) {
    return memref->sizes;
  }

  template <typename T>
  static llvm::ArrayRef<int64_t> Sizes(StridedMemRefType<T, 0>* memref) {
    return {};
  }

  template <typename T, int rank>
  static tensorflow::Tensor Convert(const ConversionContext& ctx,
                                    void* memref_ptr) {
    auto* memref = static_cast<StridedMemRefType<T, rank>*>(memref_ptr);
    auto memref_sizes = Sizes(memref);

    // Maybe forward operand tensor to the result.
    auto operand = ctx.tensor_operands.find(memref->data);
    if (operand != ctx.tensor_operands.end()) return *operand->second;

    // Build a Tensorflow TensorShape from memref sizes. It should never fail.
    tensorflow::TensorShape shape;
    auto st = tensorflow::TensorShapeUtils::MakeShape(memref_sizes, &shape);
    assert(st.ok() && "failed to build a TensorShape from memref sizes");
    (void)st;

    // Size of the memref in bytes.
    size_t size = sizeof(T);
    for (int i = 0; i < rank; ++i) size *= memref_sizes[i];

    // Create a TensorBuffer from the returned memref.
    auto dtype = tfd::GetTfDataType(tfrt::GetDType<T>());
    TF_ANNOTATE_MEMORY_IS_INITIALIZED(memref->data, size);
    auto* buffer = new MemrefTensorBuffer(
        memref->basePtr, memref->data, size,
        /*owner=*/!internal::IsStaticStorageDuration(memref));

    // Construct a tensor from the memory buffer.
    auto ptr = core::RefCountPtr<MemrefTensorBuffer>(buffer);
    tensorflow::Tensor tensor(dtype, std::move(shape), std::move(ptr));

    // Incorrect alignment will lead to a segfault in the downstream Tensorflow
    // kernels, check it before returning to the runtime.
    if (internal::IsStaticStorageDuration(memref)) {
      DCHECK(tensor.IsAligned()) << "global memref is not aligned";
    } else {
      DCHECK(tensor.IsAligned()) << "allocated memref is not aligned";
    }

    return tensor;
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_JIT_TF_CPURT_H_
