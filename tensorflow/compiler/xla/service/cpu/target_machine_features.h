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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_TARGET_MACHINE_FEATURES_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_TARGET_MACHINE_FEATURES_H_

#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/core/lib/gtl/flatmap.h"

namespace xla {
namespace cpu {

// Abstract interface for classes providing information about the target we're
// compiling for.
class TargetMachineFeatures {
 public:
  static constexpr int kX86AvxVectorByteSize = 32;

  // Input and output tensor buffers must be aligned to this many bytes if we
  // want to call an Eigen backed GEMM or Convolution.
  static constexpr int kEigenExpectedTensorAlignment = 16;

  // Return the vectorization factor, which is the number of bytes of data
  // explicitly vectorized routines will try to process at once.
  virtual int vectorization_factor_in_bytes() const = 0;

  // Return the size of the largest vector size in bytes.  We need to pass in
  // "function" since llvm functions can contain annotations for specializing
  // them to specific micro-architectures (though currently XLA does not use
  // this functionality).
  virtual int vector_register_byte_size(
      const llvm::Function& function) const = 0;

  // Return the number of elements of type `type` that can fit into the largest
  // vector register available.  We need to pass in "function" since llvm
  // functions can contain annotations for specializing them to specific
  // micro-architectures (though currently XLA does not use this functionality).
  virtual int vector_register_num_elements(const llvm::Function& function,
                                           PrimitiveType type) const = 0;

  // Returns the minimum alignment for a buffer of size size_bytes.
  virtual int64 minimum_alignment_for_allocation(int64 size_bytes) const = 0;

  virtual ~TargetMachineFeatures() = default;
};

// Implements the TargetMachineFeatures interface using an llvm::TargetMachine.
class LLVMTargetMachineFeatures : public TargetMachineFeatures {
 public:
  static constexpr int kX86AvxVectorByteSize = 32;

  LLVMTargetMachineFeatures(llvm::TargetMachine* target_machine)
      : target_machine_(target_machine) {}

  int vectorization_factor_in_bytes() const override {
    // Ideally this should be a function of the cache line size (which we can
    // get from llvm::TargetTransformInfo::getCacheLineSize) of the target
    // machine.  Guess a value of 128 bytes for now.
    return 128;
  }

  int vector_register_byte_size(const llvm::Function& function) const override {
    llvm::TargetTransformInfo* tti = GetTargetTransformInfoFor(function);
    return tti->getRegisterBitWidth(/*Vector=*/true) / 8;
  }

  int vector_register_num_elements(const llvm::Function& function,
                                   PrimitiveType type) const override {
    return vector_register_byte_size(function) /
           (primitive_util::BitWidth(type) / 8);
  }

  int64 minimum_alignment_for_allocation(int64 size_bytes) const override;

 private:
  llvm::TargetTransformInfo* GetTargetTransformInfoFor(
      const llvm::Function& function) const;

  // This cache saves us from having to create a llvm::TargetTransformInfo for
  // every call to GetTargetTransformInfoFor (creating a TargetTransformInfo
  // costs one heap allocation on X86).
  //
  // This is mutated from within `GetTargetTransformInfoFor` which is
  // semantically a getter (and thus `const`); and is therefore declared
  // mutable.  Making this mutable is okay because it has cache semantics.
  mutable tensorflow::gtl::FlatMap<const llvm::Function*,
                                   llvm::TargetTransformInfo>
      target_transform_info_cache_;
  llvm::TargetMachine* target_machine_;
};

}  // namespace cpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_TARGET_MACHINE_FEATURES_H_
