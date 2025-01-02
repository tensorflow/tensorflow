/* Copyright 2018 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_CODEGEN_TARGET_MACHINE_FEATURES_H_
#define XLA_BACKENDS_CPU_CODEGEN_TARGET_MACHINE_FEATURES_H_

#include <cstdint>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

// Resolve vectorization and alignment properties from the LLVM TargetMachine.
class TargetMachineFeatures {
 public:
  static constexpr int32_t kX86AvxVectorByteSize = 32;

  // Input and output tensor buffers must be aligned to this many bytes if we
  // want to call an Eigen backed GEMM or Convolution.
  static constexpr int32_t kEigenExpectedTensorAlignment = 16;

  explicit TargetMachineFeatures(llvm::TargetMachine* target_machine);
  virtual ~TargetMachineFeatures() = default;

  TargetMachineFeatures(TargetMachineFeatures&&) = default;
  TargetMachineFeatures& operator=(TargetMachineFeatures&&) = default;

  // Return the vectorization factor, which is the number of bytes of data
  // explicitly vectorized routines will try to process at once.
  virtual int32_t vectorization_factor_in_bytes() const;

  // Return the size of the largest vector size in bytes.  We need to pass in
  // "function" since llvm functions can contain annotations for specializing
  // them to specific micro-architectures (though currently XLA does not use
  // this functionality).
  virtual int32_t vector_register_byte_size(const llvm::Function& fn) const;

  // Return the number of elements of type `type` that can fit into the largest
  // vector register available.  We need to pass in "function" since llvm
  // functions can contain annotations for specializing them to specific
  // micro-architectures (though currently XLA does not use this functionality).
  virtual int32_t vector_register_num_elements(const llvm::Function& fn,
                                               PrimitiveType type) const;

  // Return the number of vector registers.  We need to pass in
  // "function" since llvm functions can contain annotations for specializing
  // them to specific micro-architectures (though currently XLA does not use
  // this functionality).
  virtual int32_t vector_register_count(const llvm::Function& fn) const;

  // Returns the minimum alignment for a buffer of size size_bytes.
  virtual int64_t minimum_alignment_for_allocation(int64_t size_bytes) const;

  virtual std::string get_target_feature_string() const;

 private:
  llvm::TargetTransformInfo* GetTargetTransformInfoFor(
      const llvm::Function& fn) const;

  // A cache of resolved TargetTransformInfo for LLVM functions.
  mutable absl::flat_hash_map<const llvm::Function*, llvm::TargetTransformInfo>
      target_transform_info_;

  llvm::TargetMachine* target_machine_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_CODEGEN_TARGET_MACHINE_FEATURES_H_
