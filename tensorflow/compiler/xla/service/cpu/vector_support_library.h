/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_VECTOR_SUPPORT_LIBRARY_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_VECTOR_SUPPORT_LIBRARY_H_

#include <string>

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace cpu {
// A thin wrapper around llvm_util.h to make code generating vector math flow
// more readable.
class VectorSupportLibrary {
 public:
  // This VectorSupportLibrary instance remembers `primitive_type` and
  // `vector_size`, and these are implicitly used by the methods on this
  // instance (i.e. LoadVector will load a vector of type <`vector_size` x
  // `primitive_type`>).
  VectorSupportLibrary(PrimitiveType primitive_type, int64 vector_size,
                       llvm::IRBuilder<>* ir_builder, std::string name);

  llvm::Value* Mul(llvm::Value* lhs, llvm::Value* rhs);
  llvm::Value* Mul(int64 lhs, llvm::Value* rhs) {
    return Mul(ir_builder()->getInt64(lhs), rhs);
  }

  llvm::Value* Add(llvm::Value* lhs, llvm::Value* rhs);
  llvm::Value* Add(int64 lhs, llvm::Value* rhs) {
    return Add(ir_builder()->getInt64(lhs), rhs);
  }

  llvm::Value* MulAdd(llvm::Value* a, llvm::Value* b, llvm::Value* c) {
    return Add(c, Mul(a, b));
  }

  llvm::Value* ComputeOffsetPointer(llvm::Value* base_pointer,
                                    llvm::Value* offset_elements);
  llvm::Value* ComputeOffsetPointer(llvm::Value* base_pointer,
                                    int64 offset_elements) {
    return ComputeOffsetPointer(base_pointer,
                                ir_builder()->getInt64(offset_elements));
  }

  llvm::Value* LoadVector(llvm::Value* pointer);

  llvm::Value* LoadVector(llvm::Value* base_pointer,
                          llvm::Value* offset_elements) {
    return LoadVector(ComputeOffsetPointer(base_pointer, offset_elements));
  }

  llvm::Value* LoadVector(llvm::Value* base_pointer, int64 offset_elements) {
    return LoadVector(base_pointer, ir_builder()->getInt64(offset_elements));
  }

  llvm::Value* LoadScalar(llvm::Value* pointer);

  llvm::Value* LoadScalar(llvm::Value* base_pointer,
                          llvm::Value* offset_elements) {
    return LoadScalar(ComputeOffsetPointer(base_pointer, offset_elements));
  }

  llvm::Value* LoadScalar(llvm::Value* base_pointer, int64 offset_elements) {
    return LoadScalar(base_pointer, ir_builder()->getInt64(offset_elements));
  }

  void StoreVector(llvm::Value* value, llvm::Value* pointer);

  void StoreVector(llvm::Value* value, llvm::Value* base_pointer,
                   llvm::Value* offset_elements) {
    StoreVector(value, ComputeOffsetPointer(base_pointer, offset_elements));
  }

  void StoreVector(llvm::Value* value, llvm::Value* base_pointer,
                   int64 offset_elements) {
    StoreVector(value, base_pointer, ir_builder()->getInt64(offset_elements));
  }

  void StoreScalar(llvm::Value* value, llvm::Value* pointer);
  void StoreScalar(llvm::Value* value, llvm::Value* base_pointer,
                   llvm::Value* offset_elements) {
    StoreScalar(value, ComputeOffsetPointer(base_pointer, offset_elements));
  }

  void StoreScalar(llvm::Value* value, llvm::Value* base_pointer,
                   int64 offset_elements) {
    StoreScalar(base_pointer, ir_builder()->getInt64(offset_elements));
  }

  llvm::Value* LoadBroadcast(llvm::Value* pointer);
  llvm::Value* LoadBroadcast(llvm::Value* base_pointer,
                             llvm::Value* offset_elements) {
    return LoadBroadcast(ComputeOffsetPointer(base_pointer, offset_elements));
  }
  llvm::Value* LoadBroadcast(llvm::Value* base_pointer, int64 offset_elements) {
    return LoadBroadcast(base_pointer, ir_builder()->getInt64(offset_elements));
  }

  // Compute the horizontal sum of each vector in `vectors`.  The i'th element
  // in the result vector is the (scalar) horizontal sum of the i'th vector in
  // `vectors`.  If `init_values` is not nullptr then the value in the i'th lane
  // in `init_values` is added to the i'th horizontal sum.
  std::vector<llvm::Value*> ComputeHorizontalSums(
      std::vector<llvm::Value*> vectors, llvm::Value* init_values = nullptr);

  llvm::Value* GetZeroVector();
  llvm::Value* GetZeroScalar();

  llvm::IRBuilder<>* ir_builder() const { return ir_builder_; }
  int64 vector_size() const { return vector_size_; }
  llvm::Type* vector_type() const { return vector_type_; }
  llvm::Type* vector_pointer_type() const { return vector_pointer_type_; }
  llvm::Type* scalar_type() const { return scalar_type_; }
  llvm::Type* scalar_pointer_type() const { return scalar_pointer_type_; }
  int64 scalar_byte_size() const {
    return primitive_util::BitWidth(primitive_type_) / 8;
  }

  const std::string& name() const { return name_; }

 private:
  llvm::Value* ExtractLowHalf(llvm::Value*);
  llvm::Value* ExtractHighHalf(llvm::Value*);

  llvm::Value* MulInternal(llvm::Value* lhs, llvm::Value* rhs);
  llvm::Value* AddInternal(llvm::Value* lhs, llvm::Value* rhs);

  llvm::Value* AddReduce(llvm::Value* vector);

  // Perform an X86 AVX style horizontal add between `lhs` and `rhs`.  The
  // resulting IR for an 8-float wide vector is expected to lower to a single
  // vhaddps instruction on a CPU that supports vhaddps, and not be too bad in
  // other cases.
  //
  // For a vector width of 8, the result vector is computed as:
  //   Result[0] = Lhs[0] + Lhs[1]
  //   Result[1] = Lhs[2] + Lhs[3]
  //   Result[2] = Rhs[0] + Rhs[1]
  //   Result[3] = Rhs[2] + Rhs[3]
  //   Result[4] = Lhs[4] + Lhs[5]
  //   Result[5] = Lhs[6] + Lhs[7]
  //   Result[6] = Rhs[4] + Rhs[5]
  //   Result[7] = Rhs[6] + Rhs[7]
  llvm::Value* AvxStyleHorizontalAdd(llvm::Value* lhs, llvm::Value* rhs);

  std::vector<llvm::Value*> ComputeAvxOptimizedHorizontalSums(
      std::vector<llvm::Value*> vectors, llvm::Value* init_values);

  int64 vector_size_;
  PrimitiveType primitive_type_;
  llvm::IRBuilder<>* ir_builder_;
  llvm::Type* vector_type_;
  llvm::Type* vector_pointer_type_;
  llvm::Type* scalar_type_;
  llvm::Type* scalar_pointer_type_;
  std::string name_;
};

// This wraps an alloca-backed stack variable which LLVM's SSA construction pass
// can later convert to a SSA value.
class LlvmVariable {
 public:
  LlvmVariable(llvm::Type*, llvm::IRBuilder<>* ir_builder);

  llvm::Value* Get() const;
  void Set(llvm::Value* new_value);

 private:
  llvm::AllocaInst* alloca_;
  llvm::IRBuilder<>* ir_builder_;
};

class VectorVariable : public LlvmVariable {
 public:
  VectorVariable(VectorSupportLibrary* vector_support,
                 llvm::Value* initial_value)
      : LlvmVariable(vector_support->vector_type(),
                     vector_support->ir_builder()) {
    Set(initial_value);
  }
};

class ScalarVariable : public LlvmVariable {
 public:
  ScalarVariable(VectorSupportLibrary* vector_support,
                 llvm::Value* initial_value)
      : LlvmVariable(vector_support->scalar_type(),
                     vector_support->ir_builder()) {
    Set(initial_value);
  }
};
}  // namespace cpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_VECTOR_SUPPORT_LIBRARY_H_
