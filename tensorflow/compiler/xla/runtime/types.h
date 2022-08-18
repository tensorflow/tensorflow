/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_RUNTIME_TYPES_H_
#define XLA_RUNTIME_TYPES_H_

#include <functional>
#include <memory>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "tfrt/dtype/dtype.h"  // from @tf_runtime

namespace xla {
namespace runtime {

//===----------------------------------------------------------------------===//
// Canonical XLA runtime types for the executable arguments.
//===----------------------------------------------------------------------===//

// Types supported by the compiled function signature. We do rely on the LLVM
// style RTTI (https://llvm.org/docs/HowToSetUpLLVMStyleRTTI.html) to avoid
// dependency on the MLIR types at runtime, because we don't want to depend
// on any of the compiler implementation details at runtime and we want to
// support lightweight loading and execution of AOT compiled programs.
//
// We rely on the RTTI for the open class hierarchies, because we want to allow
// users to define their own types for the arguments.
//
// If the type can be passed to the compiled function as an argument or returned
// as a result, it must define its own ABI. The ABI is defined by the MLIR to
// LLVM lowering pipeline and the runtime integration (see `runtime.h`).
class Type : public llvm::RTTIExtends<Type, llvm::RTTIRoot> {
 public:
  static constexpr char ID = 0;  // NOLINT

  // Arguments to compiled functions passed as a set of pointers. For example
  // memref descriptor passed in as a set of pointers to data, sizes and
  // strides. See `Argument::Pack` implementation for details (in `argument.h`).
  struct ArgumentAbi {
    size_t num_ptrs;
  };

  // Compiled function returns results by writing into the pre-allocated storage
  // of the given size with the requested alignment. Runtime pre-allocates
  // memory required for all results in the call frame.
  struct ResultAbi {
    size_t size;

    // TODO(ezhulenev): Add alignment to the result ABI. Alignment is an
    // important part of the result ABI that we ignore today. It all doesn't
    // crash only because all results happen to have a size that is multiple of
    // 8 bytes, and because of that all of the results are properly aligned.
    // Results memory layout in the call frame should take in account base
    // pointer alignment and alignment requirements of all results.
  };

  // Returns an Abi if the type can be used as an argument.
  virtual llvm::ErrorOr<ArgumentAbi> AsArgument() const {
    return llvm::errc::not_supported;
  }

  // Returns an Abi if the type can be returned as a result.
  virtual llvm::ErrorOr<ResultAbi> AsResult() const {
    return llvm::errc::not_supported;
  }

  virtual llvm::raw_ostream& print(llvm::raw_ostream& os) const = 0;

 protected:
  Type() = default;
};

inline llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const Type& type) {
  return type.print(os);
}

//===----------------------------------------------------------------------===//
// Async Token type corresponding to the mlir::async::TokenType
//===----------------------------------------------------------------------===//

class AsyncTokenType : public llvm::RTTIExtends<AsyncTokenType, Type> {
 public:
  static constexpr char ID = 0;  // NOLINT

  llvm::ErrorOr<ResultAbi> AsResult() const final;

  llvm::raw_ostream& print(llvm::raw_ostream& os) const final;
};

//===----------------------------------------------------------------------===//
// Async Value type corresponding to the mlir::async::ValueType.
//===----------------------------------------------------------------------===//

class AsyncValueType : public llvm::RTTIExtends<AsyncValueType, Type> {
 public:
  static constexpr char ID = 0;  // NOLINT

  explicit AsyncValueType(std::unique_ptr<Type> value_type)
      : value_type_(std::move(value_type)) {}

  const Type& value_type() const { return *value_type_; }

  llvm::ErrorOr<ResultAbi> AsResult() const final;

  llvm::raw_ostream& print(llvm::raw_ostream& os) const final;

 private:
  std::unique_ptr<Type> value_type_;
};

//===----------------------------------------------------------------------===//
// Ranked Tensor type corresponding to the mlir::RankedTensorType.
//===----------------------------------------------------------------------===//

class RankedTensorType : public llvm::RTTIExtends<RankedTensorType, Type> {
 public:
  static constexpr char ID = 0;  // NOLINT
  static constexpr int64_t kDynamicSize = -1;

  static constexpr bool IsDynamic(int64_t dim) { return dim == kDynamicSize; }

  RankedTensorType(llvm::ArrayRef<int64_t> sizes, tfrt::DType element_type)
      : sizes_(sizes.begin(), sizes.end()), element_type_(element_type) {}

  llvm::ArrayRef<int64_t> sizes() const { return sizes_; }
  unsigned rank() const { return sizes_.size(); }
  tfrt::DType element_type() const { return element_type_; }

  llvm::raw_ostream& print(llvm::raw_ostream& os) const final;

 private:
  llvm::SmallVector<int64_t> sizes_;
  tfrt::DType element_type_;
};

//===----------------------------------------------------------------------===//
// Unranked Tensor type corresponding to the mlir::UnrankedTensorType.
//===----------------------------------------------------------------------===//

class UnrankedTensorType : public llvm::RTTIExtends<UnrankedTensorType, Type> {
 public:
  static constexpr char ID = 0;  // NOLINT

  explicit UnrankedTensorType(tfrt::DType element_type)
      : element_type_(element_type) {}

  tfrt::DType element_type() const { return element_type_; }

  llvm::raw_ostream& print(llvm::raw_ostream& os) const final;

 private:
  tfrt::DType element_type_;
};

//===----------------------------------------------------------------------===//
// Ranked Memref type corresponding to the mlir::MemrefType.
//===----------------------------------------------------------------------===//

class MemrefType : public llvm::RTTIExtends<MemrefType, Type> {
 public:
  static constexpr char ID = 0;  // NOLINT
  static constexpr int64_t kDynamicSize = -1;

  static constexpr bool IsDynamic(int64_t dim) { return dim == kDynamicSize; }

  MemrefType(llvm::ArrayRef<int64_t> sizes, tfrt::DType element_type)
      : sizes_(sizes.begin(), sizes.end()), element_type_(element_type) {}

  llvm::ArrayRef<int64_t> sizes() const { return sizes_; }
  unsigned rank() const { return sizes_.size(); }
  tfrt::DType element_type() const { return element_type_; }

  llvm::ErrorOr<ArgumentAbi> AsArgument() const final;
  llvm::ErrorOr<ResultAbi> AsResult() const final;

  llvm::raw_ostream& print(llvm::raw_ostream& os) const final;

 private:
  llvm::SmallVector<int64_t> sizes_;
  tfrt::DType element_type_;
};

//===----------------------------------------------------------------------===//
// Unranked Memref type corresponding to the mlir::UnrankedMemrefType.
//===----------------------------------------------------------------------===//

class UnrankedMemrefType : public llvm::RTTIExtends<UnrankedMemrefType, Type> {
 public:
  static constexpr char ID = 0;  // NOLINT

  explicit UnrankedMemrefType(tfrt::DType element_type)
      : element_type_(element_type) {}

  tfrt::DType element_type() const { return element_type_; }

  llvm::raw_ostream& print(llvm::raw_ostream& os) const final;

 private:
  tfrt::DType element_type_;
};

//===----------------------------------------------------------------------===//
// Corresponds to the RT dialect's KernelContextType.
//===----------------------------------------------------------------------===//

class KernelContextOperandType
    : public llvm::RTTIExtends<KernelContextOperandType, Type> {
 public:
  static constexpr char ID = 0;  // NOLINT

  llvm::ErrorOr<ArgumentAbi> AsArgument() const final;

  llvm::raw_ostream& print(llvm::raw_ostream& os) const final;
};

//===----------------------------------------------------------------------===//
// Compiled function signature type corresponding to the mlir::FunctionType.
//===----------------------------------------------------------------------===//

class FunctionType {
 public:
  const Type* operand(unsigned index) const { return operands_[index].get(); }
  const Type* result(unsigned index) const { return results_[index].get(); }

  unsigned num_operands() const { return operands_.size(); }
  unsigned num_results() const { return results_.size(); }

  FunctionType(llvm::SmallVector<std::unique_ptr<Type>> operands,
               llvm::SmallVector<std::unique_ptr<Type>> results)
      : operands_(std::move(operands)), results_(std::move(results)) {}

 private:
  llvm::SmallVector<std::unique_ptr<Type>> operands_;
  llvm::SmallVector<std::unique_ptr<Type>> results_;
};

}  // namespace runtime
}  // namespace xla

#endif  // XLA_RUNTIME_TYPES_H_
