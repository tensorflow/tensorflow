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

#ifndef XLA_RUNTIME_ARGUMENTS_H_
#define XLA_RUNTIME_ARGUMENTS_H_

#include <cstddef>
#include <string>
#include <type_traits>

#include "absl/status/status.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "tensorflow/compiler/xla/runtime/types.h"

namespace xla {
namespace runtime {

//===----------------------------------------------------------------------===//
// A base class for XLA executable arguments.
//===----------------------------------------------------------------------===//

class Argument : public llvm::RTTIExtends<Type, llvm::RTTIRoot> {
 public:
  static constexpr char ID = 0;  // NOLINT

  Argument() = default;

  // Verifies that the argument matches the expected type.
  virtual absl::Status Verify(const Type& type) const = 0;

  // Packs argument into the `args` array starting at the given `offset`
  // according to the expected executable ABI. Return offset incremented by
  // the number of packed pointers, so that result will point to the offset for
  // packing the next argument.
  //
  // Arguments array is guaranteed to be properly sized to have space for all
  // arguments according to the arguments memory layout.
  virtual size_t Pack(llvm::MutableArrayRef<void*> args,
                      size_t offset) const = 0;

  virtual std::string ToString() const = 0;
};

inline llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                                     const Argument& arg) {
  return os << arg.ToString();
}

//===----------------------------------------------------------------------===//
// Owning container for storing arguments of different types.
//===----------------------------------------------------------------------===//

// Forward declare class defined below.
class ArgumentsRef;

// An owning container for the variadic arguments, optimized for storing all
// arguments of the declared types without dynamic memory allocations.
//
// Example:
//
//   Arguments<OpaqueArg, MemrefDesc> arguments;
//   arguments.emplace_back<OpaqueArg>(...);
//
// Variadic type parameter `Ts` specifies arguments of what types can be added
// to the container.
template <typename... Ts>
class Arguments {
 public:
  explicit Arguments(size_t num_args) : num_args_(num_args) {
    storage_.reserve(num_args);
  }

  ~Arguments() {
    for (size_t i = 0; i < storage_.size(); ++i) (*this)[i].~Argument();
  }

  template <typename T>
  T& push_back(T value) {
    static_assert(std::disjunction_v<std::is_same<T, Ts>...>,
                  "type is not supported by this instance of arguments");
    assert(storage_.size() < num_args_ && "arguments overflow");
    storage_.resize_for_overwrite(storage_.size() + 1);
    return *(new (&storage_.back()) T(std::forward<T>(value)));
  }

  template <typename T = std::tuple_element_t<0, std::tuple<Ts...>>,
            typename... Args>
  T& emplace_back(Args... args) {
    static_assert(std::disjunction_v<std::is_same<T, Ts>...>,
                  "type is not supported by this instance of arguments");
    assert(storage_.size() < num_args_ && "arguments overflow");
    storage_.resize_for_overwrite(storage_.size() + 1);
    return *(new (&storage_.back()) T(std::forward<Args>(args)...));
  }

  const Argument& operator[](size_t index) const {
    return *reinterpret_cast<const Argument*>(storage_[index].data);
  }

  size_t size() const { return storage_.size(); }

 private:
  friend class ArgumentsRef;

  static_assert(std::conjunction_v<std::is_base_of<Argument, Ts>...>,
                "all types must be arguments");

  // Arguments are not movable or copyable because we do manual memory
  // management using the `Storage` struct, and moving or copying bytes storing
  // the argument value is undefined behavior.
  Arguments(const Arguments&) = delete;
  Arguments& operator=(const Arguments&) = delete;
  Arguments(Arguments&&) = delete;
  Arguments& operator=(Arguments&&) = delete;

  // Avoid dynamic memory allocation for storing arguments of different types
  // by storing them in the properly aligned byte array.
  struct Storage {
    alignas(Ts...) std::byte data[std::max({sizeof(Ts)...})];
  };

  // To guarantee safe conversion between pointer to `Storage` and pointer to
  // the first byte (Argument), the storage struct must have standard layout.
  static_assert(std::is_standard_layout_v<Storage>,
                "storage must have standard layout");

  size_t num_args_;
  llvm::SmallVector<Storage> storage_;
};

// A constant reference to an array of arguments, somewhat similar to the
// `ArrayRef<Argument>`, however because `ArrayRef` of a virtual base is not
// possible, we have our own type that is constructible from the `Arguments`
// and array reference or vector of any argument subtype.
class ArgumentsRef {
  template <typename T>
  static constexpr bool is_argument = std::is_base_of_v<Argument, T>;

 public:
  template <typename... Ts>
  ArgumentsRef(const Arguments<Ts...>& args)  // NOLINT
      : data_(reinterpret_cast<const Argument*>(args.storage_.data())),
        size_(args.size()),
        stride_(sizeof(typename Arguments<Ts...>::Storage)) {}

  template <typename T, std::enable_if_t<is_argument<T>>* = nullptr>
  ArgumentsRef(llvm::ArrayRef<T> ref)  // NOLINT
      : data_(ref.data()), size_(ref.size()), stride_(sizeof(T)) {}

  template <typename T, std::enable_if_t<is_argument<T>>* = nullptr>
  ArgumentsRef(const llvm::SmallVectorImpl<T>& vec)  // NOLINT
      : ArgumentsRef(llvm::ArrayRef<T>(vec)) {}

  template <typename T, std::enable_if_t<is_argument<T>>* = nullptr>
  ArgumentsRef(const std::vector<T>& vec)  // NOLINT
      : ArgumentsRef(llvm::ArrayRef<T>(vec)) {}

  template <typename T, size_t n, std::enable_if_t<is_argument<T>>* = nullptr>
  ArgumentsRef(const std::array<T, n>& arr)  // NOLINT
      : ArgumentsRef(llvm::ArrayRef<T>(arr)) {}

  const Argument& operator[](size_t index) const {
    assert(index < size_ && "index out of bounds");
    auto* ptr = reinterpret_cast<const std::byte*>(data_) + index * stride_;
    return *reinterpret_cast<const Argument*>(ptr);
  }

  size_t size() const { return size_; }

 private:
  // Arguments stored in the contiguous memory starting at `data_` pointer,
  // with the given `stride_` in bytes.
  const Argument* data_;
  size_t size_;
  size_t stride_;
};

//===----------------------------------------------------------------------===//
// Canonical types for passing compiled kernel arguments.
//===----------------------------------------------------------------------===//

// By default we provide a set of types for passing common arguments to the
// compiled kernel. The type hierarchy is open, and users can extend it by
// definining new `Type` and `Argument` with the corresponding MLIR types and
// MLIR passes to lower types and operations to the LLVM dialect.

//===----------------------------------------------------------------------===//
// OpaqueArg for passing `!llvm.ptr` (opaque pointer) arguments.
//===----------------------------------------------------------------------===//

class OpaqueArg final : public llvm::RTTIExtends<OpaqueArg, Argument> {
 public:
  static constexpr char ID = 0;  // NOLINT

  explicit OpaqueArg(void* ptr) : ptr_(ptr) {}

  void* ptr() const { return ptr_; }

  absl::Status Verify(const Type& type) const final;
  size_t Pack(llvm::MutableArrayRef<void*> args, size_t offset) const final;
  std::string ToString() const final;

 private:
  void* ptr_;
};

//===----------------------------------------------------------------------===//
// MemrefDesc for passing `memref` arguments.
//===----------------------------------------------------------------------===//

class MemrefDesc final : public llvm::RTTIExtends<MemrefDesc, Argument> {
 public:
  static constexpr char ID = 0;  // NOLINT

  MemrefDesc(PrimitiveType dtype, void* data, int64_t offset,
             llvm::ArrayRef<int64_t> sizes, llvm::ArrayRef<int64_t> strides)
      : rank_(sizes.size()), dtype_(dtype), data_(data), offset_(offset) {
    assert(sizes.size() == strides.size() && "invalid sizes and strides pair");
    sizes_and_strides_.reserve(2 * rank_);
    sizes_and_strides_.append(sizes.begin(), sizes.end());
    sizes_and_strides_.append(strides.begin(), strides.end());
  }

  // Constructs MemrefDesc of the given rank and calls user-provided callback to
  // initialize sizes and strides.
  //
  // Expected `InitializeSizesAndStrides` callback signature:
  //
  //   void operator()(MutableArrayRef<int64_t> sizes,
  //                   MutableArrayRef<int64_t> strides);
  //
  // We pass the init callback as a template argument to be able to
  // inline it at the call site, because MemrefDesc construction is on a hot
  // path.
  template <typename InitializeSizesAndStrides>
  MemrefDesc(unsigned rank, PrimitiveType dtype, void* data, int64_t offset,
             InitializeSizesAndStrides initialize);

  // Ensure that MemrefDesc is always moved around instead of copying.
  MemrefDesc(const MemrefDesc&) = delete;
  MemrefDesc& operator=(const MemrefDesc&) = delete;
  MemrefDesc(MemrefDesc&&) = default;
  MemrefDesc& operator=(MemrefDesc&&) = default;

  unsigned rank() const { return rank_; }
  PrimitiveType dtype() const { return dtype_; }

  void* data() const { return data_; }
  int64_t offset() const { return offset_; }

  int64_t size(size_t index) const { return sizes_and_strides_[index]; }
  int64_t stride(size_t index) const {
    return sizes_and_strides_[rank_ + index];
  }

  llvm::ArrayRef<int64_t> sizes() const {
    return {sizes_and_strides_.data(), rank_};
  }
  llvm::ArrayRef<int64_t> strides() const {
    return {sizes_and_strides_.data() + rank_, rank_};
  }

  absl::Status Verify(const Type& type) const final;
  size_t Pack(llvm::MutableArrayRef<void*> args, size_t offset) const final;
  std::string ToString() const final;

 private:
  unsigned rank_;
  PrimitiveType dtype_;
  void* data_;
  int64_t offset_;
  // We keep sizes and strides in a single container to save one potential
  // memory allocation for memrefs of higher ranks, and to save one vector
  // constructor/destructor call.
  llvm::SmallVector<int64_t, 8> sizes_and_strides_;
};

template <typename InitializeSizesAndStrides>
MemrefDesc::MemrefDesc(unsigned rank, PrimitiveType dtype, void* data,
                       int64_t offset, InitializeSizesAndStrides initialize)
    : rank_(rank), dtype_(dtype), data_(data), offset_(offset) {
  sizes_and_strides_.resize(2 * rank_);
  llvm::MutableArrayRef<int64_t> ref = sizes_and_strides_;
  initialize(ref.drop_back(rank_), ref.drop_front(rank_));
}

//===----------------------------------------------------------------------===//
// Verify that argument type is compatible with the run-time memref argument.
//===----------------------------------------------------------------------===//

// Verifies that the type at the given `index` matches the run-time memref
// argument: type is a tensor of a memref with compatible element type, and all
// statically known dimensions match the run-time sizes. Returns user-friendly
// error message in case of an error.
absl::Status VerifyMemrefArgument(unsigned index, const Type& type,
                                  const MemrefDesc& arg);

//===----------------------------------------------------------------------===//
// BufferDesc for passing raw `buffer` (i.e. void ptr + size) arguments.
//===----------------------------------------------------------------------===//

class BufferDesc final : public llvm::RTTIExtends<BufferDesc, Argument> {
 public:
  static constexpr char ID = 0;  // NOLINT

  BufferDesc(void* data, size_t size) : data_(data), size_(size) {}

  void* data() const { return data_; }
  size_t size() const { return size_; }

  absl::Status Verify(const Type& type) const final;
  size_t Pack(llvm::MutableArrayRef<void*> args, size_t offset) const final;
  std::string ToString() const final;

 private:
  void* data_;
  size_t size_;
};

}  // namespace runtime
}  // namespace xla

#endif  // XLA_RUNTIME_ARGUMENTS_H_
