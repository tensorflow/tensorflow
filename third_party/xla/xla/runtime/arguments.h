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
#include <initializer_list>
#include <string>
#include <type_traits>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "xla/primitive_util.h"
#include "xla/runtime/async_runtime.h"
#include "xla/runtime/types.h"

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

  // Packs argument into the `args` view according to the expected executable
  // ABI.
  //
  // Arguments view is guaranteed to be properly sized to have space for all
  // arguments according to the arguments memory layout.
  virtual void Pack(absl::Span<void*> args) const = 0;

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
    for (size_t i = 0; i < storage_.size(); ++i) {
      reinterpret_cast<Argument*>(storage_[i].data)->~Argument();
    }
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

  const auto& operator[](size_t index) const {
    using T = std::conditional_t<sizeof...(Ts) == 1,
                                 std::tuple_element_t<0, std::tuple<Ts...>>,
                                 Argument>;
    return *reinterpret_cast<const T*>(storage_[index].data);
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
// `absl::Span<const Argument>`, however because `Span` of a virtual base is not
// possible, we have our own type that is constructible from the `Arguments`
// and array reference or vector of any argument subtype.
class ArgumentsRef {
  template <typename T>
  static constexpr bool is_argument = std::is_base_of_v<Argument, T>;

 public:
  ArgumentsRef() : data_(nullptr), size_(0), stride_(0) {}

  template <typename... Ts, std::enable_if_t<sizeof...(Ts) != 0>* = nullptr>
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

  template <typename T, std::enable_if_t<is_argument<T>>* = nullptr>
  ArgumentsRef(const std::initializer_list<T>& list)  // NOLINT
      : ArgumentsRef(llvm::ArrayRef<T>(list)) {}

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
// Canonical types for passing compiled executable arguments.
//===----------------------------------------------------------------------===//

// By default we provide a set of types for passing common arguments to the
// compiled executable. The type hierarchy is open, and users can extend it by
// definining new `Type` and `Argument` with the corresponding MLIR types and
// MLIR passes to lower types and operations to the LLVM dialect.

//===----------------------------------------------------------------------===//
// OpaqueArg for passing `!rt.opaque` arguments (lowered to `!llvm.ptr`).
//===----------------------------------------------------------------------===//

class OpaqueArg final : public llvm::RTTIExtends<OpaqueArg, Argument> {
 public:
  static constexpr char ID = 0;  // NOLINT

  explicit OpaqueArg(void* ptr) : ptr_(ptr) {}

  void* ptr() const { return ptr_; }

  absl::Status Verify(const Type& type) const final;
  void Pack(absl::Span<void*> args) const final;
  std::string ToString() const final;

 private:
  void* ptr_;
};

template <typename T>
using EnableIfScalarType = typename std::enable_if_t<
    std::disjunction_v<std::is_same<T, float>, std::is_same<T, int32_t>,
                       std::is_same<T, int64_t>>>;

//===----------------------------------------------------------------------===//
// ScalarArg for passing integer or float scalar arguments.
//===----------------------------------------------------------------------===//

class ScalarArg final : public llvm::RTTIExtends<ScalarArg, Argument> {
 public:
  static constexpr char ID = 0;  // NOLINT

  template <typename T, EnableIfScalarType<T>* = nullptr>
  explicit ScalarArg(T value)
      : type_(primitive_util::NativeToPrimitiveType<T>()), value_(value) {}

  absl::Status Verify(const Type& type) const final;
  void Pack(absl::Span<void*> args) const final;
  std::string ToString() const final;

 private:
  // We store value in a union instead of an `std::variant` so that we can pack
  // a pointer to this union as an executable argument.
  union Value {
    explicit Value(int32_t i32) : i32(i32) {}
    explicit Value(int64_t i64) : i64(i64) {}
    explicit Value(float f32) : f32(f32) {}
    int32_t i32;
    int64_t i64;
    float f32;
  };

  PrimitiveType type_;
  Value value_;
};

//===----------------------------------------------------------------------===//
// MemrefDesc for passing `memref` arguments.
//===----------------------------------------------------------------------===//

class MemrefDesc final : public llvm::RTTIExtends<MemrefDesc, Argument> {
 public:
  static constexpr char ID = 0;  // NOLINT

  MemrefDesc(PrimitiveType dtype, void* data, int64_t offset,
             absl::Span<const int64_t> sizes, absl::Span<const int64_t> strides)
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
  //   void operator()(absl::Span<int64_t> sizes,
  //                   absl::Span<int64_t> strides);
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

  absl::Span<const int64_t> sizes() const {
    return {sizes_and_strides_.data(), rank_};
  }

  absl::Span<const int64_t> strides() const {
    return {sizes_and_strides_.data() + rank_, rank_};
  }

  absl::Status Verify(const Type& type) const final;
  void Pack(absl::Span<void*> args) const final;
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
  auto ref = absl::Span<int64_t>(sizes_and_strides_);
  initialize(ref.subspan(0, rank_), ref.subspan(rank_));
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
// AsyncTokenArg for passing async token arguments
//===----------------------------------------------------------------------===//

class AsyncTokenArg final : public llvm::RTTIExtends<AsyncTokenArg, Argument> {
 public:
  static constexpr char ID = 0;  // NOLINT

  explicit AsyncTokenArg(tsl::AsyncValueRef<tsl::Chain> value)
      : storage_(AsyncRuntime::AsToken(value)) {}

  absl::Status Verify(const Type& type) const final;
  void Pack(absl::Span<void*> args) const final;
  std::string ToString() const final;

 private:
  // In the runtime execution, we unpack args with pointer to pointer
  // dereferening. We declare storage_ as a member variable (instead of a local
  // inside the Pack function) to keep its address valid when unpacking.
  AsyncRuntime::Token* storage_;
};

//===----------------------------------------------------------------------===//
// AsyncScalarArg for passing async scalar arguments
//===----------------------------------------------------------------------===//

class AsyncScalarArg final
    : public llvm::RTTIExtends<AsyncScalarArg, Argument> {
 public:
  static constexpr char ID = 0;  // NOLINT

  template <typename T, EnableIfScalarType<T>* = nullptr>
  explicit AsyncScalarArg(tsl::AsyncValueRef<T> value)
      : type_(primitive_util::NativeToPrimitiveType<T>()) {
    auto write = [](const T* v, std::byte* store) {
      T* store_t = reinterpret_cast<T*>(store);
      *store_t = *v;
    };

    storage_ = AsyncRuntime::AsValue<T>(value, sizeof(T),
                                        alignof(std::max_align_t), write);
  }

  absl::Status Verify(const Type& type) const final;
  void Pack(absl::Span<void*> args) const final;

  std::string ToString() const final;

 private:
  PrimitiveType type_;
  AsyncRuntime::Value* storage_;
};

//===----------------------------------------------------------------------===//
// AsyncMemrefArg for passing async memref arguments
//===----------------------------------------------------------------------===//
class AsyncMemrefArg final
    : public llvm::RTTIExtends<AsyncMemrefArg, Argument> {
 public:
  static constexpr char ID = 0;  // NOLINT

  explicit AsyncMemrefArg(tsl::AsyncValueRef<MemrefDesc> value);

  absl::Status Verify(const Type& type) const final;
  void Pack(absl::Span<void*> args) const final;
  std::string ToString() const final;

 private:
  tsl::AsyncValueRef<MemrefDesc> value_;
  AsyncRuntime::Value* storage_;
};
}  // namespace runtime
}  // namespace xla

#endif  // XLA_RUNTIME_ARGUMENTS_H_
