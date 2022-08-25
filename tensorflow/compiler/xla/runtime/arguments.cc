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

#include "tensorflow/compiler/xla/runtime/arguments.h"

#include <cstddef>
#include <string>
#include <type_traits>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Casting.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/runtime/errors.h"
#include "tensorflow/compiler/xla/runtime/types.h"

namespace xla {
namespace runtime {

using llvm::dyn_cast;
using llvm::isa;

using llvm::ArrayRef;
using llvm::Error;
using llvm::MutableArrayRef;
using llvm::Optional;
using llvm::raw_ostream;

static raw_ostream& operator<<(raw_ostream& os, PrimitiveType& type) {
  return os << primitive_util::LowercasePrimitiveTypeName(type);
}

raw_ostream& OpaqueArg::print(raw_ostream& os) const {
  return os << "OpaqueArg: ptr=" << ptr_;
}

raw_ostream& MemrefDesc::print(raw_ostream& os) const {
  auto print_arr = [&](llvm::StringRef name, ArrayRef<int64_t> arr) {
    os << " " << name << ": [";
    if (!arr.empty()) {
      os << arr[0];
      for (int i = 1; i < arr.size(); ++i) os << ", " << arr[i];
    }
    os << "]";
  };

  os << "MemrefDesc: dtype: " << dtype() << " offset: " << offset();
  print_arr("sizes", sizes());
  print_arr("strides", strides());

  return os;
}

//===----------------------------------------------------------------------===//
// OpaqueArg.
//===----------------------------------------------------------------------===//

Error OpaqueArg::Verify(const Type& type) const {
  if (isa<AsyncTokenType>(type)) return Error::success();
  return MakeStringError("unsupported opaque argument type: ", type);
}

size_t OpaqueArg::Pack(MutableArrayRef<void*> args, size_t offset) const {
  args[offset] = ptr_;
  return ++offset;
}

//===----------------------------------------------------------------------===//
// MemrefDesc.
//===----------------------------------------------------------------------===//

static bool AreCompatibleTypes(PrimitiveType type1, PrimitiveType type2) {
  auto compatible = [&](PrimitiveType fromType, PrimitiveType toType) {
    return (type1 == fromType && type2 == toType) ||
           (type1 == toType && type2 == fromType);
  };
  // I1 and I8 types are compatible since they both are 1-byte size at runtime.
  if (compatible(PrimitiveType::PRED, PrimitiveType::S8)) return true;

  // Signed and unsigned integers of the same size are compatible in memory.
  if (compatible(PrimitiveType::S8, PrimitiveType::U8) ||
      compatible(PrimitiveType::S16, PrimitiveType::U16) ||
      compatible(PrimitiveType::S32, PrimitiveType::U32) ||
      compatible(PrimitiveType::S64, PrimitiveType::U64))
    return true;

  return type1 == type2;
}

static Error VerifyMemrefArgument(PrimitiveType element_type,
                                  Optional<absl::Span<const int64_t>> sizes,
                                  const MemrefDesc& memref) {
  // Format memref argument and expected type for user-friendly error messages.
  auto pretty_print = [&]() -> std::string {
    std::string err;
    llvm::raw_string_ostream os(err);

    auto dim = [](int64_t d) -> std::string {
      return d == MemrefType::kDynamicSize ? "?" : std::to_string(d);
    };

    auto print_shaped = [&](Optional<absl::Span<const int64_t>> dims,
                            PrimitiveType dtype) {
      if (!dims.has_value()) {
        os << "[*x" << dtype << "]";
        return;
      }

      if (dims->empty()) {
        os << "[" << dtype << "]";
        return;
      }

      os << "[" << dim((*dims)[0]);
      for (int i = 1; i < dims->size(); ++i) os << "x" << dim((*dims)[i]);
      os << "x" << dtype << "]";
    };

    os << "got ";
    print_shaped({memref.sizes()}, memref.dtype());
    os << " vs expected ";
    print_shaped(sizes, element_type);

    return err;
  };

  // Check that memref data type is compatible with the expected element type.
  if (LLVM_UNLIKELY(!AreCompatibleTypes(element_type, memref.dtype()))) {
    return MakeStringError(
        "type is not compatible with the expected element type: ",
        primitive_util::LowercasePrimitiveTypeName(memref.dtype()), " vs ",
        primitive_util::LowercasePrimitiveTypeName(element_type), " (",
        pretty_print(), ")");
  }

  // Skip sizes verification if they are not available (unranked tensor or
  // memref type is compatible with run-time arguments of any shape).
  if (!sizes.has_value()) return Error::success();

  // Check that memref rank is the same as the expected rank.
  if (LLVM_UNLIKELY(memref.rank() != sizes->size()))
    return MakeStringError(
        "rank does not match expected input rank: ", memref.rank(), " vs ",
        sizes->size(), " (", pretty_print(), ")");

  // Check that all statically known dimensions matches the memref dimensions.
  for (const auto& pair : llvm::enumerate(llvm::zip(memref.sizes(), *sizes))) {
    int64_t argument_dim = std::get<0>(pair.value());
    int64_t expected_dim = std::get<1>(pair.value());

    bool is_dynamic_dim = MemrefType::IsDynamic(expected_dim);

    if (LLVM_UNLIKELY(argument_dim != expected_dim && !is_dynamic_dim))
      return MakeStringError(
          "dimension #", pair.index(),
          " does not match expected input dimension: ", argument_dim, " vs ",
          expected_dim, " (", pretty_print(), ")");
  }

  return Error::success();
}

Error MemrefDesc::Verify(const Type& type) const {
  // Only ranked memrefs have a defined ABI and can be passed as an argument.
  if (auto* memref = dyn_cast<MemrefType>(&type))
    return VerifyMemrefArgument(memref->element_type(), memref->sizes(), *this);
  return MakeStringError("unsupported memref type: ", type);
}

size_t MemrefDesc::Pack(MutableArrayRef<void*> args, size_t offset) const {
  // Write into the arguments data starting from the given offset.
  void** storage = &args[offset];

  auto cast = [](const void* p) { return const_cast<void*>(p); };

  // Packs memref with a rank not known at compile time.
  auto pack_memref = [&](int64_t rank) -> size_t {
    storage[0] = cast(&data_);  // memref.basePtr
    storage[1] = cast(&data_);  // memref.data
    storage[2] = cast(&offset_);
    for (int64_t d = 0; d < rank; ++d) {
      storage[3 + d] = cast(&sizes_and_strides_[d]);
      storage[3 + rank + d] = cast(&sizes_and_strides_[rank_ + d]);
    }

    // Move offsets to the next argument position.
    return offset + 3 + rank * 2;
  };

  // Packs memref with a rank known at compile time.
  auto pack_ranked_memref = [&](auto rank_tag) -> size_t {
    static constexpr int64_t rank = decltype(rank_tag)::value;
    return pack_memref(rank);
  };

  // Dispatch to lambda with a statically known rank parameter for the most
  // common ranks. It allows to inline the nested lambda, and generate better
  // code without for loops on a hot path.
  switch (rank_) {
    case 0:
      return pack_ranked_memref(std::integral_constant<int64_t, 0>{});
    case 1:
      return pack_ranked_memref(std::integral_constant<int64_t, 1>{});
    case 2:
      return pack_ranked_memref(std::integral_constant<int64_t, 2>{});
    case 3:
      return pack_ranked_memref(std::integral_constant<int64_t, 3>{});
    case 4:
      return pack_ranked_memref(std::integral_constant<int64_t, 4>{});
    default:
      return pack_memref(rank_);
  }
}

//===----------------------------------------------------------------------===//
// Verify that argument type is compatible with the run-time memref argument.
//===----------------------------------------------------------------------===//

static Error VerifyMemrefArgument(const Type& type, const MemrefDesc& arg) {
  if (auto* memref = dyn_cast<MemrefType>(&type))
    return VerifyMemrefArgument(memref->element_type(), memref->sizes(), arg);
  if (auto* memref = dyn_cast<UnrankedMemrefType>(&type))
    return VerifyMemrefArgument(memref->element_type(), llvm::None, arg);

  if (auto* tensor = dyn_cast<RankedTensorType>(&type))
    return VerifyMemrefArgument(tensor->element_type(), tensor->sizes(), arg);
  if (auto* tensor = dyn_cast<UnrankedTensorType>(&type))
    return VerifyMemrefArgument(tensor->element_type(), llvm::None, arg);

  return MakeStringError("unsupported memref type: ", type);
}

Error VerifyMemrefArgument(unsigned index, const Type& type,
                           const MemrefDesc& arg) {
  if (auto err = VerifyMemrefArgument(type, arg))
    return MakeStringError("argument #", index, " ", err);
  return Error::success();
}

// -------------------------------------------------------------------------- //
// BufferDesc.
// -------------------------------------------------------------------------- //

raw_ostream& BufferDesc::print(raw_ostream& os) const {
  return os << "BufferDesc: data: " << data() << " size: " << size();
}

static Error VerifyBufferDesc(PrimitiveType element_type,
                              Optional<absl::Span<const int64_t>> sizes,
                              const BufferDesc& buffer) {
  size_t n_elem = !sizes.hasValue() || sizes->empty() ? 1 : (*sizes)[0];
  size_t expected_buffer_size =
      primitive_util::ByteWidth(element_type) * n_elem;
  if (LLVM_UNLIKELY(expected_buffer_size != buffer.size())) {
    return MakeStringError(
        "buffer size is not equal to that expected from the element type: got ",
        buffer.size(), " vs expected ", expected_buffer_size, ".");
  }
  return Error::success();
}

Error BufferDesc::Verify(const Type& type) const {
  // BufferDesc doesn't have its own type signature; it works with MemrefType.
  if (auto* memref = dyn_cast<MemrefType>(&type))
    return VerifyBufferDesc(memref->element_type(), memref->sizes(), *this);
  return MakeStringError("unsupported memref type: ", type);
}

size_t BufferDesc::Pack(MutableArrayRef<void*> args, size_t offset) const {
  auto cast = [](const void* ptr) { return const_cast<void*>(ptr); };
  // Write into the arguments data starting from the given offset.
  void** p = &args[offset];
  p[0] = cast(&data_);
  p[1] = cast(&data_);
  p[2] = cast(&size_);
  return offset + 3;
}

}  // namespace runtime
}  // namespace xla
