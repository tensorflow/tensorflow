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
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/runtime/types.h"

namespace xla {
namespace runtime {

using absl::InvalidArgumentError;
using absl::Status;
using absl::StrCat;
using absl::StrFormat;
using absl::StrJoin;

using llvm::dyn_cast;
using llvm::isa;

using xla::primitive_util::LowercasePrimitiveTypeName;

//===----------------------------------------------------------------------===//
// OpaqueArg.
//===----------------------------------------------------------------------===//

Status OpaqueArg::Verify(const Type& type) const {
  if (isa<AsyncTokenType>(type)) return absl::OkStatus();
  return InvalidArgumentError(
      StrCat("unsupported opaque argument type: ", type.ToString()));
}

size_t OpaqueArg::Pack(absl::Span<void*> args, size_t offset) const {
  args[offset] = ptr_;
  return ++offset;
}

std::string OpaqueArg::ToString() const {
  return StrFormat("OpaqueArg: ptr=%p", ptr_);
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

static Status VerifyMemrefArgument(
    PrimitiveType element_type, std::optional<absl::Span<const int64_t>> sizes,
    const MemrefDesc& memref) {
  // Format memref argument and expected type for user-friendly error messages.
  auto pretty_print = [&]() -> std::string {
    std::string err;
    llvm::raw_string_ostream os(err);

    auto dim = [](int64_t d) -> std::string {
      return d == MemrefType::kDynamicSize ? "?" : std::to_string(d);
    };

    auto print_shaped = [&](std::optional<absl::Span<const int64_t>> dims,
                            PrimitiveType dtype) {
      if (!dims.has_value()) {
        os << "[*x" << LowercasePrimitiveTypeName(dtype) << "]";
        return;
      }

      if (dims->empty()) {
        os << "[" << LowercasePrimitiveTypeName(dtype) << "]";
        return;
      }

      os << "[" << dim((*dims)[0]);
      for (int i = 1; i < dims->size(); ++i) os << "x" << dim((*dims)[i]);
      os << "x" << LowercasePrimitiveTypeName(dtype) << "]";
    };

    os << "got ";
    print_shaped({memref.sizes()}, memref.dtype());
    os << " vs expected ";
    print_shaped(sizes, element_type);

    return err;
  };

  // Check that memref data type is compatible with the expected element type.
  if (LLVM_UNLIKELY(!AreCompatibleTypes(element_type, memref.dtype()))) {
    return InvalidArgumentError(
        StrCat("type is not compatible with the expected element type: ",
               primitive_util::LowercasePrimitiveTypeName(memref.dtype()),
               " vs ", primitive_util::LowercasePrimitiveTypeName(element_type),
               " (", pretty_print(), ")"));
  }

  // Skip sizes verification if they are not available (unranked tensor or
  // memref type is compatible with run-time arguments of any shape).
  if (!sizes.has_value()) return absl::OkStatus();

  // Check that memref rank is the same as the expected rank.
  if (LLVM_UNLIKELY(memref.rank() != sizes->size()))
    return InvalidArgumentError(
        StrCat("rank does not match expected input rank: ", memref.rank(),
               " vs ", sizes->size(), " (", pretty_print(), ")"));

  // Check that all statically known dimensions matches the memref dimensions.
  for (const auto& pair : llvm::enumerate(llvm::zip(memref.sizes(), *sizes))) {
    int64_t argument_dim = std::get<0>(pair.value());
    int64_t expected_dim = std::get<1>(pair.value());

    bool is_dynamic_dim = MemrefType::IsDynamic(expected_dim);

    if (LLVM_UNLIKELY(argument_dim != expected_dim && !is_dynamic_dim))
      return InvalidArgumentError(
          StrCat("dimension #", pair.index(),
                 " does not match expected input dimension: ", argument_dim,
                 " vs ", expected_dim, " (", pretty_print(), ")"));
  }

  return absl::OkStatus();
}

Status MemrefDesc::Verify(const Type& type) const {
  // Only ranked memrefs have a defined ABI and can be passed as an argument.
  if (auto* memref = dyn_cast<MemrefType>(&type))
    return VerifyMemrefArgument(memref->element_type(), memref->sizes(), *this);
  return InvalidArgumentError(
      StrCat("unsupported memref type: ", type.ToString()));
}

size_t MemrefDesc::Pack(absl::Span<void*> args, size_t offset) const {
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

std::string MemrefDesc::ToString() const {
  return StrFormat("MemrefDesc: dtype: %s offset: %i sizes: [%s] strides: [%s]",
                   LowercasePrimitiveTypeName(dtype()), offset(),
                   StrJoin(sizes(), ", "), StrJoin(strides(), ", "));
}

//===----------------------------------------------------------------------===//
// Verify that argument type is compatible with the run-time memref argument.
//===----------------------------------------------------------------------===//

static Status VerifyMemrefArgument(const Type& type, const MemrefDesc& arg) {
  if (auto* memref = dyn_cast<MemrefType>(&type))
    return VerifyMemrefArgument(memref->element_type(), memref->sizes(), arg);
  if (auto* memref = dyn_cast<UnrankedMemrefType>(&type))
    return VerifyMemrefArgument(memref->element_type(), std::nullopt, arg);

  if (auto* tensor = dyn_cast<RankedTensorType>(&type))
    return VerifyMemrefArgument(tensor->element_type(), tensor->sizes(), arg);
  if (auto* tensor = dyn_cast<UnrankedTensorType>(&type))
    return VerifyMemrefArgument(tensor->element_type(), std::nullopt, arg);

  return InvalidArgumentError(
      StrCat("unsupported memref type: ", type.ToString()));
}

Status VerifyMemrefArgument(unsigned index, const Type& type,
                            const MemrefDesc& arg) {
  if (auto st = VerifyMemrefArgument(type, arg); !st.ok())
    return InvalidArgumentError(StrCat("argument #", index, " ", st.message()));
  return absl::OkStatus();
}

// -------------------------------------------------------------------------- //
// BufferDesc.
// -------------------------------------------------------------------------- //

static Status VerifyBufferDesc(PrimitiveType element_type,
                               std::optional<absl::Span<const int64_t>> sizes,
                               const BufferDesc& buffer) {
  size_t n_elem = !sizes.has_value() || sizes->empty() ? 1 : (*sizes)[0];
  size_t expected_buffer_size =
      primitive_util::ByteWidth(element_type) * n_elem;
  if (LLVM_UNLIKELY(expected_buffer_size != buffer.size())) {
    return InvalidArgumentError(StrCat(
        "buffer size is not equal to that expected from the element type: got ",
        buffer.size(), " vs expected ", expected_buffer_size, "."));
  }
  return absl::OkStatus();
}

Status BufferDesc::Verify(const Type& type) const {
  // BufferDesc doesn't have its own type signature; it works with MemrefType.
  if (auto* memref = dyn_cast<MemrefType>(&type))
    return VerifyBufferDesc(memref->element_type(), memref->sizes(), *this);
  return InvalidArgumentError(
      StrCat("unsupported memref type: ", type.ToString()));
}

size_t BufferDesc::Pack(absl::Span<void*> args, size_t offset) const {
  auto cast = [](const void* ptr) { return const_cast<void*>(ptr); };
  // Write into the arguments data starting from the given offset.
  void** p = &args[offset];
  p[0] = cast(&data_);
  p[1] = cast(&data_);
  p[2] = cast(&size_);
  return offset + 3;
}

std::string BufferDesc::ToString() const {
  return StrFormat("BufferDesc: data: %p size: %i", data(), size());
}

}  // namespace runtime
}  // namespace xla
