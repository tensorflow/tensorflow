/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/runtime/arguments.h"

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "xla/primitive_util.h"
#include "xla/runtime/async_runtime.h"
#include "xla/runtime/types.h"

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
  if (isa<OpaqueOperandType>(type)) return absl::OkStatus();
  return InvalidArgumentError(
      StrCat("unsupported opaque argument type: ", type.ToString()));
}

void OpaqueArg::Pack(absl::Span<void*> args) const {
  args[0] = const_cast<void*>(reinterpret_cast<const void*>(&ptr_));
}

std::string OpaqueArg::ToString() const {
  return StrFormat("OpaqueArg: ptr=%p", ptr_);
}

//===----------------------------------------------------------------------===//
// ScalarArg.
//===----------------------------------------------------------------------===//

Status ScalarArg::Verify(const Type& type) const {
  auto* scalar = dyn_cast<ScalarType>(&type);
  if (scalar && scalar->type() == type_) return absl::OkStatus();
  return InvalidArgumentError(
      StrCat("unsupported scalar argument type: ", type.ToString()));
}

void ScalarArg::Pack(absl::Span<void*> args) const {
  args[0] = const_cast<void*>(reinterpret_cast<const void*>(&value_));
}

std::string ScalarArg::ToString() const {
  return primitive_util::LowercasePrimitiveTypeName(type_);
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
      return d == MemrefType::kDynamic ? "?" : std::to_string(d);
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

void MemrefDesc::Pack(absl::Span<void*> args) const {
  auto cast = [](const void* p) { return const_cast<void*>(p); };

  // Packs memref with a rank not known at compile time.
  auto pack_memref = [&](int64_t rank) {
    args[0] = cast(&data_);  // memref.basePtr
    args[1] = cast(&data_);  // memref.data
    args[2] = cast(&offset_);
    for (int64_t d = 0; d < rank; ++d) {
      args[3 + d] = cast(&sizes_and_strides_[d]);
      args[3 + rank + d] = cast(&sizes_and_strides_[rank_ + d]);
    }
  };

  // Packs memref with a rank known at compile time.
  auto pack_ranked_memref = [&](auto rank_tag) {
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

//===----------------------------------------------------------------------===//
// AsyncTokenArg.
//===----------------------------------------------------------------------===//

Status AsyncTokenArg::Verify(const Type& type) const {
  if (isa<AsyncTokenType>(type)) return absl::OkStatus();
  return InvalidArgumentError(
      absl::StrCat("expected async token type, got: ", type.ToString()));
}

void AsyncTokenArg::Pack(absl::Span<void*> args) const {
  args[0] = const_cast<void*>(reinterpret_cast<const void*>(&storage_));
}

std::string AsyncTokenArg::ToString() const { return "Async token argument"; }

//===----------------------------------------------------------------------===//
// AsyncScalarArg.
//===----------------------------------------------------------------------===//

absl::Status AsyncScalarArg::Verify(const Type& type) const {
  auto* value_type = llvm::dyn_cast<AsyncValueType>(&type);
  if (!value_type)
    return absl::InvalidArgumentError(
        absl::StrCat("expected async value type, got: ", type.ToString()));

  auto* scalar = llvm::dyn_cast<ScalarType>(&value_type->value_type());
  if (scalar && scalar->type() == type_) return absl::OkStatus();
  return absl::InvalidArgumentError(
      absl::StrCat("unsupported scalar argument type: ", type.ToString()));
}

void AsyncScalarArg::Pack(absl::Span<void*> args) const {
  args[0] = const_cast<void*>(reinterpret_cast<const void*>(&storage_));
}

std::string AsyncScalarArg::ToString() const {
  return absl::StrFormat("Async value type: %s",
                         primitive_util::LowercasePrimitiveTypeName(type_));
}

//===----------------------------------------------------------------------===//
// AsyncMemrefArg.
//===----------------------------------------------------------------------===//

AsyncMemrefArg::AsyncMemrefArg(tsl::AsyncValueRef<MemrefDesc> value)
    : value_(value) {
  struct MemrefDescriptor {
    void* allocated_ptr;
    void* aligned_ptr;
    int64_t offset;
    int64_t dims[];
  };

  auto size_and_alignment =
      [](const MemrefDesc* desc) -> std::pair<size_t, size_t> {
    size_t size = 3 * sizeof(int64_t) + 2 * desc->rank() * sizeof(int64_t);
    return std::make_pair(size, alignof(std::max_align_t));
  };

  auto write = [](const MemrefDesc* v, std::byte* store) {
    MemrefDescriptor* store_t = reinterpret_cast<MemrefDescriptor*>(store);
    auto rank = v->rank();
    for (unsigned i = 0; i < rank; ++i) {
      store_t->dims[i] = v->size(i);
      store_t->dims[i + rank] = v->stride(i);
    }

    store_t->allocated_ptr = v->data();
    store_t->aligned_ptr = v->data();
    store_t->offset = 0;
  };

  storage_ =
      AsyncRuntime::AsValue<MemrefDesc>(value_, size_and_alignment, write);
}

Status AsyncMemrefArg::Verify(const Type& type) const {
  auto* value_type = llvm::dyn_cast<AsyncValueType>(&type);
  if (!value_type)
    return InvalidArgumentError(
        absl::StrCat("expected async value type, got: ", type.ToString()));
  auto* memref = llvm::dyn_cast<MemrefType>(&value_type->value_type());
  if (!memref)
    return InvalidArgumentError(
        absl::StrCat("expected async memref type, got ",
                     value_type->value_type().ToString()));
  value_.AndThen([memref](absl::StatusOr<MemrefDesc*> status_or) {
    if (!status_or.ok()) {
      llvm::errs() << status_or.status().message();
      assert(false && "async memref argument is in error state");
    } else {
      auto status = VerifyMemrefArgument(memref->element_type(),
                                         memref->sizes(), **status_or);
      if (!status.ok()) {
        llvm::errs() << status.message();
        assert(false && "failed to verify memref argument");
      }
    }
  });
  return absl::OkStatus();
}

void AsyncMemrefArg::Pack(absl::Span<void*> args) const {
  args[0] = const_cast<void*>(reinterpret_cast<const void*>(&storage_));
}

std::string AsyncMemrefArg::ToString() const { return "Async memref argument"; }

}  // namespace runtime
}  // namespace xla
