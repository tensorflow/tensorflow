//===- LLVM.h - Import and forward declare core LLVM types ------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file forward declares and imports various common LLVM datatypes that
// MLIR wants to use unqualified.
//
// Note that most of these are forward declared and then imported into the MLIR
// namespace with using decls, rather than being #included.  This is because we
// want clients to explicitly #include the files they need.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_LLVM_H
#define MLIR_SUPPORT_LLVM_H

// We include these two headers because they cannot be practically forward
// declared, and are effectively language features.
#include "llvm/ADT/None.h"
#include "llvm/Support/Casting.h"

// Forward declarations.
namespace llvm {
// Containers.
class StringRef;
class StringLiteral;
class Twine;
template <typename T> class SmallPtrSetImpl;
template <typename T, unsigned N> class SmallPtrSet;
template <typename T> class SmallVectorImpl;
template <typename T, unsigned N> class SmallVector;
template <unsigned N> class SmallString;
template <typename T> class ArrayRef;
template <typename T> class MutableArrayRef;
template <typename T> class TinyPtrVector;
template <typename T> class Optional;
template <typename... PT> class PointerUnion;
namespace detail {
template <typename KeyT, typename ValueT> struct DenseMapPair;
}
template <typename T> struct DenseMapInfo;
template <typename ValueT, typename ValueInfoT> class DenseSet;
template <typename KeyT, typename ValueT, typename KeyInfoT, typename BucketT>
class DenseMap;
template <typename Fn> class function_ref;
template <typename IteratorT> class iterator_range;

// Other common classes.
class raw_ostream;
class APInt;
class APFloat;
} // end namespace llvm

namespace mlir {
// Casting operators.
using llvm::cast;
using llvm::cast_or_null;
using llvm::dyn_cast;
using llvm::dyn_cast_or_null;
using llvm::isa;
using llvm::isa_and_nonnull;

// Containers.
using llvm::ArrayRef;
using llvm::DenseMapInfo;
template <typename KeyT, typename ValueT,
          typename KeyInfoT = DenseMapInfo<KeyT>,
          typename BucketT = llvm::detail::DenseMapPair<KeyT, ValueT>>
using DenseMap = llvm::DenseMap<KeyT, ValueT, KeyInfoT, BucketT>;
template <typename ValueT, typename ValueInfoT = DenseMapInfo<ValueT>>
using DenseSet = llvm::DenseSet<ValueT, ValueInfoT>;
template <typename Fn> using function_ref = llvm::function_ref<Fn>;
using llvm::iterator_range;
using llvm::MutableArrayRef;
using llvm::None;
using llvm::Optional;
using llvm::PointerUnion;
using llvm::SmallPtrSet;
using llvm::SmallPtrSetImpl;
using llvm::SmallString;
using llvm::SmallVector;
using llvm::SmallVectorImpl;
using llvm::StringLiteral;
using llvm::StringRef;
using llvm::TinyPtrVector;
using llvm::Twine;

// Other common classes.
using llvm::APFloat;
using llvm::APInt;
using llvm::raw_ostream;
} // namespace mlir

#endif // MLIR_SUPPORT_LLVM_H
