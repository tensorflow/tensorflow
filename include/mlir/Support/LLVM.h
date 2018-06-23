//===- LLVM.h - Import and forward declare core LLVM types ------*- C++ -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
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
#include "llvm/Support/Casting.h"
#include "llvm/ADT/None.h"

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
  template<typename T> class ArrayRef;
  template<typename T> class MutableArrayRef;
  template<typename T> class TinyPtrVector;
  template<typename T> class Optional;
  template <typename PT1, typename PT2> class PointerUnion;
  namespace detail {
    template <typename KeyT, typename ValueT> struct DenseMapPair;
  }
  template<typename T> struct DenseMapInfo;
  template <typename KeyT, typename ValueT, typename KeyInfoT, typename BucketT>
  class DenseMap;

  // Other common classes.
  class raw_ostream;
  class APInt;
  class APFloat;
} // end namespace llvm


namespace mlir {
  // Casting operators.
  using llvm::isa;
  using llvm::cast;
  using llvm::dyn_cast;
  using llvm::dyn_cast_or_null;
  using llvm::cast_or_null;

  // Containers.
  using llvm::None;
  using llvm::Optional;
  using llvm::SmallPtrSetImpl;
  using llvm::SmallPtrSet;
  using llvm::SmallString;
  using llvm::StringRef;
  using llvm::StringLiteral;
  using llvm::Twine;
  using llvm::SmallVectorImpl;
  using llvm::SmallVector;
  using llvm::ArrayRef;
  using llvm::MutableArrayRef;
  using llvm::TinyPtrVector;
  using llvm::PointerUnion;
  using llvm::DenseMap;

  // Other common classes.
  using llvm::raw_ostream;
  using llvm::APInt;
  using llvm::APFloat;
  using llvm::NoneType;
} // end namespace swift

#endif // MLIR_SUPPORT_LLVM_H
