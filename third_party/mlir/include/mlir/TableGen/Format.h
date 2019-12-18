//===- Format.h - Utilities for String Format -------------------*- C++ -*-===//
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
// This file declares utilities for formatting strings. They are specially
// tailored to the needs of TableGen'ing op definitions and rewrite rules,
// so they are not expected to be used as widely applicable utilities.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_FORMAT_H_
#define MLIR_TABLEGEN_FORMAT_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FormatVariadic.h"

namespace mlir {
namespace tblgen {

/// Format context containing substitutions for special placeholders.
///
/// This context divides special placeholders into two categories: builtin ones
/// and custom ones.
///
/// Builtin placeholders are baked into `FmtContext` and each one of them has a
/// dedicated setter. They can be used in all dialects. Their names follow the
/// convention of `$_<name>`. The rationale of the leading underscore is to
/// avoid confusion and name collision: op arguments/attributes/results are
/// named as $<name>, and we can potentially support referencing those entities
/// directly in the format template in the future.
//
/// Custom ones are registered by dialect-specific TableGen backends and use the
/// same unified setter.
class FmtContext {
public:
  // Placeholder kinds
  enum class PHKind : char {
    None,
    Custom,  // For custom placeholders
    Builder, // For the $_builder placeholder
    Op,      // For the $_op placeholder
    Self,    // For the $_self placeholder
  };

  FmtContext() = default;

  // Setter for custom placeholders
  FmtContext &addSubst(StringRef placeholder, Twine subst);

  // Setters for builtin placeholders
  FmtContext &withBuilder(Twine subst);
  FmtContext &withOp(Twine subst);
  FmtContext &withSelf(Twine subst);

  Optional<StringRef> getSubstFor(PHKind placeholder) const;
  Optional<StringRef> getSubstFor(StringRef placeholder) const;

  static PHKind getPlaceHolderKind(StringRef str);

private:
  struct PHKindInfo : DenseMapInfo<PHKind> {
    using CharInfo = DenseMapInfo<char>;

    static inline PHKind getEmptyKey() {
      return static_cast<PHKind>(CharInfo::getEmptyKey());
    }
    static inline PHKind getTombstoneKey() {
      return static_cast<PHKind>(CharInfo::getTombstoneKey());
    }
    static unsigned getHashValue(const PHKind &val) {
      return CharInfo::getHashValue(static_cast<char>(val));
    }

    static bool isEqual(const PHKind &lhs, const PHKind &rhs) {
      return lhs == rhs;
    }
  };

  llvm::SmallDenseMap<PHKind, std::string, 4, PHKindInfo> builtinSubstMap;
  llvm::StringMap<std::string> customSubstMap;
};

/// Struct representing a replacement segment for the formatted string. It can
/// be a segment of the formatting template (for `Literal`) or a replacement
/// parameter (for `PositionalPH` and `SpecialPH`).
struct FmtReplacement {
  enum class Type { Empty, Literal, PositionalPH, SpecialPH };

  FmtReplacement() = default;
  explicit FmtReplacement(StringRef literal)
      : type(Type::Literal), spec(literal) {}
  FmtReplacement(StringRef spec, size_t index)
      : type(Type::PositionalPH), spec(spec), index(index) {}
  FmtReplacement(StringRef spec, FmtContext::PHKind placeholder)
      : type(Type::SpecialPH), spec(spec), placeholder(placeholder) {}

  Type type = Type::Empty;
  StringRef spec;
  size_t index = 0;
  FmtContext::PHKind placeholder = FmtContext::PHKind::None;
};

class FmtObjectBase {
private:
  static std::pair<FmtReplacement, StringRef> splitFmtSegment(StringRef fmt);
  static std::vector<FmtReplacement> parseFormatString(StringRef fmt);

protected:
  // The parameters are stored in a std::tuple, which does not provide runtime
  // indexing capabilities.  In order to enable runtime indexing, we use this
  // structure to put the parameters into a std::vector.  Since the parameters
  // are not all the same type, we use some type-erasure by wrapping the
  // parameters in a template class that derives from a non-template superclass.
  // Essentially, we are converting a std::tuple<Derived<Ts...>> to a
  // std::vector<Base*>.
  struct CreateAdapters {
    template <typename... Ts>
    std::vector<llvm::detail::format_adapter *> operator()(Ts &... items) {
      return std::vector<llvm::detail::format_adapter *>{&items...};
    }
  };

  StringRef fmt;
  const FmtContext *context;
  std::vector<llvm::detail::format_adapter *> adapters;
  std::vector<FmtReplacement> replacements;

public:
  FmtObjectBase(StringRef fmt, const FmtContext *ctx, size_t numParams)
      : fmt(fmt), context(ctx), replacements(parseFormatString(fmt)) {}

  FmtObjectBase(const FmtObjectBase &that) = delete;

  FmtObjectBase(FmtObjectBase &&that)
      : fmt(std::move(that.fmt)), context(that.context),
        adapters(), // adapters are initialized by FmtObject
        replacements(std::move(that.replacements)) {}

  void format(llvm::raw_ostream &s) const;

  std::string str() const {
    std::string result;
    llvm::raw_string_ostream s(result);
    format(s);
    return s.str();
  }

  template <unsigned N> SmallString<N> sstr() const {
    SmallString<N> result;
    llvm::raw_svector_ostream s(result);
    format(s);
    return result;
  }

  template <unsigned N> operator SmallString<N>() const { return sstr<N>(); }

  operator std::string() const { return str(); }
};

template <typename Tuple> class FmtObject : public FmtObjectBase {
  // Storage for the parameter adapters.  Since the base class erases the type
  // of the parameters, we have to own the storage for the parameters here, and
  // have the base class store type-erased pointers into this tuple.
  Tuple parameters;

public:
  FmtObject(StringRef fmt, const FmtContext *ctx, Tuple &&params)
      : FmtObjectBase(fmt, ctx, std::tuple_size<Tuple>::value),
        parameters(std::move(params)) {
    adapters.reserve(std::tuple_size<Tuple>::value);
    adapters = llvm::apply_tuple(CreateAdapters(), parameters);
  }

  FmtObject(FmtObject const &that) = delete;

  FmtObject(FmtObject &&that)
      : FmtObjectBase(std::move(that)), parameters(std::move(that.parameters)) {
    adapters.reserve(that.adapters.size());
    adapters = llvm::apply_tuple(CreateAdapters(), parameters);
  }
};

/// Formats text by substituting placeholders in format string with replacement
/// parameters.
///
/// There are two categories of placeholders accepted, both led by a '$' sign:
///
/// 1. Positional placeholder: $[0-9]+
/// 2. Special placeholder:    $[a-zA-Z_][a-zA-Z0-9_]*
///
/// Replacement parameters for positional placeholders are supplied as the
/// `vals` parameter pack with 1:1 mapping. That is, $0 will be replaced by the
/// first parameter in `vals`, $1 by the second one, and so on. Note that you
/// can use the positional placeholders in any order and repeat any times, for
/// example, "$2 $1 $1 $0" is accepted.
///
/// Replacement parameters for special placeholders are supplied using the `ctx`
/// format context.
///
/// The `fmt` is recorded as a `StringRef` inside the returned `FmtObject`.
/// The caller needs to make sure the underlying data is available when the
/// `FmtObject` is used.
///
/// `ctx` accepts a nullptr if there is no special placeholder is used.
///
/// If no substitution is provided for a placeholder or any error happens during
/// format string parsing or replacement, the placeholder will be outputted
/// as-is with an additional marker '<no-subst-found>', to aid debugging.
///
/// To print a '$' literally, escape it with '$$'.
///
/// This utility function is inspired by LLVM formatv(), with modifications
/// specially tailored for TableGen C++ generation usage:
///
/// 1. This utility use '$' instead of '{' and '}' for denoting the placeholder
///    because '{' and '}' are frequently used in C++ code.
/// 2. This utility does not support format layout because it is rarely needed
///    in C++ code generation.
template <typename... Ts>
inline auto tgfmt(StringRef fmt, const FmtContext *ctx, Ts &&... vals)
    -> FmtObject<decltype(std::make_tuple(
        llvm::detail::build_format_adapter(std::forward<Ts>(vals))...))> {
  using ParamTuple = decltype(std::make_tuple(
      llvm::detail::build_format_adapter(std::forward<Ts>(vals))...));
  return FmtObject<ParamTuple>(
      fmt, ctx,
      std::make_tuple(
          llvm::detail::build_format_adapter(std::forward<Ts>(vals))...));
}

} // end namespace tblgen
} // end namespace mlir

#endif // MLIR_TABLEGEN_FORMAT_H_
