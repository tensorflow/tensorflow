//===- Format.cpp - Utilities for String Format ---------------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines utilities for formatting strings. They are specially
// tailored to the needs of TableGen'ing op definitions and rewrite rules,
// so they are not expected to be used as widely applicable utilities.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/Format.h"
#include <cctype>

using namespace mlir;
using namespace mlir::tblgen;

// Marker to indicate an error happened when replacing a placeholder.
const char *const kMarkerForNoSubst = "<no-subst-found>";

FmtContext &tblgen::FmtContext::addSubst(StringRef placeholder, Twine subst) {
  customSubstMap[placeholder] = subst.str();
  return *this;
}

FmtContext &tblgen::FmtContext::withBuilder(Twine subst) {
  builtinSubstMap[PHKind::Builder] = subst.str();
  return *this;
}

FmtContext &tblgen::FmtContext::withOp(Twine subst) {
  builtinSubstMap[PHKind::Op] = subst.str();
  return *this;
}

FmtContext &tblgen::FmtContext::withSelf(Twine subst) {
  builtinSubstMap[PHKind::Self] = subst.str();
  return *this;
}

Optional<StringRef>
tblgen::FmtContext::getSubstFor(FmtContext::PHKind placeholder) const {
  if (placeholder == FmtContext::PHKind::None ||
      placeholder == FmtContext::PHKind::Custom)
    return {};
  auto it = builtinSubstMap.find(placeholder);
  if (it == builtinSubstMap.end())
    return {};
  return StringRef(it->second);
}

Optional<StringRef>
tblgen::FmtContext::getSubstFor(StringRef placeholder) const {
  auto it = customSubstMap.find(placeholder);
  if (it == customSubstMap.end())
    return {};
  return StringRef(it->second);
}

FmtContext::PHKind tblgen::FmtContext::getPlaceHolderKind(StringRef str) {
  return llvm::StringSwitch<FmtContext::PHKind>(str)
      .Case("_builder", FmtContext::PHKind::Builder)
      .Case("_op", FmtContext::PHKind::Op)
      .Case("_self", FmtContext::PHKind::Self)
      .Case("", FmtContext::PHKind::None)
      .Default(FmtContext::PHKind::Custom);
}

std::pair<FmtReplacement, StringRef>
tblgen::FmtObjectBase::splitFmtSegment(StringRef fmt) {
  size_t begin = fmt.find_first_of('$');
  if (begin == StringRef::npos) {
    // No placeholders: the whole format string should be returned as a
    // literal string.
    return {FmtReplacement{fmt}, StringRef()};
  }
  if (begin != 0) {
    // The first placeholder is not at the beginning: we can split the format
    // string into a literal string and the rest.
    return {FmtReplacement{fmt.substr(0, begin)}, fmt.substr(begin)};
  }

  // The first placeholder is at the beginning

  if (fmt.size() == 1) {
    // The whole format string just contains '$': treat as literal.
    return {FmtReplacement{fmt}, StringRef()};
  }

  // Allow escaping dollar with '$$'
  if (fmt[1] == '$') {
    return {FmtReplacement{fmt.substr(0, 1)}, fmt.substr(2)};
  }

  // First try to see if it's a positional placeholder, and then handle special
  // placeholders.

  size_t end = fmt.find_if_not([](char c) { return std::isdigit(c); }, 1);
  if (end != 1) {
    // We have a positional placeholder. Parse the index.
    size_t index = 0;
    if (fmt.substr(1, end - 1).consumeInteger(0, index)) {
      llvm_unreachable("invalid replacement sequence index");
    }

    if (end == StringRef::npos) {
      // All the remaining characters are part of the positional placeholder.
      return {FmtReplacement{fmt, index}, StringRef()};
    }
    return {FmtReplacement{fmt.substr(0, end), index}, fmt.substr(end)};
  }

  end = fmt.find_if_not([](char c) { return std::isalnum(c) || c == '_'; }, 1);
  auto placeholder = FmtContext::getPlaceHolderKind(fmt.substr(1, end - 1));
  if (end == StringRef::npos) {
    // All the remaining characters are part of the special placeholder.
    return {FmtReplacement{fmt, placeholder}, StringRef()};
  }
  return {FmtReplacement{fmt.substr(0, end), placeholder}, fmt.substr(end)};
}

std::vector<FmtReplacement> FmtObjectBase::parseFormatString(StringRef fmt) {
  std::vector<FmtReplacement> replacements;
  FmtReplacement repl;
  while (!fmt.empty()) {
    std::tie(repl, fmt) = splitFmtSegment(fmt);
    if (repl.type != FmtReplacement::Type::Empty)
      replacements.push_back(repl);
  }
  return replacements;
}

void FmtObjectBase::format(raw_ostream &s) const {
  for (auto &repl : replacements) {
    if (repl.type == FmtReplacement::Type::Empty)
      continue;

    if (repl.type == FmtReplacement::Type::Literal) {
      s << repl.spec;
      continue;
    }

    if (repl.type == FmtReplacement::Type::SpecialPH) {
      if (repl.placeholder == FmtContext::PHKind::None) {
        s << repl.spec;
      } else if (!context) {
        // We need the context to replace special placeholders.
        s << repl.spec << kMarkerForNoSubst;
      } else {
        Optional<StringRef> subst;
        if (repl.placeholder == FmtContext::PHKind::Custom) {
          // Skip the leading '$' sign for the custom placeholder
          subst = context->getSubstFor(repl.spec.substr(1));
        } else {
          subst = context->getSubstFor(repl.placeholder);
        }
        if (subst)
          s << *subst;
        else
          s << repl.spec << kMarkerForNoSubst;
      }
      continue;
    }

    assert(repl.type == FmtReplacement::Type::PositionalPH);

    if (repl.index >= adapters.size()) {
      s << repl.spec << kMarkerForNoSubst;
      continue;
    }
    adapters[repl.index]->format(s, /*Options=*/"");
  }
}
