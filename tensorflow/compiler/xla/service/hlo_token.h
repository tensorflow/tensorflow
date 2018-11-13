/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_TOKEN_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_TOKEN_H_

#include <string>

#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// Defines different kinds of tokens in a hlo module string.
//
// You shouldn't need to use this directly unless you're using HloLexer
// directly, and you probably don't need to do that.  Use hlo_parser instead.
enum class TokKind {
  // Markers
  kEof,
  kError,

  // Tokens with no info.
  kEqual,  // =
  kComma,  // ,
  kColon,  // :
  kLsquare,
  kRsquare,  // [  ]
  kLbrace,
  kRbrace,  // {  }
  kLparen,
  kRparen,  // (  )

  kArrow,    // ->

  // Keywords
  kw_HloModule,
  kw_ENTRY,
  kw_ROOT,
  kw_true,
  kw_false,
  kw_maximal,
  kw_replicated,
  kw_nan,
  kw_inf,

  kNegInf,  // -inf

  // Typed tokens.
  kName,           // %foo
  kAttributeName,  // dimensions=
  kDimLabels,      // [0-9bf]{2,}_[0-9io]{2,}->[0-9bf]{2,}
  kDxD,            // [0-9]+(x[0-9]+)+
  kPad,            // [0-9]+_[0-9]+(_[0-9]+)?(x[0-9]+_[0-9]+(_[0-9]+)?)*
  kIdent,          // other identifiers
  kString,         // "abcd\"\n"
  kShape,          // f32[2,3]{1,0}
  kInt,            // 42
  kDecimal,        // 4.2
};

string TokKindToString(TokKind kind);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_TOKEN_H_
