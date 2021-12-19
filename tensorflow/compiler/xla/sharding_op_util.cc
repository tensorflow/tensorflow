/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/sharding_op_util.h"

#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/hlo_lexer.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
namespace sharding_op_util {

std::string EncodeAttributes(absl::Span<const int64_t> unspecified_dims) {
  if (unspecified_dims.empty()) {
    return "";
  }
  return absl::StrCat("unspecified_dims=[",
                      absl::StrJoin(unspecified_dims, ","), "]");
}

Status ParseAttributes(absl::string_view opaque,
                       std::vector<int64_t>* unspecified_dims) {
  HloLexer lexer(opaque);
  while (lexer.Lex() != TokKind::kEof) {
    if (lexer.GetKind() != TokKind::kAttributeName) {
      return InvalidArgumentStrCat("Cannot parse sharding op attributes: ",
                                   opaque);
    }
    std::string attr_name = lexer.GetStrVal();
    if (attr_name == "unspecified_dims") {
      TF_RET_CHECK(lexer.Lex() == TokKind::kLsquare);
      while (lexer.Lex() == TokKind::kInt) {
        unspecified_dims->push_back(lexer.GetInt64Val());
        if (lexer.Lex() != TokKind::kComma) {
          break;
        }
      }
      TF_RET_CHECK(lexer.GetKind() == TokKind::kRsquare);
    } else {
      return InvalidArgumentStrCat("Unknown attribute name in sharding op: ",
                                   attr_name);
    }
  }
  return Status::OK();
}

}  // namespace sharding_op_util
}  // namespace xla
