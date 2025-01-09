/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/dtensor/cc/slice_util.h"

#include <optional>
#include <string>
#include <vector>

#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"

namespace tensorflow {
namespace dtensor {
namespace slice_util {

namespace {

// Computes the size of the ellipsis and the output rank.
StatusOr<int64_t> GetEllipsisSize(int64_t input_rank,
                                  const std::vector<Token>& tokens,
                                  int64_t* output_rank) {
  bool found = false;
  int64_t regular_axis = 0;
  int64_t new_axis = 0;
  int64_t shrink_axis = 0;
  for (const auto& token : tokens) {
    switch (token.token_type) {
      case Token::ELLIPSIS:
        if (found) {
          return absl::InvalidArgumentError(
              "More than one ellipsis was found.");
        }
        found = true;
        break;
      case Token::NEW_AXIS:
        ++new_axis;
        break;
      case Token::SHRINK_AXIS:
        ++shrink_axis;
        break;
      case Token::REGULAR:
        ++regular_axis;
        break;
    }
  }
  int64_t ellipsis_size = input_rank - (regular_axis + shrink_axis);
  if (found && ellipsis_size < 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Ellipsis was found, but there is no remaining axis for it.",
        " input_rank=", input_rank, " regular_axis=", regular_axis,
        " shrink_axis=", shrink_axis));
  }
  *output_rank = regular_axis + ellipsis_size + new_axis;
  return ellipsis_size;
}

}  // namespace

Token Token::normalize(int64_t dim_size) const {
  if (dynamic_mask) {
    return *this;
  }
  int64_t new_begin = begin;
  int dir = (stride > 0) ? 1 : -1;
  if (begin_mask) {
    if (dir > 0) {
      new_begin = 0;
    } else {
      new_begin = dim_size - 1;
    }
  }
  int64_t new_end = end;
  if (end_mask) {
    if (dir > 0) {
      new_end = dim_size;
    } else {
      new_end = -1;
    }
  }
  // Shift begin and end by same number of periods to distinguish full cycle
  // from empty.
  int64_t shift = (new_begin - new_begin % dim_size);
  new_begin -= shift;
  new_end -= shift;

  int64_t n = dir * (new_end - new_begin + stride - dir) / (dir * stride);

  // Round end by cycle size to ensure `(end - begin) / strides` is the
  // number of result elements. To support cases like begin=0, end=-1.
  if (n < 0) {
    new_end = new_end + dir * dim_size;
  }
  n = dir * (new_end - new_begin + stride - dir) / (dir * stride);
  new_end = new_begin + n * stride;

  Token r = *this;
  r.begin = new_begin;
  r.end = new_end;
  return r;
}

// Returns a Token for local slicing if no relayout along this axis
// is needed. If no such local slicing is possible, returns nullopt.
std::optional<Token> Token::GetLocalToken(int64_t dim_size,
                                          int64_t num_shards) const {
  Token token = normalize(dim_size);
  VLOG(5) << "Compute: "
          << "dim_size=" << dim_size << " num_shards=" << num_shards
          << " token.begin=" << token.begin << " token.end=" << token.end
          << " token.stride=" << token.stride;
  if (token.begin_mask && token.end_mask) return token;
  if (token.dynamic_mask) return std::nullopt;
  if (token.stride < 0) return std::nullopt;
  int64_t shard_dim_size = dim_size / num_shards;
  if (shard_dim_size % token.stride == 0) {
    // Simple striped slicing, where every 1 out of stride items
    // are selected can remain sharded the same way.
    if (token.begin >= 0 && token.begin < token.stride &&
        token.end >= dim_size && token.end < dim_size + token.stride) {
      token.end = shard_dim_size + (token.end - dim_size);
      return token;
    }
  }
  return std::nullopt;
}

absl::Status TokenProcessor::Run(const std::vector<Token>& tokens) {
  int64_t input_rank = input_rank_;
  int64_t output_rank;
  TF_ASSIGN_OR_RETURN(int64_t ellipsis_size,
                      GetEllipsisSize(input_rank, tokens, &output_rank));

  PrepareResults(tokens.size(), input_rank, output_rank);

  bool out_of_bound = false;
  int64_t input_index = 0;
  int64_t output_index = 0;

  for (const auto& token : tokens) {
    switch (token.token_type) {
      case Token::ELLIPSIS:
        VisitEllipsisAxis(token);
        out_of_bound = VisitLoop(input_rank, output_rank, ellipsis_size,
                                 &input_index, &output_index);
        ellipsis_size = 0;
        break;
      case Token::SHRINK_AXIS:
        VisitShrinkAxis(token, input_index, output_index);
        ++input_index;
        break;
      case Token::NEW_AXIS:
        VisitNewAxis(token, input_index, output_index);
        ++output_index;
        break;
      case Token::REGULAR:
        if (input_index >= input_rank) {
          out_of_bound = true;
          break;
        }
        VisitRegularAxis(token, input_index, output_index);
        ++input_index;
        ++output_index;
        break;
    }

    if (out_of_bound) {
      break;
    }
  }
  if (ellipsis_size > 0) {
    out_of_bound = VisitLoop(input_rank, output_rank, ellipsis_size,
                             &input_index, &output_index);
  }
  if (out_of_bound) {
    return absl::InvalidArgumentError(
        "Reading axis beyond the input tensor's rank. "
        "The slicing token is incorrect.");
  }

  return FinalizeResults(input_rank, output_rank);
}

bool TokenProcessor::VisitLoop(int64_t input_rank, int64_t output_rank,
                               int64_t ellipsis_size, int64_t* input_index,
                               int64_t* output_index) {
  for (int64_t k = 0; k < ellipsis_size; ++k) {
    if (*input_index >= input_rank) {
      return true;
    }
    VisitImplicitAxis(*input_index, *output_index);
    ++*input_index;
    ++*output_index;
  }
  return false;
}

}  // namespace slice_util
}  // namespace dtensor
}  // namespace tensorflow
