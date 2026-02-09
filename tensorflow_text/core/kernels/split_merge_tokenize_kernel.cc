// Copyright 2025 TF.Text Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "icu4c/source/common/unicode/uchar.h"
#include "icu4c/source/common/unicode/umachine.h"
#include "icu4c/source/common/unicode/utf8.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace text {

namespace {

// Returns the length (number of bytes) of the UTF8 code point starting at src,
// by reading only the byte from address src.
//
// The result is a number from the set {1, 2, 3, 4}.
int OneCharLen(const char* src) {
  // On most platforms, char is unsigned by default, but iOS is an exception.
  // The cast below makes sure we always interpret *src as an unsigned char.
  return "\1\1\1\1\1\1\1\1\1\1\1\1\2\2\3\4"
      [(*(reinterpret_cast<const unsigned char*>(src)) & 0xFF) >> 4];
}

bool GetUTF8Chars(absl::string_view text,
                  std::vector<absl::string_view>* chars) {
  const char* start = text.data();
  const char* end = text.data() + text.size();
  while (start < end) {
    const int char_length = OneCharLen(start);
    if (char_length <= 0) {
      return false;
    }
    chars->emplace_back(start, char_length);
    start += char_length;
  }
  return true;
}

bool IsBreakChar(absl::string_view text) {
  UChar32 c;
  int position = 0;
  U8_NEXT_OR_FFFD(text.data(), position, text.length(), c);
  return u_isUWhiteSpace(c);
}

Status TokenizeByLabel(const absl::string_view& text,
                       const Tensor& labels_tensor,
                       bool force_split_at_break_character,
                       std::vector<std::string>* tokens,
                       std::vector<int>* begin_offset,
                       std::vector<int>* end_offset, int* num_tokens) {
  std::vector<absl::string_view> chars;
  if (!GetUTF8Chars(text, &chars)) {
    return Status(static_cast<::absl::StatusCode>(
                      absl::StatusCode::kInvalidArgument),
                  absl::StrCat("Input string is not utf8 valid: ", text));
  }

  if (chars.size() > labels_tensor.dim_size(0)) {
    return Status(static_cast<::absl::StatusCode>(
                      absl::StatusCode::kInvalidArgument),
                  absl::StrCat("Number of labels ", labels_tensor.dim_size(0),
                               " is insufficient for text ", text));
  }

  const int split_label = 0;
  bool last_character_is_break_character = false;
  int start = 0;
  bool has_new_token_generated_for_text = false;
  const auto& labels = labels_tensor.unaligned_flat<int32>();
  for (int i = 0; i < chars.size(); ++i) {
    const bool is_break_character = IsBreakChar(chars[i]);
    if (!is_break_character) {
      if (labels(i) == split_label || !has_new_token_generated_for_text ||
          (last_character_is_break_character &&
           force_split_at_break_character)) {
        tokens->emplace_back(chars[i].data(), chars[i].length());
        begin_offset->push_back(start);
        end_offset->push_back(start + chars[i].length());
        *num_tokens += 1;
        has_new_token_generated_for_text = true;
      } else {
        tokens->back().append(chars[i].data(), chars[i].length());
        end_offset->back() = start + chars[i].length();
      }
    }

    start += chars[i].length();
    last_character_is_break_character = is_break_character;
  }

  return absl::OkStatus();
}

}  // namespace

class SplitMergeTokenizeWithOffsetsOp : public OpKernel {
 public:
  explicit SplitMergeTokenizeWithOffsetsOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("force_split_at_break_character",
                                     &force_split_at_break_character_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* input_values;
    OP_REQUIRES_OK(ctx, ctx->input("input_values", &input_values));

    const Tensor* labels;
    OP_REQUIRES_OK(ctx, ctx->input("labels", &labels));
    const Tensor* row_splits;
    OP_REQUIRES_OK(ctx, ctx->input("row_splits", &row_splits));
    OP_REQUIRES(ctx, input_values->dim_size(0) == row_splits->dim_size(0) - 1,
                errors::InvalidArgument("Expecting row_splits have ",
                                        input_values->dim_size(0) + 1,
                                        " elements, got ",
                                        row_splits->dim_size(0)));

    std::vector<string> tokens;
    std::vector<int> begin_offset;
    std::vector<int> end_offset;
    std::vector<int> output_row_splits(1, 0);

    // Iterate through all the values and tokenize them.
    const auto& values_vec = input_values->flat<tstring>();
    const auto& row_splits_vec = row_splits->flat<int32>();
    for (int i = 0; i < values_vec.size(); ++i) {
      // Tokenize into tokens and record the offset locations.
      int num_tokens = 0;
      OP_REQUIRES_OK(
          ctx, TokenizeByLabel(
                   values_vec(i),
                   labels->Slice(row_splits_vec(i), row_splits_vec(i + 1)),
                   force_split_at_break_character_, &tokens, &begin_offset,
                   &end_offset, &num_tokens));

      // Record the row splits.
      output_row_splits.push_back(num_tokens + output_row_splits.back());
    }

    std::vector<int64> output_tokens_shape;
    output_tokens_shape.push_back(tokens.size());

    std::vector<int64> output_row_splits_shape;
    output_row_splits_shape.push_back(output_row_splits.size());

    Tensor* output_values;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("output_values",
                                             TensorShape(output_tokens_shape),
                                             &output_values));
    auto output_values_vec = output_values->vec<tstring>();

    Tensor* output_row_splits_tensor;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("output_row_splits",
                                        TensorShape(output_row_splits_shape),
                                        &output_row_splits_tensor));
    auto output_row_splits_vec = output_row_splits_tensor->vec<int64>();

    Tensor* start_values;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("start_values",
                                             TensorShape(output_tokens_shape),
                                             &start_values));
    auto start_values_vec = start_values->vec<int64>();

    Tensor* limit_values;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("limit_values",
                                             TensorShape(output_tokens_shape),
                                             &limit_values));
    auto limit_values_vec = limit_values->vec<int64>();

    for (int i = 0; i < tokens.size(); ++i) {
      output_values_vec(i) = tokens[i];
    }

    for (int i = 0; i < output_row_splits.size(); ++i) {
      output_row_splits_vec(i) = output_row_splits[i];
    }

    for (int i = 0; i < begin_offset.size(); ++i) {
      start_values_vec(i) = begin_offset[i];
    }

    for (int i = 0; i < end_offset.size(); ++i) {
      limit_values_vec(i) = end_offset[i];
    }
  }

 private:
  bool force_split_at_break_character_;

  TF_DISALLOW_COPY_AND_ASSIGN(SplitMergeTokenizeWithOffsetsOp);
};

REGISTER_KERNEL_BUILDER(
    Name("SplitMergeTokenizeWithOffsets").Device(DEVICE_CPU),
    SplitMergeTokenizeWithOffsetsOp);

}  // namespace text
}  // namespace tensorflow
