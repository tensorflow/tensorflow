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

// Tokenizes text, the input string #(batch_index).  Knowing the batch_index
// allows us to retrieve the corresponding data from logits.  I.e., the logits
// for the i-th character from text are logits(batch_index, i, 0) (for the
// "split" action) and logits(batch_index, i, 1) (for the "merge" action).
Status TokenizeByLogits(const absl::string_view& text,
                        const TTypes<const float, 3>::Tensor& logits,
                        int batch_index,
                        bool force_split_at_break_character,
                        std::vector<std::string>* tokens,
                        std::vector<int>* begin_offset,
                        std::vector<int>* end_offset, int* num_tokens) {
  std::vector<absl::string_view> chars;
  if (!GetUTF8Chars(text, &chars)) {
    return Status(
        static_cast<absl::StatusCode>(absl::StatusCode::kInvalidArgument),
        absl::StrCat("Input string is not utf8 valid: ", text));
  }

  if (chars.size() > logits.dimension(1)) {
    return Status(
        static_cast<absl::StatusCode>(absl::StatusCode::kInvalidArgument),
        absl::StrCat("Number of logits, ", logits.dimension(1),
                     ", is insufficient for text \"", text, "\""));
  }

  bool last_character_is_break_character = false;
  int start = 0;
  bool has_new_token_generated_for_text = false;
  for (int i = 0; i < chars.size(); ++i) {
    const bool is_break_character = IsBreakChar(chars[i]);
    if (!is_break_character) {
      const float logit_split = logits(batch_index, i, 0);
      const float logit_merge = logits(batch_index, i, 1);
      if ((logit_split > logit_merge) ||
          !has_new_token_generated_for_text ||
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

class TokenizerFromLogitsOp : public OpKernel {
 public:
  explicit TokenizerFromLogitsOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor* strings;
    OP_REQUIRES_OK(ctx, ctx->input("strings", &strings));
    const Tensor* logits;
    OP_REQUIRES_OK(ctx, ctx->input("logits", &logits));
    OP_REQUIRES(ctx, strings->dim_size(0) == logits->dim_size(0),
                errors::InvalidArgument("Expecting logits to have ",
                                        strings->dim_size(0),
                                        " rows, got ",
                                        logits->dim_size(0)));
    const Tensor* force_split_at_break_character;
    OP_REQUIRES_OK(ctx, ctx->input("force_split_at_break_character",
                                   &force_split_at_break_character));
    const bool force_split_at_break_character_bool =
        force_split_at_break_character->scalar<bool>()();

    std::vector<string> tokens;
    std::vector<int> begin_offset;
    std::vector<int> end_offset;
    std::vector<int> output_row_splits(1, 0);

    // Tensor to access values from logits.
    const TTypes<const float, 3>::Tensor logits_tensor =
        logits->tensor<float, 3>();

    // Iterate through all the values and tokenize them.
    const auto& strings_vec = strings->flat<tstring>();
    OP_REQUIRES(ctx, logits_tensor.dimension(0) >= strings_vec.size(),
                errors::Internal("Bad logits dimension #0: ",
                                 logits_tensor.dimension(0), " < ",
                                 strings_vec.size()));
    // Dimension #1 of logits will be checked inside TokenizeByLogits.
    OP_REQUIRES(ctx, logits_tensor.dimension(2) == 2,
                errors::Internal("Bad logits dimension #2: ",
                                 logits_tensor.dimension(2), " != 2"));
    for (int i = 0; i < strings_vec.size(); ++i) {
      // Tokenize into tokens and record the offset locations.
      int num_tokens = 0;
      OP_REQUIRES_OK(
          ctx, TokenizeByLogits(
                   strings_vec(i),
                   logits_tensor, i,
                   force_split_at_break_character_bool,
                   &tokens, &begin_offset, &end_offset, &num_tokens));

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
                   ctx->allocate_output("row_splits",
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
  TF_DISALLOW_COPY_AND_ASSIGN(TokenizerFromLogitsOp);
};

REGISTER_KERNEL_BUILDER(
    Name("TokenizerFromLogits").Device(DEVICE_CPU),
    TokenizerFromLogitsOp);

}  // namespace text
}  // namespace tensorflow
