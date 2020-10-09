/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/string_ops.cc.

#include <string>

#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {
namespace {
// Split input string `str` based on a character delimiter.
// Returns a vector of StringPieces which are valid as long as input `str`
// is valid.
// Note: The single character delimiter is a common case and is implemented as
// a series of finds in the input string, making it much more efficient than
// SplitOnCharSet.
template <typename Predicate>
std::vector<StringPiece> SplitOnChar(const tstring& str, const char delim,
                                     Predicate p) {
  std::vector<StringPiece> result;
  StringPiece text(str);
  auto f = text.find(delim);
  while (f != StringPiece::npos) {
    StringPiece token = text.substr(0, f);
    if (p(token)) {
      result.emplace_back(token);
    }
    text.remove_prefix(f + 1);
    f = text.find(delim);
  }
  if (p(text)) {
    result.push_back(text);
  }
  return result;
}

// Split input string `str` based on a set of character delimiters.
// Returns a vector of StringPieces which are valid as long as input `str`
// is valid.
// Based on str_util::Split.
template <typename Predicate>
std::vector<StringPiece> SplitOnCharSet(const tstring& str,
                                        const tstring& delim_set, Predicate p) {
  std::vector<StringPiece> result;
  StringPiece text(str);
  StringPiece delims(delim_set);
  size_t token_start = 0;
  for (size_t i = 0; i < text.size() + 1; i++) {
    if ((i == text.size()) || (delims.find(text[i]) != StringPiece::npos)) {
      StringPiece token(text.data() + token_start, i - token_start);
      if (p(token)) {
        result.emplace_back(token);
      }
      token_start = i + 1;
    }
  }
  return result;
}

// Split input string `str` based on given delimiter.
// Returns a vector of StringPieces which are valid as long as input `str`
// is valid.
template <typename Predicate>
std::vector<StringPiece> Split(const tstring& str, const tstring& delimiter,
                               Predicate predicate) {
  if (str.empty()) {
    return std::vector<StringPiece>();
  }
  if (delimiter.empty()) {
    std::vector<StringPiece> result;
    result.resize(str.size());
    for (size_t i = 0; i < str.size(); ++i) {
      result[i] = StringPiece(str.data() + i, 1);
    }
    return result;
  }
  if (delimiter.size() == 1) {
    return SplitOnChar(str, delimiter[0], predicate);
  }
  return SplitOnCharSet(str, delimiter, predicate);
}

std::vector<StringPiece> SplitV2(const tstring& str, StringPiece sep,
                                 int maxsplit) {
  // This SplitV2 method matches the behavior of python's str.split:
  //   If sep is given, consecutive delimiters are not grouped together
  //   and are deemed to delimit empty strings (for example, '1,,2'.split(',')
  //   returns ['1', '', '2']). The sep argument may consist of multiple
  //   characters (for example, '1<>2<>3'.split('<>') returns ['1', '2', '3']).
  //   Splitting an empty string with a specified separator returns [''].
  //
  //   If sep is not specified or is None, a different splitting algorithm is
  //   applied: runs of consecutive whitespace are regarded as a single
  //   separator, and the result will contain no empty strings at the start or
  //   end if the string has leading or trailing whitespace. Consequently,
  //   splitting an empty string or a string consisting of just whitespace
  //   with a None separator returns [].

  std::vector<StringPiece> result;

  StringPiece text(str);
  if (maxsplit == 0) {
    result.emplace_back(text);
    return result;
  }

  if (sep.empty()) {
    StringPiece token;
    // Remove leading whitespaces.
    str_util::RemoveLeadingWhitespace(&text);
    int split = 0;
    while (str_util::ConsumeNonWhitespace(&text, &token)) {
      result.push_back(token);
      str_util::RemoveLeadingWhitespace(&text);
      ++split;
      if (maxsplit > 0 && split == maxsplit) {
        result.push_back(text);
        return result;
      }
    }
    return result;
  }
  auto p = std::search(text.begin(), text.end(), sep.begin(), sep.end());
  int split = 0;
  while (p != text.end()) {
    StringPiece token = text.substr(0, p - text.begin());
    result.push_back(token);
    text.remove_prefix(token.size());
    text.remove_prefix(sep.size());
    ++split;
    if (maxsplit > 0 && split == maxsplit) {
      result.push_back(StringPiece(text));
      return result;
    }
    p = std::search(text.begin(), text.end(), sep.begin(), sep.end());
  }
  result.push_back(text);
  return result;
}

}  // namespace

class StringSplitOp : public OpKernel {
 public:
  explicit StringSplitOp(OpKernelConstruction* context)
      : OpKernel(context), skip_empty_(true) {
    bool skip_empty;
    // By default skip_empty_ is true. We only get the value from attr if it is
    // available, so that it is backward compatible.
    if (context->GetAttr("skip_empty", &skip_empty).ok()) {
      skip_empty_ = skip_empty;
    }
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("input", &input_tensor));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(input_tensor->shape()),
                errors::InvalidArgument("input must be a vector, got shape: ",
                                        input_tensor->shape().DebugString()));

    const auto input_vec = input_tensor->vec<tstring>();
    const int64 batch_size = input_vec.dimension(0);

    const Tensor* delimiter_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("delimiter", &delimiter_tensor));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(delimiter_tensor->shape()),
        errors::InvalidArgument("delimiter must be a scalar, got shape: ",
                                delimiter_tensor->shape().DebugString()));
    const auto delimiter_vec = delimiter_tensor->flat<tstring>();
    const tstring& delimiter = delimiter_vec(0);
    // Empty delimiter means split the input character by character.
    std::vector<StringPiece> tokens;
    // Guess that we'll be unpacking a handful of tokens per example.
    static constexpr int kReserveSize = 4;
    tokens.reserve(batch_size * kReserveSize);

    int64 output_size = 0;
    int64 max_num_entries = 0;
    std::vector<int64> num_indices(batch_size);
    for (int64 i = 0; i < batch_size; ++i) {
      std::vector<StringPiece> parts =
          skip_empty_ ? Split(input_vec(i), delimiter, str_util::SkipEmpty())
                      : Split(input_vec(i), delimiter, str_util::AllowEmpty());
      int64 n_entries = parts.size();
      num_indices[i] = n_entries;
      output_size += n_entries;
      max_num_entries = std::max(max_num_entries, n_entries);
      tokens.insert(tokens.end(), std::make_move_iterator(parts.begin()),
                    std::make_move_iterator(parts.end()));
    }

    Tensor* sp_indices_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({output_size, 2}),
                                             &sp_indices_t));
    Tensor* sp_tokens_t;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(1, TensorShape({output_size}), &sp_tokens_t));
    Tensor* sp_shape_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, TensorShape({2}), &sp_shape_t));

    auto sp_indices = sp_indices_t->matrix<int64>();
    auto sp_tokens = sp_tokens_t->vec<tstring>();
    auto sp_shape = sp_shape_t->vec<int64>();
    sp_shape(0) = batch_size;
    sp_shape(1) = max_num_entries;
    size_t c = 0;
    for (size_t i = 0; i < batch_size; ++i) {
      for (size_t j = 0; j < num_indices[i]; ++j) {
        sp_indices(c, 0) = i;
        sp_indices(c, 1) = j;
        sp_tokens(c).assign(tokens[c].data(), tokens[c].size());
        ++c;
      }
    }
  }

 private:
  bool skip_empty_;
};

class StringSplitV2Op : public OpKernel {
 public:
  explicit StringSplitV2Op(OpKernelConstruction* context)
      : OpKernel(context), maxsplit_(-1) {
    OP_REQUIRES_OK(context, context->GetAttr("maxsplit", &maxsplit_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("input", &input_tensor));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(input_tensor->shape()),
                errors::InvalidArgument("input must be a vector, got shape: ",
                                        input_tensor->shape().DebugString()));

    const auto input_vec = input_tensor->vec<tstring>();
    const int64 batch_size = input_vec.dimension(0);

    const Tensor* sep_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("sep", &sep_tensor));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(sep_tensor->shape()),
                errors::InvalidArgument("sep must be a scalar, got shape: ",
                                        sep_tensor->shape().DebugString()));
    const auto sep_vec = sep_tensor->flat<tstring>();
    StringPiece sep(sep_vec(0));
    std::vector<StringPiece> tokens;
    // Guess that we'll be unpacking a handful of tokens per example.
    static constexpr int kReserveSize = 4;
    tokens.reserve(batch_size * kReserveSize);

    int64 output_size = 0;
    int64 max_num_entries = 0;
    std::vector<int64> num_indices(batch_size);
    for (int64 i = 0; i < batch_size; ++i) {
      std::vector<StringPiece> parts = SplitV2(input_vec(i), sep, maxsplit_);
      int64 n_entries = parts.size();
      num_indices[i] = n_entries;
      output_size += n_entries;
      max_num_entries = std::max(max_num_entries, n_entries);
      tokens.insert(tokens.end(), parts.begin(), parts.end());
    }

    Tensor* sp_indices_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({output_size, 2}),
                                             &sp_indices_t));
    Tensor* sp_tokens_t;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(1, TensorShape({output_size}), &sp_tokens_t));
    Tensor* sp_shape_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, TensorShape({2}), &sp_shape_t));

    auto sp_indices = sp_indices_t->matrix<int64>();
    auto sp_tokens = sp_tokens_t->vec<tstring>();
    auto sp_shape = sp_shape_t->vec<int64>();
    sp_shape(0) = batch_size;
    sp_shape(1) = max_num_entries;
    size_t c = 0;
    for (size_t i = 0; i < batch_size; ++i) {
      for (size_t j = 0; j < num_indices[i]; ++j) {
        sp_indices(c, 0) = i;
        sp_indices(c, 1) = j;
        sp_tokens(c).assign(tokens[c].data(), tokens[c].size());
        ++c;
      }
    }
  }

 private:
  int maxsplit_;
};

REGISTER_KERNEL_BUILDER(Name("StringSplit").Device(DEVICE_CPU), StringSplitOp);
REGISTER_KERNEL_BUILDER(Name("StringSplitV2").Device(DEVICE_CPU),
                        StringSplitV2Op);

}  // namespace tensorflow
