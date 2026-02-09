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

#include <memory>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "icu4c/source/common/unicode/uchar.h"
#include "icu4c/source/common/unicode/ucnv_err.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow_text/core/kernels/sentence_breaking_utils.h"
#include "tensorflow_text/core/kernels/sentence_fragmenter.h"

using ::tensorflow::tstring;
using ::tensorflow::errors::InvalidArgument;

namespace tensorflow {
namespace text {

// TODO(thuang513): This is copied from unicode_ops.cc, move this to a separate
//               util lib in tensorflow and reuse it here instead.
namespace {
// Lifecycle wrapper for UConverter making it easier to use with thread_local.
// TODO(gregbillock): Consider whether to use the higher-level convert API and
// create a specialized fast code path for UTF8.
class WrappedConverter {
 public:
  WrappedConverter() {}

  ~WrappedConverter() {
    if (converter_) {
      ucnv_close(converter_);
    }
  }

  void init(const string& name) {
    if (converter_ && name == name_) {
      // Note: this reset is not typically needed, but if not done, then in some
      // cases the cached converter will maintain state of input endianness
      // which isn't valid from input to input in every batched case.
      ucnv_reset(converter_);
      return;
    }

    if (converter_) {
      ucnv_close(converter_);
      converter_ = nullptr;
      name_ = "";
    }

    UErrorCode status = U_ZERO_ERROR;
    converter_ = ucnv_open(name.c_str(), &status);
    if (U_FAILURE(status)) {
      if (converter_) {
        ucnv_close(converter_);
        converter_ = nullptr;
      }
    } else {
      name_ = name;
    }
  }

  UConverter* converter_ = nullptr;
  string name_;
};

struct ErrorOptions {
  UChar32 subst = 0xFFFD;
  bool elide_replacement = false;
  bool replace_control_chars = false;
  bool error_on_malformatting = false;
};

absl::Status GetErrorOptions(OpKernelConstruction* context, ErrorOptions* out) {
  *out = ErrorOptions();

  string error_policy;
  TF_RETURN_IF_ERROR(context->GetAttr("errors", &error_policy));

  if (error_policy == "replace") {
    out->elide_replacement = false;
  } else if (error_policy == "ignore") {
    out->elide_replacement = true;
  } else if (error_policy == "strict") {
    out->error_on_malformatting = true;
  } else {
    return InvalidArgument(
        "errors policy must be one of 'strict', 'replace', or 'ignore'");
  }

  int32 replacement_char;
  TF_RETURN_IF_ERROR(context->GetAttr("replacement_char", &replacement_char));

  if (replacement_char >= UCHAR_MIN_VALUE &&
      replacement_char <= UCHAR_MAX_VALUE) {
    out->subst = replacement_char;
  } else {
    return InvalidArgument("replacement_char out of unicode codepoint range");
  }

  if (context->HasAttr("replace_control_characters")) {
    TF_RETURN_IF_ERROR(context->GetAttr("replace_control_characters",
                                        &(out->replace_control_chars)));
  }

  return absl::OkStatus();
}

inline bool ShouldHandleFormatError(const ErrorOptions& error_options,
                                    UChar32 ch, bool format_error) {
  return ((error_options.replace_control_chars && ch <= 0x1F) || format_error);
}

}  // namespace

class SentenceFragmentsOp : public OpKernel {
 public:
  explicit SentenceFragmentsOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, GetErrorOptions(context, &error_options_));

    OP_REQUIRES_OK(context,
                   context->GetAttr("input_encoding", &input_encoding_));
    // Make a temporary UConverter to ensure it will create without error
    // at execution time (and to warm any data caches the converter needs).
    // This instance is not used.
    std::unique_ptr<WrappedConverter> input_encoder =
        std::make_unique<WrappedConverter>();
    input_encoder->init(input_encoding_);
    OP_REQUIRES(
        context, input_encoder->converter_,
        InvalidArgument("Could not create converter for input encoding: " +
                        input_encoding_));
  }

  void Compute(::tensorflow::OpKernelContext* context) override {
#define DECLARE_AND_VALIDATE_INPUT_VECTOR(name, dtype)                        \
  const Tensor* name##_tensor;                                                \
  OP_REQUIRES_OK(context, context->input(#name, &name##_tensor));             \
  OP_REQUIRES(context, TensorShapeUtils::IsVector(name##_tensor->shape()),    \
              InvalidArgument(                                                \
                  absl::StrCat("'", #name, "' must be a vector, got shape: ", \
                               name##_tensor->shape().DebugString())));       \
  const auto& name = name##_tensor->vec<dtype>();

    DECLARE_AND_VALIDATE_INPUT_VECTOR(row_lengths, int64);
    DECLARE_AND_VALIDATE_INPUT_VECTOR(token_start, int64);
    DECLARE_AND_VALIDATE_INPUT_VECTOR(token_end, int64);
    DECLARE_AND_VALIDATE_INPUT_VECTOR(token_word, tstring);
    DECLARE_AND_VALIDATE_INPUT_VECTOR(token_properties, int64);

#undef DECLARE_AND_VALIDATE_INPUT_TENSOR

    static thread_local std::unique_ptr<WrappedConverter> input_encoder;
    if (!input_encoder) {
      input_encoder = std::make_unique<WrappedConverter>();
    }
    input_encoder->init(input_encoding_);
    OP_REQUIRES(
        context, input_encoder->converter_,
        InvalidArgument("Could not create converter for input encoding: " +
                        input_encoding_));

    UConverter* converter = input_encoder->converter_;
    UnicodeUtil util(converter);

    int num_elements = 0;
    for (int i = 0; i < row_lengths.size(); ++i) {
      num_elements += row_lengths(i);
    }
    OP_REQUIRES(context,
                num_elements == token_start.size() &&
                    token_start.size() == token_end.size() &&
                    token_end.size() == token_word.size(),
                InvalidArgument(absl::StrCat(
                    "num_elements(", num_elements, "), token_start(",
                    token_start.size(), "), token_end(", token_end.size(),
                    "), token_word(", token_word.size(),
                    ") must all be the same size.")));

    // Iterate through the text
    int token_index = 0;
    int num_fragments = 0;
    std::vector<std::vector<SentenceFragment>> fragments;
    for (int i = 0; i < row_lengths.size(); ++i) {
      std::vector<Token> tokens;
      Document doc(&tokens);
      for (int j = 0; j < row_lengths(i); ++j) {
        doc.AddToken(
            token_word(token_index), token_start(token_index),
            token_end(token_index), Token::SPACE_BREAK,
            static_cast<Token::TextProperty>(token_properties(token_index)));
        ++token_index;
      }

      // Find fragments.
      SentenceFragmenter fragmenter(&doc, &util);
      std::vector<SentenceFragment> frags;
      OP_REQUIRES_OK(context, fragmenter.FindFragments(&frags));

      num_fragments += frags.size();
      fragments.push_back(std::move(frags));
    }

    std::vector<int64> fragment_shape;
    fragment_shape.push_back(num_fragments);

    std::vector<int64> doc_batch_shape;
    doc_batch_shape.push_back(fragments.size());

#define DECLARE_OUTPUT_TENSOR(name, out_shape)                                 \
  Tensor* name##_tensor = nullptr;                                             \
  OP_REQUIRES_OK(context, context->allocate_output(                            \
                              #name, TensorShape(out_shape), &name##_tensor)); \
  auto name = name##_tensor->vec<int64>();

    DECLARE_OUTPUT_TENSOR(fragment_start, fragment_shape);
    DECLARE_OUTPUT_TENSOR(fragment_end, fragment_shape);
    DECLARE_OUTPUT_TENSOR(fragment_properties, fragment_shape);
    DECLARE_OUTPUT_TENSOR(terminal_punc_token, fragment_shape);
    DECLARE_OUTPUT_TENSOR(output_row_lengths, doc_batch_shape);

#undef DECLARE_OUTPUT_TENSOR

    // output_row_splits should have shape of
    // [number of fragments over the entire batch]
    int element_index = 0;
    // Iterate through all the documents
    for (int i = 0; i < fragments.size(); ++i) {
      const std::vector<SentenceFragment>& fragments_in_doc = fragments[i];
      // Iterate through all the fragments of a document
      for (int j = 0; j < fragments_in_doc.size(); ++j) {
        const SentenceFragment& fragment = fragments_in_doc[j];
        fragment_start(element_index) = fragment.start;
        fragment_end(element_index) = fragment.limit;
        fragment_properties(element_index) = fragment.properties;
        terminal_punc_token(element_index) = fragment.terminal_punc_token;
        ++element_index;
      }
      output_row_lengths(i) = fragments_in_doc.size();
    }
  }

 private:
  string input_encoding_;
  ErrorOptions error_options_;
};

REGISTER_KERNEL_BUILDER(Name("SentenceFragments").Device(DEVICE_CPU),
                        SentenceFragmentsOp);

}  // namespace text
}  // namespace tensorflow
