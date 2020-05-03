/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include <stdint.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "unicode/appendable.h"  // from @icu
#include "unicode/schriter.h"  // from @icu
#include "unicode/uchar.h"  // from @icu
#include "unicode/ucnv.h"  // from @icu
#include "unicode/ucnv_err.h"  // from @icu
#include "unicode/umachine.h"  // from @icu
#include "unicode/uniset.h"  // from @icu
#include "unicode/unistr.h"  // from @icu
#include "unicode/uset.h"  // from @icu
#include "unicode/utf.h"  // from @icu
#include "unicode/utypes.h"  // from @icu
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/string_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/bcast.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {
namespace {

void Encode(const UnicodeEncoding encoding, const icu::UnicodeString& in,
            tstring* out) {
  if (encoding == UnicodeEncoding::UTF8) {
    out->clear();
    in.toUTF8String(*out);
  } else if (encoding == UnicodeEncoding::UTF16BE) {
    // TODO(gbillock): consider using the
    // extract(char *dest, int32_t destCapacity, UConverter *cnv)
    // for UTF16/32
    out->clear();  // subtle: must come before reserve()
    out->reserve(2 * in.length() + 1);
    const char16_t* buf = in.getBuffer();
    for (int i = 0; i < in.length(); ++i) {
      // Emit big-endian encoding for UTF-16 always.
      out->push_back((buf[i] & 0xFF00) >> 8);
      out->push_back(buf[i] & 0x00FF);
    }
  } else if (encoding == UnicodeEncoding::UTF32BE) {
    out->clear();  // subtle: must come before reserve()
    out->reserve(4 * in.countChar32() + 1);
    icu::StringCharacterIterator it(in);
    UChar32 ch;
    while (it.hasNext()) {
      ch = it.next32PostInc();
      out->push_back((ch & 0xFF000000) >> 24);
      out->push_back((ch & 0x00FF0000) >> 16);
      out->push_back((ch & 0x0000FF00) >> 8);
      out->push_back((ch & 0x000000FF));
    }
  }
}

// This error callback is only useful for finding illegal encoding errors when
// we want to be strict -- otherwise illegal encodings are replaced on read
// with 0xFFFD and signaled to the callback.
void unicode_error_callback(const void* context, UConverterToUnicodeArgs* args,
                            const char* codeUnits, int32_t length,
                            UConverterCallbackReason reason,
                            UErrorCode* pErrorCode) {
  // Careful: this depends on setting up the context settings when the
  // callback is registered.
  bool* format_error = const_cast<bool*>(static_cast<const bool*>(context));

  if (reason == UCNV_UNASSIGNED || reason == UCNV_ILLEGAL ||
      reason == UCNV_IRREGULAR) {
    *format_error = true;
  }

  // Side note: the default behavior in this case is that without a substitution
  // made by the callback, the UConverter will signal an error to the iterator
  // making the string iteration bail out. Instead, forward to the built-in
  // substitution handler.
  UCNV_TO_U_CALLBACK_SUBSTITUTE(nullptr, args, codeUnits, length, reason,
                                pErrorCode);
}

// Iterates through a source string given the provided input UConverter specific
// to the encoding for that string. Calls a provided callback for each codepoint
// consumed. Provides the callback with the codepoint and the number of bytes
// consumed from the input string to produce it. If there are invalid encoding
// loci in the source string, they will be provided as a 0xFFFD codepoint to
// the callback, unless the "fail_on_formatting_error" arg is set, in which
// case the callback will be passed the signal that there is such an invalid
// encoding position.
// callback: function(UChar32 codepoint, int num_bytes_consumed_from_source_str,
//                    bool fatal_format_error)
void IterateUnicodeString(const string& str, UConverter* converter,
                          std::function<void(UChar32, int, bool)> callback) {
  const char* source = str.data();
  const char* limit = str.data() + str.length();
  UErrorCode status = U_ZERO_ERROR;

  UConverterToUCallback oldAction = nullptr;
  const void* oldContext = nullptr;
  bool format_error = false;

  // Subtle. You can't make a function pointer from a std::function. :-(
  // Instead, we pass the boolean pointer as the "context" object.
  ucnv_setToUCallBack(converter, unicode_error_callback, &format_error,
                      &oldAction, &oldContext, &status);
  if (U_FAILURE(status)) {
    LOG(ERROR) << "Could not set unicode error callback on converter";
    return;
  }

  while (source < limit) {
    const char* source_pre_fetch = source;
    // Note: ucnv_getNextUChar returns 0xFFFD on an encoding error.
    UChar32 next_char = ucnv_getNextUChar(converter, &source, limit, &status);
    if (U_FAILURE(status)) {
      source = limit;
    }
    int bytes_consumed = source - source_pre_fetch;
    callback(next_char, bytes_consumed, format_error);
    format_error = false;
  }

  ucnv_setToUCallBack(converter, oldAction, oldContext, nullptr, nullptr,
                      &status);
}

// Lifecycle wrapper for UConverter making it easier to use with thread_local.
// TODO(gbillock): Consider whether to use the higher-level convert API and
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

Status GetErrorOptions(OpKernelConstruction* ctx, ErrorOptions* out) {
  *out = ErrorOptions();

  string error_policy;
  TF_RETURN_IF_ERROR(ctx->GetAttr("errors", &error_policy));

  if (error_policy == "replace") {
    out->elide_replacement = false;
  } else if (error_policy == "ignore") {
    out->elide_replacement = true;
  } else if (error_policy == "strict") {
    out->error_on_malformatting = true;
  } else {
    return errors::InvalidArgument(
        "errors policy must be one of 'strict', 'replace', or 'ignore'");
  }

  int32 replacement_char;
  TF_RETURN_IF_ERROR(ctx->GetAttr("replacement_char", &replacement_char));

  if (replacement_char >= UCHAR_MIN_VALUE &&
      replacement_char <= UCHAR_MAX_VALUE) {
    out->subst = replacement_char;
  } else {
    return errors::InvalidArgument(
        "replacement_char out of unicode codepoint range");
  }

  if (ctx->HasAttr("replace_control_characters")) {
    TF_RETURN_IF_ERROR(ctx->GetAttr("replace_control_characters",
                                    &(out->replace_control_chars)));
  }

  return Status::OK();
}

inline bool ShouldHandleFormatError(const ErrorOptions& error_options,
                                    UChar32 ch, bool format_error) {
  return ((error_options.replace_control_chars && ch <= 0x1F) || format_error);
}

}  // namespace

class UnicodeTranscodeOp : public OpKernel {
 public:
  explicit UnicodeTranscodeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, GetErrorOptions(ctx, &error_options_));

    string output_encoding;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_encoding", &output_encoding));
    OP_REQUIRES_OK(ctx,
                   ParseUnicodeEncoding(output_encoding, &output_encoding_));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("input_encoding", &input_encoding_));
    // Make a temporary UConverter to ensure it will create without error
    // at execution time (and to warm any data caches the converter needs).
    // This instance is not used.
    std::unique_ptr<WrappedConverter> input_encoder =
        absl::make_unique<WrappedConverter>();
    input_encoder->init(input_encoding_);
    OP_REQUIRES(ctx, input_encoder->converter_,
                errors::InvalidArgument(
                    "Could not create converter for input encoding: " +
                    input_encoding_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("input", &input_tensor));

    static thread_local std::unique_ptr<WrappedConverter> input_encoder;
    if (!input_encoder) {
      input_encoder.reset(new WrappedConverter());
    }
    input_encoder->init(input_encoding_);
    OP_REQUIRES(ctx, input_encoder->converter_,
                errors::InvalidArgument(
                    "Could not create converter for input encoding: " +
                    input_encoding_));

    // Output may be forwardable from input, in which case work in-place.
    Tensor* output_tensor;
    std::unique_ptr<Tensor> maybe_forwarded =
        ctx->forward_input(0 /*input_index*/, 0 /*output_index*/,
                           tensorflow::DT_STRING, input_tensor->shape(),
                           ctx->input_memory_type(0), ctx->input_alloc_attr(0));
    if (maybe_forwarded) {
      output_tensor = maybe_forwarded.get();
      OP_REQUIRES_OK(ctx, ctx->set_output("output", *output_tensor));
    } else {
      OP_REQUIRES_OK(ctx, ctx->allocate_output("output", input_tensor->shape(),
                                               &output_tensor));
      output_tensor->flat<tstring>() = input_tensor->flat<tstring>();
    }

    auto output_flat = output_tensor->flat<tstring>();
    bool found_any_format_error = false;
    for (size_t i = 0; i < output_flat.size(); ++i) {
      Transcode(&(output_flat(i)), input_encoder->converter_,
                &found_any_format_error);
    }
    if (error_options_.error_on_malformatting && found_any_format_error) {
      ctx->CtxFailure(
          errors::InvalidArgument("Invalid formatting on input string"));
    }
  }

 private:
  // Consume a codepoint from the input string and add it to the buffer.
  // This function takes care of any replacement configuration on invalid or
  // out-of-range inputs.
  void TranslateCodepoints(icu::UnicodeString* s, bool* found_any_format_error,
                           UChar32 ch, int src_bytes, bool format_error) {
    if (ShouldHandleFormatError(error_options_, ch, format_error)) {
      *found_any_format_error = true;
      if (error_options_.elide_replacement) {
        return;
      } else {
        ch = error_options_.subst;
      }
    }
    s->append(ch);
  }

  // Transcode the string from input encoding to the output_encoding_. If
  // non-valid characters are encountered, use the subst_/elide_replacement_
  // config to handle them.
  void Transcode(tstring* s, UConverter* input_encoder,
                 bool* found_any_format_error) {
    icu::UnicodeString source;
    IterateUnicodeString(
        *s, input_encoder,
        std::bind(&UnicodeTranscodeOp::TranslateCodepoints, this, &source,
                  found_any_format_error, std::placeholders::_1,
                  std::placeholders::_2, std::placeholders::_3));

    Encode(output_encoding_, source, s);
  }

  string input_encoding_;
  ErrorOptions error_options_;
  UnicodeEncoding output_encoding_ = UnicodeEncoding::UTF8;
};

REGISTER_KERNEL_BUILDER(Name("UnicodeTranscode").Device(DEVICE_CPU),
                        UnicodeTranscodeOp);

template <typename SPLITS_TYPE>
class UnicodeDecodeBaseOp : public OpKernel {
 public:
  explicit UnicodeDecodeBaseOp(OpKernelConstruction* ctx, bool generate_offsets)
      : OpKernel(ctx), generate_offsets_(generate_offsets) {
    OP_REQUIRES_OK(ctx, GetErrorOptions(ctx, &error_options_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("input_encoding", &input_encoding_));
    // Make a temporary UConverter to ensure it will create without error
    // at execution time (and to warm any data caches the converter needs).
    // This instance is not used.
    std::unique_ptr<WrappedConverter> input_encoder =
        absl::make_unique<WrappedConverter>();
    input_encoder->init(input_encoding_);
    OP_REQUIRES(ctx, input_encoder->converter_,
                errors::InvalidArgument(
                    "Could not create converter for input encoding: " +
                    input_encoding_));
  }

  void Decode(OpKernelContext* ctx, std::vector<UChar32>* char_values,
              std::vector<SPLITS_TYPE>* offset_values, int* current_offset,
              SPLITS_TYPE* next_row_split, UChar32 char_value, int char_length,
              bool found_any_format_error) {
    if (error_options_.error_on_malformatting && found_any_format_error) {
      ctx->CtxFailure(
          errors::InvalidArgument("Invalid formatting on input string"));
    }
    UChar32 decoded_value = char_value;
    if (ShouldHandleFormatError(error_options_, char_value,
                                found_any_format_error)) {
      if (error_options_.elide_replacement && (offset_values != nullptr)) {
        *current_offset += char_length;
        return;
      } else {
        decoded_value = error_options_.subst;
      }
    }

    // Emit the char value.
    char_values->push_back(decoded_value);

    // Emit the byte offset
    if (offset_values != nullptr) {
      offset_values->push_back(*current_offset);
      *current_offset += char_length;
    }
    *next_row_split += 1;
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("input", &input_tensor));

    // Go through all the strings in `input`.
    const auto& input_vec = input_tensor->flat<tstring>();

    std::unique_ptr<WrappedConverter> input_encoder =
        absl::make_unique<WrappedConverter>();
    input_encoder->init(input_encoding_);
    OP_REQUIRES(ctx, input_encoder->converter_,
                errors::InvalidArgument(
                    "Could not create converter for input encoding: " +
                    input_encoding_));

    std::vector<UChar32> char_values;
    std::vector<SPLITS_TYPE> offset_values;

    Tensor* output_row_splits;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("row_splits",
                                             {input_tensor->NumElements() + 1},
                                             &output_row_splits));
    auto out_row_splits = output_row_splits->vec<SPLITS_TYPE>();

    int row_split_index = 0;
    SPLITS_TYPE next_row_split = 0;
    for (int i = 0; i < input_vec.size(); ++i) {
      const string& input = input_vec(i);
      // Convert input strings into unicode values. Output to a list of
      // char_values, record row splits and char_to_byte_starts, which are all
      // the fields needed to construct a RaggedTensor.
      out_row_splits(row_split_index) = next_row_split;
      row_split_index++;
      int current_offset = 0;
      IterateUnicodeString(
          input, input_encoder->converter_,
          std::bind(&UnicodeDecodeBaseOp::Decode, this, ctx, &char_values,
                    &offset_values, &current_offset, &next_row_split,
                    std::placeholders::_1, std::placeholders::_2,
                    std::placeholders::_3));
    }
    out_row_splits(row_split_index) = next_row_split;

    Tensor* output_char_values;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(
                 "char_values", {static_cast<SPLITS_TYPE>(char_values.size())},
                 &output_char_values));
    auto out_char_values = output_char_values->vec<int32>();
    if (generate_offsets_) {
      DCHECK(offset_values.size() == char_values.size());
      Tensor* output_offset_values;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(
                              "char_to_byte_starts",
                              {static_cast<SPLITS_TYPE>(offset_values.size())},
                              &output_offset_values));
      auto out_offset_values = output_offset_values->vec<SPLITS_TYPE>();

      // Load output tensors from intermediate value arrays.
      for (int i = 0; i < char_values.size(); ++i) {
        out_char_values(i) = static_cast<int32>(char_values[i]);
        out_offset_values(i) = offset_values[i];
      }
    } else {
      for (int i = 0; i < char_values.size(); ++i) {
        out_char_values(i) = static_cast<int32>(char_values[i]);
      }
    }
  }

 private:
  string input_encoding_;
  ErrorOptions error_options_;
  bool generate_offsets_ = false;
};

template <typename SPLITS_TYPE>
class UnicodeDecodeOp : public UnicodeDecodeBaseOp<SPLITS_TYPE> {
 public:
  explicit UnicodeDecodeOp(OpKernelConstruction* ctx)
      : UnicodeDecodeBaseOp<SPLITS_TYPE>(ctx, false) {}
};

template <typename SPLITS_TYPE>
class UnicodeDecodeWithOffsetsOp : public UnicodeDecodeBaseOp<SPLITS_TYPE> {
 public:
  explicit UnicodeDecodeWithOffsetsOp(OpKernelConstruction* ctx)
      : UnicodeDecodeBaseOp<SPLITS_TYPE>(ctx, true) {}
};

REGISTER_KERNEL_BUILDER(
    Name("UnicodeDecode").Device(DEVICE_CPU).TypeConstraint<int64>("Tsplits"),
    UnicodeDecodeOp<int64>);
REGISTER_KERNEL_BUILDER(Name("UnicodeDecodeWithOffsets")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int64>("Tsplits"),
                        UnicodeDecodeWithOffsetsOp<int64>);
REGISTER_KERNEL_BUILDER(
    Name("UnicodeDecode").Device(DEVICE_CPU).TypeConstraint<int32>("Tsplits"),
    UnicodeDecodeOp<int32>);
REGISTER_KERNEL_BUILDER(Name("UnicodeDecodeWithOffsets")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int32>("Tsplits"),
                        UnicodeDecodeWithOffsetsOp<int32>);

template <typename SPLITS_TYPE>
class UnicodeEncodeOp : public OpKernel {
 public:
  explicit UnicodeEncodeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    string encoding_tmp;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_encoding", &encoding_tmp));
    OP_REQUIRES_OK(ctx, ParseUnicodeEncoding(encoding_tmp, &encoding_));
    OP_REQUIRES_OK(ctx, GetErrorOptions(ctx, &error_options_));
  }

  /**
   * Encodes Unicode codepoints into the desired string representation.
   *
   * We lose a dimension while encoding, since a series of integer codepoints is
   * encoded into a single string.
   *
   * This accepts two input tensors: a rank 1 tensor of code point values and
   * a single rank 1 tensor of splits which determine where each string begins
   * and ends from the provided code points.
   */
  void Compute(OpKernelContext* context) override {
    // Get inputs
    const Tensor& input_tensor = context->input(0);
    const auto input_tensor_flat = input_tensor.flat<int32>();
    const Tensor& input_splits = context->input(1);
    const auto input_splits_flat = input_splits.flat<SPLITS_TYPE>();

    // Since we limit to a 2-D input (flat_values of rank 1 and a single splits
    // tensor), our output dimension will be 1 with it's size equal to the
    // number of splits (outer dimension or ragged tensor).
    TensorShape output_shape({input_splits.dim_size(0) - 1});
    Tensor* output_tensor;
    OP_REQUIRES_OK(context, context->allocate_output("output", output_shape,
                                                     &output_tensor));
    auto output_tensor_flat = output_tensor->flat<tstring>();

    // Use a single index over the flattened input values tensor.
    int idx = 0;
    // Loop through our split dimension to create a new string at each split.
    for (int i = 1; i < input_splits_flat.size(); ++i) {
      icu::UnicodeString unicode_string;
      icu::UnicodeStringAppendable appendable_unicode_string(unicode_string);
      for (; idx < input_splits_flat(i); ++idx) {
        int32 code_point = input_tensor_flat(idx);
        // Check for invalid code point
        if (!U_IS_UNICODE_CHAR(code_point)) {
          if (error_options_.error_on_malformatting) {
            context->CtxFailure(errors::InvalidArgument(
                "Code point is out of range for Unicode, or a noncharacter."));
            return;
          } else if (!error_options_.elide_replacement) {
            code_point = error_options_.subst;
          }
        }
        appendable_unicode_string.appendCodePoint(code_point);
      }
      // Encode our string and save in the output.
      tstring result;
      Encode(encoding_, unicode_string, &result);
      output_tensor_flat(i - 1) = std::move(result);
    }
  }

 private:
  UnicodeEncoding encoding_;
  ErrorOptions error_options_;
};

REGISTER_KERNEL_BUILDER(
    Name("UnicodeEncode").Device(DEVICE_CPU).TypeConstraint<int64>("Tsplits"),
    UnicodeEncodeOp<int64>);
REGISTER_KERNEL_BUILDER(
    Name("UnicodeEncode").Device(DEVICE_CPU).TypeConstraint<int32>("Tsplits"),
    UnicodeEncodeOp<int32>);

}  // namespace tensorflow
