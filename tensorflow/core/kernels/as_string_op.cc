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

#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/bfloat16.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/tstring.h"

namespace tensorflow {

class AsStringOp : public OpKernel {
 public:
  using OpKernel::OpKernel;

  explicit AsStringOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    bool scientific;
    bool shortest;
    std::string fill_string;
    DataType dtype;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("precision", &precision_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("scientific", &scientific));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shortest", &shortest));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("width", &width_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("fill", &fill_string));
    if (dtype != DT_STRING && !DataTypeIsFloating(dtype) &&
        !DataTypeIsComplex(dtype)) {
      OP_REQUIRES(ctx, !(scientific || shortest),
                  absl::InvalidArgumentError(
                      absl::StrCat("scientific and shortest format "
                                   "not supported for datatype ",
                                   DataTypeString(dtype))));
      OP_REQUIRES(
          ctx, precision_ < 0,
          absl::InvalidArgumentError(absl::StrCat("precision not supported "
                                                  "for datatype ",
                                                  DataTypeString(dtype))));
    }
    OP_REQUIRES(
        ctx, fill_string.size() <= 1,
        errors::InvalidArgument("Fill string must be one or fewer characters"));
    OP_REQUIRES(ctx, !(scientific && shortest),
                errors::InvalidArgument(
                    "Cannot select both scientific and shortest notation"));

    if (!fill_string.empty()) {
      switch (fill_string[0]) {
        case ' ':
        case '+':
        case '-':
        case '0':
        case '#':
          break;
        default:
          bool fill_not_supported = true;
          OP_REQUIRES(ctx, !fill_not_supported,
                      errors::InvalidArgument("Fill argument not supported: \"",
                                              fill_string, "\""));
      }
    }
    if (width_ <= -1) {
      width_ = 0;
    }
    // If input is string and width unspecified, simply forward to output.
    if (dtype == DT_STRING && width_ <= 0) {
      return;
    }
    char format_char;
    if (dtype == DT_STRING) {
      format_char = 's';
    } else if (DataTypeIsUnsigned(dtype)) {
      format_char = 'u';
    } else if (DataTypeIsSigned(dtype)) {
      format_char = 'd';
    } else if (DataTypeIsFloating(dtype) || DataTypeIsComplex(dtype)) {
      if (shortest) {
        format_char = 'g';
      } else if (scientific) {
        format_char = 'e';
      } else {
        format_char = 'f';
      }
    } else if (dtype == DT_BOOL) {
      return;
    } else if (dtype == DT_VARIANT) {
      return;
    } else {
      bool type_not_supported = true;
      OP_REQUIRES(ctx, !type_not_supported,
                  absl::InvalidArgumentError(absl::StrCat(
                      "Type not supported: ", DataTypeString(dtype))));
    }
    format_ = absl::StrCat("%", fill_string, "*.*",
                           absl::string_view(&format_char, 1));
    if (format_char == 's') {
      string_format_ = StringFormat::New(format_);
      OP_REQUIRES(ctx, string_format_ != nullptr,
                  absl::InvalidArgumentError(
                      absl::StrCat("Invalid format: ", format_)));
    } else if (format_char == 'u' || format_char == 'd') {
      integral_format_ = IntegralFormat::New(format_);
      OP_REQUIRES(ctx, integral_format_ != nullptr,
                  absl::InvalidArgumentError(
                      absl::StrCat("Invalid format: ", format_)));
    } else {
      floating_format_ = FloatingFormat::New(format_);
      OP_REQUIRES(ctx, floating_format_ != nullptr,
                  absl::InvalidArgumentError(
                      absl::StrCat("Invalid format: ", format_)));
    }
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));
    const DataType& dtype = input_tensor->dtype();

    // If input is string and width unspecified, simply forward to output.
    if (dtype == DT_STRING && width_ <= 0) {
      context->set_output(0, context->input(0));
      return;
    }

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output("output", input_tensor->shape(),
                                            &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

    if (dtype == DT_BOOL) {
      const auto& input_flat = input_tensor->flat<bool>();
      for (int i = 0; i < input_flat.size(); ++i) {
        output_flat(i) = (input_flat(i)) ? "true" : "false";
      }
      return;
    }

    if (dtype == DT_VARIANT) {
      const auto& input_flat = input_tensor->flat<Variant>();
      for (int i = 0; i < input_flat.size(); ++i) {
        output_flat(i) = input_flat(i).DebugString();
      }
      return;
    }

    // All other cases use the format string.

#define ENCODE_TYPE(type, T, enc_fmt)                                   \
  case (type): {                                                        \
    const auto& input_flat = input_tensor->flat<T>();                   \
    for (int i = 0; i < input_flat.size(); ++i) {                       \
      output_flat(i) =                                                  \
          absl::StrFormat(*enc_fmt, width_, precision_, input_flat(i)); \
    }                                                                   \
  } break

    switch (dtype) {
      ENCODE_TYPE(DT_UINT8, uint8_t, integral_format_);
      ENCODE_TYPE(DT_UINT16, uint16_t, integral_format_);
      ENCODE_TYPE(DT_UINT32, uint32_t, integral_format_);
      ENCODE_TYPE(DT_UINT64, uint64_t, integral_format_);
      ENCODE_TYPE(DT_INT8, int8_t, integral_format_);
      ENCODE_TYPE(DT_INT16, int16_t, integral_format_);
      ENCODE_TYPE(DT_INT32, int32_t, integral_format_);
      ENCODE_TYPE(DT_INT64, int64_t, integral_format_);
      ENCODE_TYPE(DT_FLOAT, float, floating_format_);
      ENCODE_TYPE(DT_DOUBLE, double, floating_format_);
      case (DT_STRING): {
        const auto& input_flat = input_tensor->flat<tstring>();
        for (int i = 0; i < input_flat.size(); ++i) {
          output_flat(i) = absl::StrFormat(*string_format_, width_, precision_,
                                           absl::string_view(input_flat(i)));
        }
      } break;
      case (DT_HALF): {
        const auto& input_flat = input_tensor->flat<Eigen::half>();
        for (int i = 0; i < input_flat.size(); ++i) {
          output_flat(i) =
              absl::StrFormat(*floating_format_, width_, precision_,
                              static_cast<float>(input_flat(i)));
        }
      } break;
      case (DT_BFLOAT16): {
        const auto& input_flat = input_tensor->flat<bfloat16>();
        for (int i = 0; i < input_flat.size(); ++i) {
          output_flat(i) =
              absl::StrFormat(*floating_format_, width_, precision_,
                              static_cast<float>(input_flat(i)));
        }
      } break;
      case (DT_COMPLEX64): {
        const auto& input_flat = input_tensor->flat<complex64>();
        for (int i = 0; i < input_flat.size(); ++i) {
          output_flat(i) =
              absl::StrCat("(",
                           absl::StrFormat(*floating_format_, width_,
                                           precision_, input_flat(i).real()),
                           ",",
                           absl::StrFormat(*floating_format_, width_,
                                           precision_, input_flat(i).imag()),
                           ")");
        }
      } break;
      case (DT_COMPLEX128): {
        const auto& input_flat = input_tensor->flat<complex128>();
        for (int i = 0; i < input_flat.size(); ++i) {
          output_flat(i) =
              absl::StrCat("(",
                           absl::StrFormat(*floating_format_, width_,
                                           precision_, input_flat(i).real()),
                           ",",
                           absl::StrFormat(*floating_format_, width_,
                                           precision_, input_flat(i).imag()),
                           ")");
        }
      } break;
      default:
        bool can_encode_type = false;
        OP_REQUIRES(context, can_encode_type,
                    errors::InvalidArgument("Cannot encode input of type ",
                                            DataTypeString(dtype)));
    }

#undef ENCODE_TYPE
  }

 private:
  // Used to parse "%*.*g", etc.
  using FloatingFormat =
      absl::ParsedFormat<absl::FormatConversionCharSet::kStar,
                         absl::FormatConversionCharSet::kStar,
                         absl::FormatConversionCharSet::g |
                             absl::FormatConversionCharSet::e |
                             absl::FormatConversionCharSet::f>;

  // Used to parse "%*.*u", etc.
  using IntegralFormat =
      absl::ParsedFormat<absl::FormatConversionCharSet::kStar,
                         absl::FormatConversionCharSet::kStar,
                         absl::FormatConversionCharSet::u |
                             absl::FormatConversionCharSet::d>;

  // Used to parse "%*.*s", etc.
  using StringFormat = absl::ParsedFormat<absl::FormatConversionCharSet::kStar,
                                          absl::FormatConversionCharSet::kStar,
                                          absl::FormatConversionCharSet::s>;

  int precision_ = -1;
  int width_ = -1;
  decltype(StringFormat::New("%*.*s")) string_format_;
  decltype(IntegralFormat::New("%*.*u")) integral_format_;
  decltype(FloatingFormat::New("%*.*g")) floating_format_;
  std::string format_;
};

REGISTER_KERNEL_BUILDER(Name("AsString").Device(DEVICE_CPU), AsStringOp);

}  // namespace tensorflow
