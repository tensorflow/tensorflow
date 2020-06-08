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
#include "tensorflow/core/lib/strings/stringprintf.h"

namespace tensorflow {

class AsStringOp : public OpKernel {
 public:
  using OpKernel::OpKernel;

  explicit AsStringOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    int32 precision;
    bool scientific;
    bool shortest;
    int32 width;
    string fill_string;
    DataType dtype;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("precision", &precision));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("scientific", &scientific));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shortest", &shortest));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("width", &width));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("fill", &fill_string));
    switch (dtype) {
      case DT_FLOAT:
      case DT_DOUBLE:
      case DT_COMPLEX64:
      case DT_COMPLEX128:
        break;
      default:
        OP_REQUIRES(ctx, !(scientific || shortest),
                    errors::InvalidArgument("scientific and shortest format "
                                            "not supported for datatype ",
                                            DataTypeString(dtype)));
        OP_REQUIRES(ctx, precision < 0,
                    errors::InvalidArgument("precision not supported "
                                            "for datatype ",
                                            DataTypeString(dtype)));
    }
    OP_REQUIRES(
        ctx, fill_string.size() <= 1,
        errors::InvalidArgument("Fill string must be one or fewer characters"));
    OP_REQUIRES(ctx, !(scientific && shortest),
                errors::InvalidArgument(
                    "Cannot select both scientific and shortest notation"));
    format_ = "%";
    if (width > -1) {
      strings::Appendf(&format_, "%s%d", fill_string.c_str(), width);
    }
    if (precision > -1) {
      strings::Appendf(&format_, ".%d", precision);
    }
    switch (dtype) {
      case DT_INT8:
      case DT_INT16:
      case DT_INT32:
        strings::Appendf(&format_, "d");
        break;
      case DT_INT64:
        strings::Appendf(&format_, "lld");
        break;
      case DT_FLOAT:
      case DT_DOUBLE:
      case DT_COMPLEX64:
      case DT_COMPLEX128:
        if (shortest) {
          strings::Appendf(&format_, "g");
        } else if (scientific) {
          strings::Appendf(&format_, "e");
        } else {
          strings::Appendf(&format_, "f");
        }
        break;
      case DT_BOOL:
        break;
      default:
        bool type_not_supported = true;
        OP_REQUIRES(ctx, !type_not_supported,
                    errors::InvalidArgument("Type not supported: ",
                                            DataTypeString(dtype)));
    }

    if (dtype == DT_COMPLEX64 || dtype == DT_COMPLEX128) {
      format_ = strings::Printf("(%s,%s)", format_.c_str(), format_.c_str());
    }
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));
    const DataType& dtype = input_tensor->dtype();

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output("output", input_tensor->shape(),
                                            &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

#define ENCODE_TYPE(type, T, enc_str)                                     \
  case (type): {                                                          \
    const auto& input_flat = input_tensor->flat<T>();                     \
    for (int i = 0; i < input_flat.size(); ++i) {                         \
      output_flat(i) = strings::Printf((enc_str.c_str()), input_flat(i)); \
    }                                                                     \
  } break

    switch (dtype) {
      ENCODE_TYPE(DT_INT32, int32, format_);
      ENCODE_TYPE(DT_INT64, int64, format_);
      ENCODE_TYPE(DT_FLOAT, float, format_);
      ENCODE_TYPE(DT_DOUBLE, double, format_);
      ENCODE_TYPE(DT_INT8, int8, format_);
      ENCODE_TYPE(DT_INT16, int16, format_);
      case (DT_BOOL): {
        const auto& input_flat = input_tensor->flat<bool>();
        for (int i = 0; i < input_flat.size(); ++i) {
          output_flat(i) = (input_flat(i)) ? "true" : "false";
        }
      } break;
      case (DT_COMPLEX64): {
        const auto& input_flat = input_tensor->flat<complex64>();
        for (int i = 0; i < input_flat.size(); ++i) {
          output_flat(i) = strings::Printf(
              format_.c_str(), input_flat(i).real(), input_flat(i).imag());
        }
      } break;
      case (DT_COMPLEX128): {
        const auto& input_flat = input_tensor->flat<complex128>();
        for (int i = 0; i < input_flat.size(); ++i) {
          output_flat(i) = strings::Printf(
              format_.c_str(), input_flat(i).real(), input_flat(i).imag());
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
  string format_;
};

REGISTER_KERNEL_BUILDER(Name("AsString").Device(DEVICE_CPU), AsStringOp);

}  // namespace tensorflow
