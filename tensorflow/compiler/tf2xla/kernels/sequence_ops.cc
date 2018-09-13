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

// XLA-specific sequence and range Ops.

#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace {

template <typename T>
Status GetValue(int index, XlaOpKernelContext* ctx, T* value) {
  xla::Literal literal;
  TF_RETURN_IF_ERROR(ctx->ConstantInput(index, &literal));
  *value = literal.Get<T>({});
  return Status::OK();
}

Status GetIntValue(int index, XlaOpKernelContext* ctx, int64* value) {
  xla::Literal literal;
  TF_RETURN_IF_ERROR(ctx->ConstantInput(index, &literal));
  switch (literal.shape().element_type()) {
    case xla::S32:
      *value = literal.Get<int32>({});
      break;
    case xla::S64:
      *value = literal.Get<int64>({});
      break;
    default:
      return errors::InvalidArgument("Invalid argument type for argument",
                                     index);
  }
  return Status::OK();
}

// The type-specific part of the implementation of Range.
template <typename T>
Status CreateRangeTensor(const xla::LiteralSlice& start_literal,
                         const xla::LiteralSlice& limit_literal,
                         const xla::LiteralSlice& delta_literal,
                         Tensor* output) {
  T start = start_literal.Get<T>({});
  T limit = limit_literal.Get<T>({});
  T delta = delta_literal.Get<T>({});

  if (delta == 0) {
    return errors::InvalidArgument("Requires delta != 0: ", delta);
  }
  if (delta > 0) {
    if (start > limit) {
      return errors::InvalidArgument(
          "Requires start <= limit when delta > 0: ", start, "/", limit);
    }
  } else {
    if (start < limit) {
      return errors::InvalidArgument(
          "Requires start >= limit when delta < 0: ", start, "/", limit);
    }
  }
  int64 size =
      (std::is_integral<T>::value
           ? ((std::abs(limit - start) + std::abs(delta) - 1) / std::abs(delta))
           : std::ceil(std::abs((limit - start) / delta)));

  *output = Tensor(DataTypeToEnum<T>::v(), TensorShape({size}));
  auto flat = output->flat<T>();
  T val = start;
  for (int64 i = 0; i < size; ++i) {
    flat(i) = val;
    val += delta;
  }
  return Status::OK();
}

class RangeOp : public XlaOpKernel {
 public:
  explicit RangeOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape start_in_shape = ctx->InputShape(0);
    const TensorShape limit_in_shape = ctx->InputShape(1);
    const TensorShape delta_in_shape = ctx->InputShape(2);
    OP_REQUIRES(ctx, IsLegacyScalar(start_in_shape),
                errors::InvalidArgument("start must be a scalar, not shape ",
                                        start_in_shape.DebugString()));
    OP_REQUIRES(ctx, IsLegacyScalar(limit_in_shape),
                errors::InvalidArgument("limit must be a scalar, not shape ",
                                        limit_in_shape.DebugString()));
    OP_REQUIRES(ctx, IsLegacyScalar(delta_in_shape),
                errors::InvalidArgument("delta must be a scalar, not shape ",
                                        delta_in_shape.DebugString()));
    xla::Literal start, limit, delta;
    OP_REQUIRES_OK(ctx, ctx->ConstantInput(0, &start));
    OP_REQUIRES_OK(ctx, ctx->ConstantInput(1, &limit));
    OP_REQUIRES_OK(ctx, ctx->ConstantInput(2, &delta));

    DataType type = input_type(0);
    Tensor output;
    Status status;
    switch (type) {
      case DT_INT32:
        status = CreateRangeTensor<int32>(start, limit, delta, &output);
        break;
      case DT_INT64:
        status = CreateRangeTensor<int64>(start, limit, delta, &output);
        break;
      case DT_FLOAT:
        status = CreateRangeTensor<float>(start, limit, delta, &output);
        break;
      case DT_DOUBLE:
        status = CreateRangeTensor<double>(start, limit, delta, &output);
        break;
      default:
        status = errors::InvalidArgument("Invalid type for Range ",
                                         DataTypeString(type));
    }
    OP_REQUIRES_OK(ctx, status);
    ctx->SetConstantOutput(0, output);
  }
};

REGISTER_XLA_OP(Name("Range")
                    .CompileTimeConstInput("start")
                    .CompileTimeConstInput("limit")
                    .CompileTimeConstInput("delta"),
                RangeOp);

class LinSpaceOp : public XlaOpKernel {
 public:
  explicit LinSpaceOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape start_in_shape = ctx->InputShape(0);
    const TensorShape stop_in_shape = ctx->InputShape(1);
    const TensorShape num_in_shape = ctx->InputShape(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(start_in_shape),
                errors::InvalidArgument("start must be a scalar, not shape ",
                                        start_in_shape.DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(stop_in_shape),
                errors::InvalidArgument("stop must be a scalar, not shape ",
                                        stop_in_shape.DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(num_in_shape),
                errors::InvalidArgument("num must be a scalar, not shape ",
                                        num_in_shape.DebugString()));

    DataType type = ctx->input_type(0);

    int64 num;
    OP_REQUIRES_OK(ctx, GetIntValue(2, ctx, &num));
    OP_REQUIRES(ctx, num > 0,
                errors::InvalidArgument("Requires num > 0: ", num));
    Tensor out_constant(type, TensorShape({num}));

    switch (type) {
      case DT_FLOAT: {
        float start, stop;
        OP_REQUIRES_OK(ctx, GetValue(0, ctx, &start));
        OP_REQUIRES_OK(ctx, GetValue(1, ctx, &stop));
        auto flat = out_constant.flat<float>();
        if (num == 1) {
          flat(0) = start;
        } else {
          const float step = (stop - start) / (num - 1);
          for (int64 i = 0; i < num; ++i) {
            flat(i) = start + step * i;
          }
        }
        break;
      }
      case DT_DOUBLE: {
        double start, stop;
        OP_REQUIRES_OK(ctx, GetValue(0, ctx, &start));
        OP_REQUIRES_OK(ctx, GetValue(1, ctx, &stop));
        auto flat = out_constant.flat<double>();
        if (num == 1) {
          flat(0) = start;
        } else {
          const double step = (stop - start) / (num - 1);
          for (int64 i = 0; i < num; ++i) {
            flat(i) = start + step * i;
          }
        }
        break;
      }

      default:
        ctx->SetStatus(errors::InvalidArgument("Invalid argument type ",
                                               DataTypeString(type)));
        return;
    }
    ctx->SetConstantOutput(0, out_constant);
  }
};

REGISTER_XLA_OP(Name("LinSpace")
                    .CompileTimeConstInput("start")
                    .CompileTimeConstInput("stop")
                    .CompileTimeConstInput("num"),
                LinSpaceOp);

}  // namespace
}  // namespace tensorflow
