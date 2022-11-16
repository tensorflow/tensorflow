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

#ifndef TENSORFLOW_CORE_KERNELS_SHAPE_OPS_H_
#define TENSORFLOW_CORE_KERNELS_SHAPE_OPS_H_

#include <limits>
#include <unordered_set>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/variant_op_registry.h"

namespace tensorflow {

namespace shape_op_helpers {
inline Status GetShape(OpKernelContext* ctx, int input_index,
                       TensorShape* shape) {
  *shape = ctx->input(input_index).shape();
  return OkStatus();
}
}  // namespace shape_op_helpers

template <typename OutType>
class ShapeOp : public OpKernel {
 public:
  explicit ShapeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    TensorShape shape;
    OP_REQUIRES_OK(ctx, shape_op_helpers::GetShape(ctx, 0, &shape));
    const int rank = shape.dims();
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({rank}), &out));
    auto vec = out->vec<OutType>();
    for (int i = 0; i < rank; ++i) {
      int64_t dim_size = shape.dim_size(i);
      if (out->dtype() == DT_INT32) {
        OP_REQUIRES(
            ctx, FastBoundsCheck(dim_size, std::numeric_limits<int32>::max()),
            errors::InvalidArgument("Shape output type is 32-bit ", " but dim ",
                                    i, " is ", dim_size));
      }
      vec(i) = static_cast<OutType>(dim_size);
    }
  }

  bool IsExpensive() override { return false; }
};

template <typename OutType>
class ShapeNOp : public OpKernel {
 public:
  explicit ShapeNOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      TensorShape shape;
      OP_REQUIRES_OK(ctx, shape_op_helpers::GetShape(ctx, i, &shape));
      const int dims = shape.dims();
      Tensor* out = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(i, {dims}, &out));
      auto vec = out->vec<OutType>();

      for (int j = 0; j < dims; ++j) {
        int64_t dim_size = shape.dim_size(j);
        if (out->dtype() == DT_INT32) {
          OP_REQUIRES(
              ctx, FastBoundsCheck(dim_size, std::numeric_limits<int32>::max()),
              errors::InvalidArgument("ShapeN output type is 32-bit but shape ",
                                      i, " dim ", j, " is ", dim_size));
        }
        vec(j) = static_cast<OutType>(dim_size);
      }
    }
  }

  bool IsExpensive() override { return false; }
};

class RankOp : public OpKernel {
 public:
  explicit RankOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    TensorShape shape;
    OP_REQUIRES_OK(ctx, shape_op_helpers::GetShape(ctx, 0, &shape));
    const int rank = shape.dims();
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &out));
    out->scalar<int32>()() = rank;
  }

  bool IsExpensive() override { return false; }
};

template <typename OutType>
class SizeOp : public OpKernel {
 public:
  explicit SizeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    TensorShape shape;
    OP_REQUIRES_OK(ctx, shape_op_helpers::GetShape(ctx, 0, &shape));
    const int64_t size = shape.num_elements();
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &out));
    if (out->dtype() == DT_INT32) {
      OP_REQUIRES(
          ctx, FastBoundsCheck(size, std::numeric_limits<int32>::max()),
          errors::InvalidArgument("Number of elements was larger than "
                                  "representable by 32-bit output type"));
    }
    out->scalar<OutType>()() = static_cast<OutType>(size);
  }

  bool IsExpensive() override { return false; }
};

template <typename Tdim>
class ExpandDimsOp : public OpKernel {
 public:
  explicit ExpandDimsOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input_t = ctx->input(0);
    OP_REQUIRES(ctx, input_t.dtype() != DT_VARIANT,
                errors::InvalidArgument("ExpandDims on Variant not supported"));

    const Tensor& dim_t = ctx->input(1);
    OP_REQUIRES(
        ctx, (dim_t.NumElements() == 1),
        errors::InvalidArgument("'dim' must be a tensor with a single value"));
    DCHECK_EQ(dim_t.dtype(), DataTypeToEnum<Tdim>::v());
    Tdim dim = *static_cast<const Tdim*>(DMAHelper::base(&dim_t));
    const TensorShape& input_shape = input_t.shape();
    int input_dims = input_shape.dims();
    OP_REQUIRES(ctx, dim >= -1 - input_dims && dim <= input_dims,
                errors::InvalidArgument("Tried to expand dim index ", dim,
                                        " for tensor with ", input_dims,
                                        " dimensions."));

    // We emulate numpy's interpretation of the dim axis when
    // -input.dims() >= dim <= input.dims().
    if (dim < 0) {
      // Clamp to the end if needed.
      dim = std::min<Tdim>(dim + input_dims + 1, input_dims);
    }

    // Compute new shape with an additional dimension.
    absl::InlinedVector<int64_t, 8> output_shape_vec(input_dims + 1);
    for (int64_t i = 0; i < dim; ++i) {
      output_shape_vec[i] = input_shape.dim_size(i);
    }
    output_shape_vec[dim] = 1;
    for (int64_t i = dim + 1; i < input_dims + 1; ++i) {
      output_shape_vec[i] = input_shape.dim_size(i - 1);
    }
    TensorShape output_shape(output_shape_vec);

    Tensor output_t;
    if (!output_t.CopyFrom(input_t, output_shape)) {
      // This should never happen, since the sizes of the input and output
      // should always be the same (we only expand the dimension with 1).
      ctx->SetStatus(
          errors::Internal("Could not expand dimension with input shape ",
                           ctx->input(0).shape().DebugString(),
                           " and output shape ", output_shape.DebugString()));
    }
    ctx->set_output(0, std::move(output_t));
  }

  bool IsExpensive() override { return false; }
};

class SqueezeOp : public OpKernel {
 public:
  explicit SqueezeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    std::vector<int32> squeeze_dims;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("squeeze_dims", &squeeze_dims));
    squeeze_dims_.insert(squeeze_dims.begin(), squeeze_dims.end());
  }

  void Compute(OpKernelContext* ctx) override {
    OP_REQUIRES(ctx, ctx->input(0).dtype() != DT_VARIANT,
                errors::InvalidArgument("Squeeze on Variant not supported"));

    auto existing_dims = ctx->input(0).shape().dim_sizes();
    const int existing_dims_size = static_cast<int>(existing_dims.size());
    std::vector<int64_t> new_shape;

    std::unordered_set<int32> wrapped_squeeze_dims;
    wrapped_squeeze_dims.reserve(squeeze_dims_.size());
    // Validate squeeze dims against the input.
    for (int32_t dim : squeeze_dims_) {
      OP_REQUIRES(
          ctx, (dim >= -ctx->input(0).dims() && dim < ctx->input(0).dims()),
          errors::InvalidArgument("Tried to squeeze dim index ", dim,
                                  " for tensor with ", ctx->input(0).dims(),
                                  " dimensions."));
      // If dim is < 0, we wrap around (-1 means the last element).
      if (dim < 0) {
        dim = existing_dims_size + dim;
      }

      wrapped_squeeze_dims.insert(dim);
    }

    for (int i = 0; i < existing_dims_size; ++i) {
      auto existing_dim = existing_dims[i];

      // If squeeze_set is non-empty, only squeeze those dimensions.
      if (!wrapped_squeeze_dims.empty()) {
        if (wrapped_squeeze_dims.count(i) > 0) {
          OP_REQUIRES(ctx, existing_dim == 1,
                      errors::InvalidArgument(
                          "Can not squeeze dim[", i,
                          "], expected a dimension of 1, got ", existing_dim));
        } else {
          // This dimension is not being squeezed.
          new_shape.push_back(existing_dim);
        }
      } else {
        // Copy over all non-1-length dimensions.
        if (existing_dim != 1) {
          new_shape.push_back(existing_dim);
        }
      }
    }

    const TensorShape output_shape(new_shape);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {0}, &output));
    if (!output->CopyFrom(ctx->input(0), output_shape)) {
      // This should never happen, since the sizes of the input and
      // output should always be the same.
      ctx->SetStatus(errors::Internal("Could not squeeze input with shape ",
                                      ctx->input(0).shape().DebugString(),
                                      " and output shape ",
                                      output_shape.DebugString()));
    }
  }

  bool IsExpensive() override { return false; }

 private:
  std::unordered_set<int32> squeeze_dims_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SHAPE_OPS_H_
