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

#ifdef INTEL_MKL
#ifdef INTEL_MKL_DNN

#include "tensorflow/core/kernels/shape_ops.h"
#include "tensorflow/core/framework/register_types.h"

#include "tensorflow/core/util/mkl_util.h"

namespace tensorflow {

namespace mkl_shape_op_helper {
template <typename OutType>
// Common function to handle input at 'input_index' and set its shape at
// 'output_index' in context 'ctx'. Used by both Shape and ShapeN.
inline void SetShapeForInput(OpKernelContext* ctx, int input_index,
                             int output_index) {
  MklDnnShape mkl_in_shape;
  GetMklShape(ctx, input_index, &mkl_in_shape);

  TensorShape shape;
  if (mkl_in_shape.IsMklTensor()) {
    shape = mkl_in_shape.GetTfShape();
  } else {
    OP_REQUIRES_OK(ctx, shape_op_helpers::GetRegularOrVariantShape(ctx,
                                                  input_index, &shape));
  }

  // Snippet taken from shape_ops.h It is common irrespective of input (Mkl or
  // Tensorflow layout).
  const int rank = shape.dims();

  // Allocate output tensor.
  Tensor* out = nullptr;
  TensorShape tf_out_shape = TensorShape({rank});
  MklDnnShape mkl_out_shape;
  // Output of Shape or ShapeN is never in MKL layout.
  mkl_out_shape.SetMklTensor(false);

  AllocateOutputSetMklShape(ctx, output_index, &out, tf_out_shape,
                            mkl_out_shape);
  OP_REQUIRES_OK(ctx, ctx->status());

  // Populate output tensor.
  auto vec = out->vec<OutType>();
  for (int i = 0; i < rank; ++i) {
    int64 dim_size = shape.dim_size(i);
    if (out->dtype() == DT_INT32) {
      OP_REQUIRES(
          ctx, FastBoundsCheck(dim_size, std::numeric_limits<int32>::max()),
          errors::InvalidArgument("Shape output type is 32-bit ", " but dim ",
                                  i, " is ", dim_size));
    }
    vec(i) = static_cast<OutType>(dim_size);
  }
  ctx->SetStatus(Status::OK());
}
}  // namespace mkl_shape_op_helper

template <typename OutType>
class MklShapeOp : public OpKernel {
 public:
  explicit MklShapeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const int kInputIndex = 0;
    const int kOutputIndex = 0;
    mkl_shape_op_helper::SetShapeForInput<OutType>(ctx, kInputIndex,
                                                   kOutputIndex);
    OP_REQUIRES_OK(ctx, ctx->status());
  }

  bool IsExpensive() override { return false; }
};

template <typename OutType>
class MklShapeNOp : public OpKernel {
 public:
  explicit MklShapeNOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("N", &N_));
  }

  void Compute(OpKernelContext* ctx) override {
    for (int i = 0; i < N_; ++i) {
      // For ShapeN op, output index is same as input index.
      mkl_shape_op_helper::SetShapeForInput<OutType>(ctx, i, i);
      OP_REQUIRES_OK(ctx, ctx->status());
    }
  }

  bool IsExpensive() override { return false; }

 private:
  int N_;
};

// Currently, we are adding TypeConstraint on T to be float because for MKLDNN
// all ops support float type only. But we can eliminate it once we support
// other types.
// Shape ----------------------------------------
#define REGISTER_MKL_CPU_KERNEL(T)                                    \
  REGISTER_KERNEL_BUILDER(Name("_MklShape")                           \
                              .Device(DEVICE_CPU)                     \
                              .HostMemory("output")                   \
                              .TypeConstraint<int32>("out_type")      \
                              .TypeConstraint<T>("T")                 \
                              .Label(mkl_op_registry::kMklOpLabel),   \
                          MklShapeOp<int32>);                         \
  REGISTER_KERNEL_BUILDER(Name("_MklShape")                           \
                              .Device(DEVICE_CPU)                     \
                              .HostMemory("output")                   \
                              .TypeConstraint<int64>("out_type")      \
                              .TypeConstraint<T>("T")                 \
                              .Label(mkl_op_registry::kMklOpLabel),   \
                          MklShapeOp<int64>);

TF_CALL_float(REGISTER_MKL_CPU_KERNEL)
#undef REGISTER_MKL_CPU_KERNEL

// ShapeN ---------------------------------------

#define REGISTER_MKL_CPU_KERNEL(T)                                    \
  REGISTER_KERNEL_BUILDER(Name("_MklShapeN")                          \
                              .Device(DEVICE_CPU)                     \
                              .HostMemory("output")                   \
                              .TypeConstraint<int32>("out_type")      \
                              .TypeConstraint<T>("T")                 \
                              .Label(mkl_op_registry::kMklOpLabel),   \
                          MklShapeNOp<int32>);                        \
  REGISTER_KERNEL_BUILDER(Name("_MklShapeN")                          \
                              .Device(DEVICE_CPU)                     \
                              .HostMemory("output")                   \
                              .TypeConstraint<int64>("out_type")      \
                              .TypeConstraint<T>("T")                 \
                              .Label(mkl_op_registry::kMklOpLabel),   \
                          MklShapeNOp<int64>);

TF_CALL_float(REGISTER_MKL_CPU_KERNEL)
#undef REGISTER_MKL_CPU_KERNEL
}  // namespace tensorflow

#endif  // INTEL_MKL_DNN
#endif  // INTEL_MKL
