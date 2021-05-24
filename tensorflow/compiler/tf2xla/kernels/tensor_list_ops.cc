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

// XLA TensorList operators.

#include <limits>
#include <vector>

#include "tensorflow/compiler/tf2xla/kernels/gather_op_helpers.h"
#include "tensorflow/compiler/tf2xla/kernels/tensor_list_utils.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace {

// GetTensorListDynamicDims collects the dynamic dimensions that a tensorlist
// may carry and returns them in a 2D vector: XlaOp[ElementSize][DimSize]. If a
// dimension is static, a constant dimension is returned. If a dim is dynamic, a
// dynamic XlaOp representing the dynamic size is returned.
StatusOr<std::vector<std::vector<xla::XlaOp>>> GetTensorListDynamicDims(
    XlaOpKernelContext* ctx, const xla::Shape& element_shape,
    const xla::Shape& list_shape, int64 num_elements) {
  std::vector<int64> dynamic_sizes;
  // The multiplier can be a dynamic value.
  TF_RETURN_IF_ERROR(ctx->ConstantInputAsIntVector(0, &dynamic_sizes));
  std::vector<bool> dims_are_dynamic;
  TF_RETURN_IF_ERROR(
      ctx->ResolveInputDynamismIntoPredVector(0, &dims_are_dynamic));
  bool leading_dim_is_dynamic;
  TF_RETURN_IF_ERROR(
      ctx->ResolveInputDynamismIntoPred(1, &leading_dim_is_dynamic));
  std::vector<std::vector<xla::XlaOp>> list_dynamic_dims;
  // Set dynamic dimension size to 0 for initialization value.
  std::vector<xla::XlaOp> dynamic_dims;
  if (leading_dim_is_dynamic) {
    dynamic_dims.push_back(ctx->Input(1));
  } else {
    dynamic_dims.push_back(
        xla::ConstantR0<int32>(ctx->builder(), num_elements));
  }
  for (int64 dim = 0; dim < element_shape.dimensions_size(); ++dim) {
    if (dims_are_dynamic[dim]) {
      auto dynamic_dim_size = xla::Slice(ctx->Input(0), {dim}, {dim + 1}, {1});
      dynamic_dim_size = xla::Reshape(dynamic_dim_size, {});
      dynamic_dim_size = xla::ConvertElementType(dynamic_dim_size, xla::S32);
      dynamic_dims.push_back(dynamic_dim_size);
    } else {
      dynamic_dims.push_back(
          xla::ConstantR0<int32>(ctx->builder(), dynamic_sizes[dim]));
    }
  }
  list_dynamic_dims.push_back(dynamic_dims);
  return list_dynamic_dims;
}

class TensorListLengthOp : public XlaOpKernel {
 public:
  explicit TensorListLengthOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    int64 leading_dim;
    xla::XlaOp leading_dim_size;
    bool leading_dim_is_dynamic;
    OP_REQUIRES_OK(ctx, GetLeadingDimForTensorList(ctx->Input(0), &leading_dim,
                                                   &leading_dim_is_dynamic,
                                                   &leading_dim_size));
    ctx->SetOutput(0, leading_dim_size);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(TensorListLengthOp);
};

REGISTER_XLA_OP(Name("TensorListLength").IsMetadataOp(), TensorListLengthOp);

// "input" is the shape input for EmptyTensorList/TensorListReserve ops.
// If "input" is a compile time constant and not "unknown rank" (-1), return
// its value in "*shape".
Status TryGetElementShapeFromInput(XlaOpKernelContext* ctx, xla::XlaOp input,
                                   xla::PrimitiveType dtype, bool* got_shape,
                                   xla::Shape* shape) {
  auto is_compile_time_constant_or = input.builder()->IsConstant(input);
  TF_RETURN_IF_ERROR(is_compile_time_constant_or.status());

  bool is_compile_time_constant = is_compile_time_constant_or.ValueOrDie();
  if (!is_compile_time_constant) {
    *got_shape = false;
    return Status::OK();
  }

  PartialTensorShape partial_shape;
  TF_RETURN_IF_ERROR(ctx->ConstantInputAsPartialShape(0, &partial_shape));
  if (!partial_shape.IsFullyDefined()) {
    *got_shape = false;
    return Status::OK();
  }

  *shape = xla::ShapeUtil::MakeShape(dtype, partial_shape.dim_sizes());
  *got_shape = true;
  return Status::OK();
}

class TensorListReserveOp : public XlaOpKernel {
 public:
  explicit TensorListReserveOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("element_dtype", &dtype_));
    // Only non-nested TensorList is supported for now.
    OP_REQUIRES(
        ctx, dtype_ != DT_VARIANT,
        errors::Unimplemented(
            "Only non-nested TensorList is supported for TensorListReserve."));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    int64 num_elements;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntScalar(1, &num_elements));
    bool num_element_is_dynamic;
    OP_REQUIRES_OK(
        ctx, ctx->ResolveInputDynamismIntoPred(1, &num_element_is_dynamic));
    OP_REQUIRES(
        ctx, num_elements >= 0,
        errors::InvalidArgument(
            "XLA compilation requires a fixed tensor list size. Set the number "
            "of elements. This could also happen if you're using a TensorArray "
            "in a while loop that does not have its maximum_iteration set, you "
            "can fix this by setting maximum_iteration to a suitable value."));

    // If element shape is compile time constant and it's not "unknown rank"
    // shape (-1), create an initialized TensorList. Otherwise create an
    // uninitialized TensorList.
    xla::XlaOp element_shape_handle = ctx->Input(0);
    xla::PrimitiveType type;
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(dtype_, &type));
    bool got_shape;
    xla::Shape element_shape;
    OP_REQUIRES_OK(ctx,
                   TryGetElementShapeFromInput(ctx, element_shape_handle, type,
                                               &got_shape, &element_shape));
    if (got_shape) {
      xla::Shape list_shape;
      OP_REQUIRES_OK(ctx, GetTensorListShapeFromElementShape(
                              element_shape, num_elements,
                              num_element_is_dynamic, &list_shape));
      // Set up dynamic dimension sizes to create the zero tensor.
      auto list_dynamic_dims_or = GetTensorListDynamicDims(
          ctx, element_shape, list_shape, num_elements);
      OP_REQUIRES_OK(ctx, list_dynamic_dims_or.status());
      xla::XlaOp new_list;
      OP_REQUIRES_OK(ctx, CreateZerosTensorListWithShape(
                              ctx->builder(), list_shape,
                              list_dynamic_dims_or.ValueOrDie(), &new_list));
      xla::XlaOp result;
      OP_REQUIRES_OK(
          ctx,
          SetTensorListPushIndex(
              new_list, xla::ConstantR0<int32>(ctx->builder(), num_elements),
              &result));
      ctx->SetTensorListOutput(0, result);
      return;
    }

    xla::XlaOp result = BuildUninitializedTensorList(
        ctx->builder(), num_elements, num_element_is_dynamic, ctx->Input(1));
    ctx->SetTensorListOutput(0, result);
  }

 private:
  DataType dtype_;

  TF_DISALLOW_COPY_AND_ASSIGN(TensorListReserveOp);
};

REGISTER_XLA_OP(Name("TensorListReserve")
                    .CompileTimeConstantInput("element_shape")
                    .CompileTimeConstantInput("num_elements"),
                TensorListReserveOp);

class EmptyTensorListOp : public XlaOpKernel {
 public:
  explicit EmptyTensorListOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("element_dtype", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    int64 max_num_elements;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntScalar(1, &max_num_elements));
    bool num_element_is_dynamic;
    OP_REQUIRES_OK(
        ctx, ctx->ResolveInputDynamismIntoPred(1, &num_element_is_dynamic));
    OP_REQUIRES(ctx, max_num_elements >= 0,
                errors::InvalidArgument(
                    "XLA compilation requires a fixed tensor list size. Set "
                    "the max number of elements. This could also happen if "
                    "you're using a TensorArray in a while loop that does not "
                    "have its maximum_iteration set, you can fix this by "
                    "setting maximum_iteration to a suitable value."));

    if (dtype_ != DT_VARIANT) {
      // We are creating a non-nested TensorList.
      // If element shape is compile time constant and it's not "unknown
      // rank" shape (-1), create an initialized TensorList. Otherwise
      // create an uninitialized TensorList.
      xla::XlaOp element_shape_handle = ctx->Input(0);
      xla::PrimitiveType type;
      OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(dtype_, &type));
      bool got_shape;
      xla::Shape element_shape;
      OP_REQUIRES_OK(
          ctx, TryGetElementShapeFromInput(ctx, element_shape_handle, type,
                                           &got_shape, &element_shape));
      if (got_shape) {
        xla::Shape list_shape;
        OP_REQUIRES_OK(ctx, GetTensorListShapeFromElementShape(
                                element_shape, max_num_elements,
                                num_element_is_dynamic, &list_shape));
        // Set up dynamic dimension sizes to create the zero tensor.
        auto list_dynamic_dims_or = GetTensorListDynamicDims(
            ctx, element_shape, list_shape, max_num_elements);
        OP_REQUIRES_OK(ctx, list_dynamic_dims_or.status());

        xla::XlaOp result;
        OP_REQUIRES_OK(ctx, CreateZerosTensorListWithShape(
                                ctx->builder(), list_shape,
                                list_dynamic_dims_or.ValueOrDie(), &result));

        ctx->SetTensorListOutput(0, result);
        return;
      }
    }

    // We are creating a nested TensorList or a non-nested TensorList with
    // unknown shape. Just create an uninitialized TensorList.
    xla::XlaOp result =
        BuildUninitializedTensorList(ctx->builder(), max_num_elements,
                                     num_element_is_dynamic, ctx->Input(1));
    ctx->SetTensorListOutput(0, result);
  }

 private:
  DataType dtype_;

  TF_DISALLOW_COPY_AND_ASSIGN(EmptyTensorListOp);
};

REGISTER_XLA_OP(Name("EmptyTensorList")
                    .CompileTimeConstantInput("element_shape")
                    .CompileTimeConstantInput("max_num_elements")
                    .AllowVariantTypes(),
                EmptyTensorListOp);

class TensorListElementShapeOp : public XlaOpKernel {
 public:
  explicit TensorListElementShapeOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shape_type", &shape_type_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    // Check that the TensorList is initialized.
    bool is_initialized;
    OP_REQUIRES_OK(ctx,
                   (IsTensorListInitialized(ctx->Input(0), &is_initialized)));
    OP_REQUIRES(ctx, is_initialized,
                errors::InvalidArgument("TensorList is not initialized"));

    // Only non-nested TensorList is supported for now.
    bool is_nested;
    OP_REQUIRES_OK(ctx, IsNestedTensorList(ctx->Input(0), &is_nested));
    OP_REQUIRES(ctx, !is_nested,
                errors::Unimplemented("Only non-nested TensorList is supported "
                                      "for TensorListElementShape."));

    // For non-nested TensorList, element shape is the buffer shape without
    // the first dimension.
    xla::XlaBuilder* b = ctx->builder();
    xla::Shape list_shape;
    OP_REQUIRES_OK(ctx, GetTensorListBufferShape(ctx->Input(0), &list_shape));
    list_shape.DeleteDimension(0);

    switch (shape_type_) {
      case DT_INT64:
        ctx->SetOutput(0, xla::ConstantR1<int64>(b, list_shape.dimensions()));
        break;
      case DT_INT32: {
        std::vector<int32> size;
        for (int64 s : list_shape.dimensions()) {
          size.push_back(s);
        }
        ctx->SetOutput(0, xla::ConstantR1<int32>(b, size));
        break;
      }
      default:
        ctx->CtxFailure(
            errors::InvalidArgument("Unsupported shape type requested"));
        return;
    }
  }

 private:
  DataType shape_type_;

  TF_DISALLOW_COPY_AND_ASSIGN(TensorListElementShapeOp);
};

REGISTER_XLA_OP(Name("TensorListElementShape").IsMetadataOp(),
                TensorListElementShapeOp);

class TensorListGetItemOp : public XlaOpKernel {
 public:
  explicit TensorListGetItemOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("element_dtype", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    // Check that the TensorList is initialized.
    bool is_initialized;
    OP_REQUIRES_OK(ctx,
                   (IsTensorListInitialized(ctx->Input(0), &is_initialized)));
    OP_REQUIRES(ctx, is_initialized,
                errors::InvalidArgument("TensorList is not initialized"));

    // Only non-nested TensorList is supported for now.
    bool is_nested;
    OP_REQUIRES_OK(ctx, IsNestedTensorList(ctx->Input(0), &is_nested));
    OP_REQUIRES(ctx, !is_nested,
                errors::Unimplemented("Only non-nested TensorList is supported "
                                      "for TensorListGetItem."));

    xla::XlaOp list = ctx->Input(0);
    xla::XlaOp index = ctx->Input(1);

    xla::XlaOp result;
    OP_REQUIRES_OK(ctx, ExecuteTensorListGetItem(list, index, &result));

    ctx->SetOutput(0, result);
  }

 private:
  DataType dtype_;

  TF_DISALLOW_COPY_AND_ASSIGN(TensorListGetItemOp);
};

REGISTER_XLA_OP(Name("TensorListGetItem"), TensorListGetItemOp);

class TensorListGatherOp : public XlaOpKernel {
 public:
  explicit TensorListGatherOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("element_dtype", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    // Check that the TensorList is initialized.
    bool is_initialized;
    OP_REQUIRES_OK(ctx,
                   (IsTensorListInitialized(ctx->Input(0), &is_initialized)));
    OP_REQUIRES(ctx, is_initialized,
                errors::InvalidArgument("TensorList is not initialized"));

    // Only non-nested TensorList is supported for now.
    bool is_nested;
    OP_REQUIRES_OK(ctx, IsNestedTensorList(ctx->Input(0), &is_nested));
    OP_REQUIRES(ctx, !is_nested,
                errors::Unimplemented("Only non-nested TensorList is supported "
                                      "for TensorListGather."));

    DataType indices_type = ctx->input_type(1);

    const TensorShape indices_shape = ctx->InputShape(1);
    OP_REQUIRES(ctx, indices_shape.dims() == 1,
                errors::InvalidArgument("indices must be rank 1"));

    xla::XlaOp list = ctx->Input(0);
    xla::XlaOp indices = ctx->Input(1);

    xla::XlaOp buffer;
    OP_REQUIRES_OK(ctx, GetTensorListBuffer(list, &buffer));
    xla::Shape buffer_xla_shape;
    OP_REQUIRES_OK(ctx, GetTensorListBufferShape(list, &buffer_xla_shape));
    TensorShape buffer_shape;
    OP_REQUIRES_OK(ctx, XLAShapeToTensorShape(buffer_xla_shape, &buffer_shape));

    xla::XlaOp result;
    OP_REQUIRES_OK(
        ctx, XlaGather(buffer, buffer_shape, indices, indices_shape, /*axis=*/0,
                       /*indices_are_nd=*/false, dtype_, indices_type,
                       ctx->builder(), &result));
    ctx->SetOutput(0, result);
  }

 private:
  DataType dtype_;

  TF_DISALLOW_COPY_AND_ASSIGN(TensorListGatherOp);
};

REGISTER_XLA_OP(Name("TensorListGather"), TensorListGatherOp);

class TensorListStackOp : public XlaOpKernel {
 public:
  explicit TensorListStackOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    // Check that the TensorList is initialized.
    bool is_initialized;
    OP_REQUIRES_OK(ctx,
                   (IsTensorListInitialized(ctx->Input(0), &is_initialized)));
    OP_REQUIRES(ctx, is_initialized,
                errors::InvalidArgument("TensorList is not initialized"));

    // Only non-nested TensorList is supported for now.
    bool is_nested;
    OP_REQUIRES_OK(ctx, IsNestedTensorList(ctx->Input(0), &is_nested));
    OP_REQUIRES(ctx, !is_nested,
                errors::Unimplemented("Only non-nested TensorList is supported "
                                      "for TensorListGetItem."));

    xla::XlaOp buffer;
    OP_REQUIRES_OK(ctx, GetTensorListBuffer(ctx->Input(0), &buffer));
    ctx->SetOutput(0, buffer);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(TensorListStackOp);
};

REGISTER_XLA_OP(Name("TensorListStack"), TensorListStackOp);

class TensorListConcatOp : public XlaOpKernel {
 public:
  explicit TensorListConcatOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaOp input = ctx->Input(0);

    // Check that the TensorList is initialized.
    bool is_initialized;
    OP_REQUIRES_OK(ctx, (IsTensorListInitialized(input, &is_initialized)));
    OP_REQUIRES(ctx, is_initialized,
                errors::InvalidArgument("TensorList is not initialized"));

    // Only non-nested TensorList is supported for now.
    bool is_nested;
    OP_REQUIRES_OK(ctx, IsNestedTensorList(input, &is_nested));
    OP_REQUIRES(ctx, !is_nested,
                errors::Unimplemented("Only non-nested TensorList is supported "
                                      "for TensorListConcat."));

    xla::XlaOp buffer;
    OP_REQUIRES_OK(ctx, GetTensorListBuffer(input, &buffer));

    xla::XlaBuilder* b = input.builder();
    auto shape_or = b->GetShape(buffer);
    OP_REQUIRES_OK(ctx, shape_or.status());
    xla::Shape element_shape = shape_or.ConsumeValueOrDie();
    std::vector<int64> element_dims =
        xla::SpanToVector(element_shape.dimensions());
    OP_REQUIRES(
        ctx, element_dims.size() > 1,
        errors::Unimplemented("TensorList of scalars is not supported"));
    int64 num_elements = element_dims[0];
    int64 tensor_lengths = element_dims[1];

    std::vector<int64> new_dims = {num_elements * tensor_lengths};

    for (int i = 2; i < element_dims.size(); i++) {
      new_dims.push_back(element_dims[i]);
    }

    xla::XlaOp out = xla::Reshape(buffer, new_dims);
    ctx->SetOutput(0, out);

    // Second output is a tensor of lengths of returned tensors.
    xla::XlaOp lengths = xla::ConstantR1(b, num_elements, tensor_lengths);
    ctx->SetOutput(1, lengths);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(TensorListConcatOp);
};

REGISTER_XLA_OP(Name("TensorListConcatV2"), TensorListConcatOp);

class TensorListSplitOp : public XlaOpKernel {
 public:
  explicit TensorListSplitOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("element_dtype", &dtype_));
    // Only non-nested TensorList is supported for now.
    OP_REQUIRES(
        ctx, dtype_ != DT_VARIANT,
        errors::Unimplemented(
            "Only non-nested TensorList is supported for TensorListReserve."));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaOp input_tensor = ctx->Input(0);

    xla::XlaBuilder* b = input_tensor.builder();
    auto shape_or = b->GetShape(input_tensor);
    OP_REQUIRES_OK(ctx, shape_or.status());
    xla::Shape element_shape = shape_or.ConsumeValueOrDie();
    std::vector<int64> element_dims =
        xla::SpanToVector(element_shape.dimensions());
    OP_REQUIRES(
        ctx, !element_dims.empty(),
        errors::Unimplemented("Element dimensions have to be non-empty"));

    std::vector<int64> lengths;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntVector(2, &lengths));
    OP_REQUIRES(ctx, !lengths.empty(),
                errors::Unimplemented("Length has to be non-empty"));
    int64 length = lengths[0];
    for (int64 len : lengths) {
      OP_REQUIRES(ctx, len == length,
                  errors::Unimplemented("All lengths have to be the same"));
    }
    OP_REQUIRES(
        ctx, element_dims[0] % length == 0,
        errors::Unimplemented("Buffer size has to be a multiple of length"));
    std::vector<int64> new_dims = {element_dims[0] / length, length};
    for (int i = 1; i < element_dims.size(); i++) {
      new_dims.push_back(element_dims[i]);
    }

    xla::XlaOp reshaped = xla::Reshape(input_tensor, new_dims);

    xla::XlaOp result;
    OP_REQUIRES_OK(ctx, ExecuteTensorListFromTensor(length, reshaped, &result));
    ctx->SetTensorListOutput(0, result);
  }

 private:
  DataType dtype_;

  TF_DISALLOW_COPY_AND_ASSIGN(TensorListSplitOp);
};

REGISTER_XLA_OP(Name("TensorListSplit")
                    .CompileTimeConstantInput("element_shape")
                    .CompileTimeConstantInput("lengths"),
                TensorListSplitOp);

class TensorListFromTensorOp : public XlaOpKernel {
 public:
  explicit TensorListFromTensorOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape& tensor_shape = ctx->InputShape(0);
    int num_elements = tensor_shape.dim_size(0);
    const xla::XlaOp tensor = ctx->Input(0);
    xla::XlaOp result;
    OP_REQUIRES_OK(ctx,
                   ExecuteTensorListFromTensor(num_elements, tensor, &result));
    auto list_shape_or = ctx->builder()->GetShape(result);
    ctx->SetTensorListOutput(0, result);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(TensorListFromTensorOp);
};

REGISTER_XLA_OP(
    Name("TensorListFromTensor").CompileTimeConstantInput("element_shape"),
    TensorListFromTensorOp);

class TensorListSetItemOp : public XlaOpKernel {
 public:
  explicit TensorListSetItemOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaOp list = ctx->Input(0);
    xla::XlaOp index = ctx->Input(1);
    xla::XlaOp element = ctx->Input(2);
    xla::XlaOp initialized_list;
    OP_REQUIRES_OK(ctx, GetInitializedTensorListForElement(
                            list, element, /*element_is_tensor_list=*/false,
                            &initialized_list));

    // Only non-nested TensorList is supported for now.
    bool is_nested;
    OP_REQUIRES_OK(ctx, IsNestedTensorList(initialized_list, &is_nested));
    OP_REQUIRES(ctx, !is_nested,
                errors::Unimplemented("Only non-nested TensorList is supported "
                                      "for TensorListSetItem."));

    xla::XlaOp result;
    OP_REQUIRES_OK(ctx, ExecuteTensorListSetItem(initialized_list, index,
                                                 element, &result));

    ctx->SetTensorListOutput(0, result);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(TensorListSetItemOp);
};

REGISTER_XLA_OP(Name("TensorListSetItem"), TensorListSetItemOp);

class TensorListPushBackOp : public XlaOpKernel {
 public:
  explicit TensorListPushBackOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaOp list = ctx->Input(0);
    xla::XlaOp element = ctx->Input(1);
    bool element_is_tensor_list = IsTensorListInput(ctx, 1);
    xla::XlaOp initialized_list;
    OP_REQUIRES_OK(
        ctx, GetInitializedTensorListForElement(
                 list, element, element_is_tensor_list, &initialized_list));

    xla::XlaOp result;
    OP_REQUIRES_OK(ctx,
                   ExecuteTensorListPushBack(initialized_list, element,
                                             element_is_tensor_list, &result));

    ctx->SetTensorListOutput(0, result);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(TensorListPushBackOp);
};

REGISTER_XLA_OP(Name("TensorListPushBack").AllowVariantTypes(),
                TensorListPushBackOp);

class TensorListPopBackOp : public XlaOpKernel {
 public:
  explicit TensorListPopBackOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    // Check that the TensorList is initialized.
    bool is_initialized;
    OP_REQUIRES_OK(ctx,
                   (IsTensorListInitialized(ctx->Input(0), &is_initialized)));
    OP_REQUIRES(ctx, is_initialized,
                errors::InvalidArgument("TensorList is not initialized"));

    xla::XlaOp list = ctx->Input(0);
    xla::XlaOp list_result, element_result;
    bool element_is_tensor_list;
    OP_REQUIRES_OK(ctx,
                   ExecuteTensorListPopBack(list, &list_result, &element_result,
                                            &element_is_tensor_list));

    ctx->SetTensorListOutput(0, list_result);
    if (element_is_tensor_list) {
      ctx->SetTensorListOutput(1, element_result);
    } else {
      ctx->SetOutput(1, element_result);
    }
  }

 private:
  DataType dtype_;

  TF_DISALLOW_COPY_AND_ASSIGN(TensorListPopBackOp);
};

REGISTER_XLA_OP(Name("TensorListPopBack").AllowVariantTypes(),
                TensorListPopBackOp);

}  // anonymous namespace
}  // namespace tensorflow
