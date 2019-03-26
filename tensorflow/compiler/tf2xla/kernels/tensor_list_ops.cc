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

#include "tensorflow/compiler/tf2xla/kernels/tensor_list_utils.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/status_macros.h"
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

class TensorListLengthOp : public XlaOpKernel {
 public:
  explicit TensorListLengthOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaOp index;
    OP_REQUIRES_OK(ctx, GetTensorListPushIndex(ctx->Input(0), &index));
    ctx->SetOutput(0, index);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(TensorListLengthOp);
};

REGISTER_XLA_OP(Name("TensorListLength"), TensorListLengthOp);

// Creates an empty list with size (leading_dim, *element_shape) if
// element_shape is known at compile time. Otherwise creates one with size
// (leading_dim, 0) which gets initialized later in `GetInitializedList`.
Status CreateZerosList(XlaOpKernelContext* ctx, int element_shape_index,
                       int64 leading_dim, DataType dtype, xla::XlaOp* list) {
  TensorShape list_shape;
  list_shape.AddDim(leading_dim);
  xla::XlaOp element_shape_handle = ctx->Input(element_shape_index);
  TF_ASSIGN_OR_RETURN(
      bool is_element_shape_compile_time_const,
      element_shape_handle.builder()->IsConstant(element_shape_handle));
  PartialTensorShape partial_element_shape;
  if (is_element_shape_compile_time_const) {
    TF_RETURN_IF_ERROR(ctx->ConstantInputAsPartialShape(
        element_shape_index, &partial_element_shape));
  }
  if (is_element_shape_compile_time_const &&
      partial_element_shape.IsFullyDefined()) {
    TensorShape element_shape;
    partial_element_shape.AsTensorShape(&element_shape);
    list_shape.AppendShape(element_shape);
  } else {
    // If element_shape is not a compile time constant or if it is not fully
    // defined we will have to wait for the first write call to fully allocate
    // the array.
    // TODO(srbs): We are using element_shape of [0] as a proxy to denote an
    // uninitialized list. A better implementation may be to represent the
    // list as a 3-tuple containining an explicit "initialized" flag. However,
    // we would still need to create a dummy tensor for the first tuple
    // element.
    list_shape.AddDim(0);
  }
  *list = xla::Broadcast(XlaHelpers::Zero(ctx->builder(), dtype),
                         list_shape.dim_sizes());
  return Status::OK();
}

class TensorListReserveOp : public XlaOpKernel {
 public:
  explicit TensorListReserveOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("element_dtype", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    int64 num_elements;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntScalar(1, &num_elements));

    xla::XlaOp buffer;
    OP_REQUIRES_OK(ctx, CreateZerosList(ctx, 0, num_elements, dtype_, &buffer));

    xla::XlaOp output_list;
    OP_REQUIRES_OK(
        ctx, BuildTensorList(
                 buffer, xla::ConstantR0<int32>(ctx->builder(), num_elements),
                 &output_list));
    ctx->SetTensorListOutput(0, output_list);
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
    OP_REQUIRES(
        ctx, max_num_elements >= 0,
        errors::InvalidArgument("XLA compilation requires a fixed tensor list "
                                "size. Set the max number of elements."));

    xla::XlaOp buffer;
    OP_REQUIRES_OK(ctx,
                   CreateZerosList(ctx, 0, max_num_elements, dtype_, &buffer));

    xla::XlaOp output_list;
    OP_REQUIRES_OK(
        ctx, BuildTensorList(buffer, xla::ConstantR0<int32>(ctx->builder(), 0),
                             &output_list));
    ctx->SetTensorListOutput(0, output_list);
  }

 private:
  DataType dtype_;

  TF_DISALLOW_COPY_AND_ASSIGN(EmptyTensorListOp);
};

REGISTER_XLA_OP(Name("EmptyTensorList")
                    .CompileTimeConstantInput("element_shape")
                    .CompileTimeConstantInput("max_num_elements"),
                EmptyTensorListOp);

class TensorListElementShapeOp : public XlaOpKernel {
 public:
  explicit TensorListElementShapeOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shape_type", &shape_type_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* b = ctx->builder();
    TensorShape shape;
    OP_REQUIRES_OK(ctx, GetTensorListBufferShape(ctx->Input(0), &shape));
    shape.RemoveDim(0);

    switch (shape_type_) {
      case DT_INT64:
        ctx->SetOutput(0, xla::ConstantR1<int64>(b, shape.dim_sizes()));
        break;
      case DT_INT32: {
        std::vector<int32> size;
        for (int64 s : shape.dim_sizes()) {
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

REGISTER_XLA_OP(Name("TensorListElementShape"), TensorListElementShapeOp);

class TensorListGetItemOp : public XlaOpKernel {
 public:
  explicit TensorListGetItemOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("element_dtype", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* b = ctx->builder();
    xla::XlaOp state = ctx->Input(0);

    TensorShape shape;
    OP_REQUIRES_OK(ctx, GetTensorListBufferShape(ctx->Input(0), &shape));

    xla::XlaOp buffer;
    OP_REQUIRES_OK(ctx, GetTensorListBuffer(state, &buffer));
    xla::XlaOp index = ctx->Input(1);

    // start_indices of the DynamicSlice are [index, 0, 0, ..., 0].
    std::vector<xla::XlaOp> start_indices(shape.dims(),
                                          xla::ConstantR0<int32>(b, 0));
    start_indices[0] = index;
    auto slice_shape = shape.dim_sizes();
    slice_shape[0] = 1LL;

    xla::XlaOp read = xla::DynamicSlice(buffer, start_indices, slice_shape);
    // Remove the leading '1' dimension.
    std::vector<int64> value_shape(slice_shape.begin() + 1, slice_shape.end());

    ctx->SetOutput(0, xla::Reshape(read, value_shape));
  }

 private:
  DataType dtype_;

  TF_DISALLOW_COPY_AND_ASSIGN(TensorListGetItemOp);
};

REGISTER_XLA_OP(Name("TensorListGetItem"), TensorListGetItemOp);

class TensorListStackOp : public XlaOpKernel {
 public:
  explicit TensorListStackOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("element_dtype", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaOp buffer;
    OP_REQUIRES_OK(ctx, GetTensorListBuffer(ctx->Input(0), &buffer));
    ctx->SetOutput(0, buffer);
  }

 private:
  DataType dtype_;

  TF_DISALLOW_COPY_AND_ASSIGN(TensorListStackOp);
};

REGISTER_XLA_OP(Name("TensorListStack"), TensorListStackOp);

class TensorListFromTensorOp : public XlaOpKernel {
 public:
  explicit TensorListFromTensorOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("element_dtype", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    PartialTensorShape element_shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsPartialShape(1, &element_shape));

    const TensorShape tensor_shape = ctx->InputShape(0);
    // Ensure that tensor_shape is compatible with element_shape.
    PartialTensorShape unused;
    OP_REQUIRES_OK(
        ctx,
        element_shape.MergeWith(
            PartialTensorShape(
                absl::Span<const int64>(tensor_shape.dim_sizes()).subspan(1)),
            &unused));
    OP_REQUIRES(ctx, tensor_shape.dims() > 0,
                errors::InvalidArgument("Input value must be at least a "
                                        "vector but received shape: ",
                                        tensor_shape.DebugString()));
    const int num_elements = tensor_shape.dim_size(0);

    xla::XlaBuilder* b = ctx->builder();
    const xla::XlaOp tensor = ctx->Input(0);

    xla::XlaOp output_list;
    OP_REQUIRES_OK(
        ctx, BuildTensorList(tensor, xla::ConstantR0<int32>(b, num_elements),
                             &output_list));
    ctx->SetTensorListOutput(0, output_list);
  }

 private:
  DataType dtype_;

  TF_DISALLOW_COPY_AND_ASSIGN(TensorListFromTensorOp);
};

REGISTER_XLA_OP(
    Name("TensorListFromTensor").CompileTimeConstantInput("element_shape"),
    TensorListFromTensorOp);

// Returns the 0'th element of `tuple` containing the list tensor if it has been
// initialized already else creates one lazily. This allows lazy initialization
// of the list on the first call to SetItem or PushBack.
Status GetInitializedList(const xla::XlaOp& input_list,
                          const TensorShape& element_shape, DataType dtype,
                          xla::XlaOp* output_list_buffer) {
  bool is_already_initialized;
  TF_RETURN_IF_ERROR(
      IsTensorListInitialized(input_list, &is_already_initialized));
  TensorShape input_list_shape;
  TF_RETURN_IF_ERROR(GetTensorListBufferShape(input_list, &input_list_shape));
  TensorShape input_list_element_shape = input_list_shape;
  input_list_element_shape.RemoveDim(0);

  if (is_already_initialized) {
    TF_RET_CHECK(element_shape == input_list_element_shape);
    TF_RETURN_IF_ERROR(GetTensorListBuffer(input_list, output_list_buffer));
    return Status::OK();
  }

  int64 leading_dim = input_list_shape.dim_size(0);
  TensorShape output_list_shape = element_shape;
  output_list_shape.InsertDim(0, leading_dim);

  xla::XlaOp output_list;
  TF_RETURN_IF_ERROR(
      InitializeTensorList(input_list, output_list_shape, &output_list));
  TF_RETURN_IF_ERROR(GetTensorListBuffer(output_list, output_list_buffer));
  return Status::OK();
}

class TensorListSetItemOp : public XlaOpKernel {
 public:
  explicit TensorListSetItemOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("element_dtype", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* b = ctx->builder();
    xla::XlaOp tl = ctx->Input(0);
    TensorShape elem_shape = ctx->InputShape(2);

    xla::XlaOp buffer;
    OP_REQUIRES_OK(ctx, GetInitializedList(tl, elem_shape, dtype_, &buffer));
    xla::XlaOp push_index;
    OP_REQUIRES_OK(ctx, GetTensorListPushIndex(tl, &push_index));

    xla::XlaOp index = ctx->Input(1);
    xla::XlaOp value = ctx->Input(2);

    // start_indices of the DynamicUpdateSlice are [index, 0, 0, ..., 0].
    std::vector<xla::XlaOp> start_indices(elem_shape.dims() + 1,
                                          xla::ConstantR0<int32>(b, 0));
    start_indices[0] = index;

    TensorShape slice_shape = elem_shape;
    slice_shape.InsertDim(0, 1LL);
    auto update = xla::Reshape(value, slice_shape.dim_sizes());

    xla::XlaOp output_list;
    OP_REQUIRES_OK(ctx, BuildTensorList(xla::DynamicUpdateSlice(buffer, update,
                                                                start_indices),
                                        push_index, &output_list));
    ctx->SetTensorListOutput(0, output_list);
  }

 private:
  DataType dtype_;

  TF_DISALLOW_COPY_AND_ASSIGN(TensorListSetItemOp);
};

REGISTER_XLA_OP(Name("TensorListSetItem"), TensorListSetItemOp);

class TensorListPushBackOp : public XlaOpKernel {
 public:
  explicit TensorListPushBackOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("element_dtype", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* b = ctx->builder();
    xla::XlaOp list_tuple = ctx->Input(0);
    TensorShape elem_shape = ctx->InputShape(1);

    xla::XlaOp buffer;
    OP_REQUIRES_OK(ctx,
                   GetInitializedList(list_tuple, elem_shape, dtype_, &buffer));

    xla::XlaOp index;
    OP_REQUIRES_OK(ctx, GetTensorListPushIndex(list_tuple, &index));
    xla::XlaOp value = ctx->Input(1);

    // start_indices of the DynamicUpdateSlice are [index, 0, 0, ..., 0].
    std::vector<xla::XlaOp> start_indices(elem_shape.dims() + 1,
                                          xla::ConstantR0<int32>(b, 0));
    start_indices[0] = index;

    TensorShape slice_shape = elem_shape;
    slice_shape.InsertDim(0, 1LL);
    auto update = xla::Reshape(value, slice_shape.dim_sizes());

    xla::XlaOp output_list;
    OP_REQUIRES_OK(
        ctx,
        BuildTensorList(xla::DynamicUpdateSlice(buffer, update, start_indices),
                        index + xla::ConstantR0<int32>(b, 1), &output_list));
    ctx->SetTensorListOutput(0, output_list);
  }

 private:
  DataType dtype_;

  TF_DISALLOW_COPY_AND_ASSIGN(TensorListPushBackOp);
};

REGISTER_XLA_OP(Name("TensorListPushBack"), TensorListPushBackOp);

class TensorListPopBackOp : public XlaOpKernel {
 public:
  explicit TensorListPopBackOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("element_dtype", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* b = ctx->builder();
    xla::XlaOp state = ctx->Input(0);

    TensorShape shape;
    OP_REQUIRES_OK(ctx, GetTensorListBufferShape(ctx->Input(0), &shape));

    xla::XlaOp ta;
    OP_REQUIRES_OK(ctx, GetTensorListBuffer(state, &ta));
    xla::XlaOp index;
    OP_REQUIRES_OK(ctx, GetTensorListPushIndex(state, &index));

    index = index - xla::ConstantR0<int32>(b, 1);

    // start_indices of the DynamicSlice are [index, 0, 0, ..., 0].
    std::vector<xla::XlaOp> start_indices(shape.dims(),
                                          xla::ConstantR0<int32>(b, 0));
    start_indices[0] = index;
    auto slice_shape = shape.dim_sizes();
    slice_shape[0] = 1LL;

    xla::XlaOp read = xla::DynamicSlice(ta, start_indices, slice_shape);
    // Remove the leading '1' dimension.
    std::vector<int64> value_shape(slice_shape.begin() + 1, slice_shape.end());

    xla::XlaOp output_list;
    OP_REQUIRES_OK(ctx, BuildTensorList(ta, index, &output_list));
    ctx->SetTensorListOutput(0, output_list);
    ctx->SetOutput(1, xla::Reshape(read, value_shape));
  }

 private:
  DataType dtype_;

  TF_DISALLOW_COPY_AND_ASSIGN(TensorListPopBackOp);
};

REGISTER_XLA_OP(Name("TensorListPopBack"), TensorListPopBackOp);

}  // anonymous namespace
}  // namespace tensorflow
