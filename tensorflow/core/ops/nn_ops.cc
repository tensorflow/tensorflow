/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/mirror_pad_mode.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

namespace {

Status FractionalPoolShapeFn(InferenceContext* c) {
  ShapeHandle input;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input));

  std::vector<float> pooling_ratio;
  TF_RETURN_IF_ERROR(c->GetAttr("pooling_ratio", &pooling_ratio));
  if (pooling_ratio.size() != 4) {
    return errors::InvalidArgument(
        "pooling_ratio field must specify 4 dimensions");
  }
  std::vector<DimensionHandle> output_dims;
  for (int i = 0; i < 4; ++i) {
    DimensionHandle d = c->Dim(input, i);
    if (c->ValueKnown(d)) {
      // This must match the same logic in the kernel function in
      // core/kernels/fractional_max_pool_op.cc.
      auto val = static_cast<int64>(floor(c->Value(d) / pooling_ratio[i]));
      if (val < 0) {
        return errors::InvalidArgument("Size computed for dim ", i,
                                       " is negative: ", val);
      }
      output_dims.push_back(c->MakeDim(val));
    } else {
      output_dims.push_back(c->UnknownDim());
    }
  }

  c->set_output(0, c->MakeShape(output_dims));
  c->set_output(1, c->Vector(output_dims[1]));
  c->set_output(2, c->Vector(output_dims[2]));
  return Status::OK();
}

}  // namespace

// --------------------------------------------------------------------------

REGISTER_OP("AvgPool")
    .Input("value: T")
    .Output("output: T")
    .Attr("ksize: list(int) >= 4")
    .Attr("strides: list(int) >= 4")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("T: {half, bfloat16, float, double}")
    .SetShapeFn(shape_inference::AvgPoolShape);

REGISTER_OP("AvgPoolGrad")
    .Input("orig_input_shape: int32")
    .Input("grad: T")
    .Output("output: T")
    .Attr("ksize: list(int) >= 4")
    .Attr("strides: list(int) >= 4")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("T: {half, bfloat16, float, double}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &s));
      TF_RETURN_IF_ERROR(c->WithRank(s, 4, &s));
      c->set_output(0, s);
      return Status::OK();
    });

// --------------------------------------------------------------------------

REGISTER_OP("BatchNormWithGlobalNormalization")
    .Input("t: T")
    .Input("m: T")
    .Input("v: T")
    .Input("beta: T")
    .Input("gamma: T")
    .Output("result: T")
    .Attr("T: numbertype")
    .Attr("variance_epsilon: float")
    .Attr("scale_after_normalization: bool")
    .Deprecated(9, "Use tf.nn.batch_normalization()")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input));

      DimensionHandle last_dim = c->Dim(input, 3);
      for (int i = 1; i < 5; ++i) {  // covers m, v, beta, gamma
        ShapeHandle vec;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 1, &vec));
        TF_RETURN_IF_ERROR(c->Merge(last_dim, c->Dim(vec, 0), &last_dim));
      }

      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->ReplaceDim(input, 3, last_dim, &out));
      c->set_output(0, out);
      return Status::OK();
    });

REGISTER_OP("BatchNormWithGlobalNormalizationGrad")
    .Input("t: T")
    .Input("m: T")
    .Input("v: T")
    .Input("gamma: T")
    .Input("backprop: T")
    .Output("dx: T")
    .Output("dm: T")
    .Output("dv: T")
    .Output("db: T")
    .Output("dg: T")
    .Attr("T: numbertype")
    .Attr("variance_epsilon: float")
    .Attr("scale_after_normalization: bool")
    .Deprecated(9, "Use tf.nn.batch_normalization()")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input));
      TF_RETURN_IF_ERROR(
          c->Merge(input, c->input(4), &input));  // with backprop

      DimensionHandle last_dim = c->Dim(input, 3);
      for (int i = 1; i < 4; ++i) {  // covers m, v, gamma
        ShapeHandle vec;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 1, &vec));
        TF_RETURN_IF_ERROR(c->Merge(last_dim, c->Dim(vec, 0), &last_dim));
      }

      ShapeHandle dx;
      TF_RETURN_IF_ERROR(c->ReplaceDim(input, 3, last_dim, &dx));
      c->set_output(0, dx);

      ShapeHandle vector_shape = c->Vector(last_dim);
      c->set_output(1, vector_shape);
      c->set_output(2, vector_shape);
      c->set_output(3, vector_shape);
      c->set_output(4, vector_shape);
      return Status::OK();
    });

// --------------------------------------------------------------------------

REGISTER_OP("FusedBatchNorm")
    .Input("x: T")
    .Input("scale: T")
    .Input("offset: T")
    .Input("mean: T")
    .Input("variance: T")
    .Output("y: T")
    .Output("batch_mean: T")
    .Output("batch_variance: T")
    .Output("reserve_space_1: T")
    .Output("reserve_space_2: T")
    .Attr("T: {float}")
    .Attr("epsilon: float = 0.0001")
    .Attr("data_format: string = 'NHWC'")
    .Attr("is_training: bool = true")
    .SetShapeFn(shape_inference::FusedBatchNormShape);

REGISTER_OP("FusedBatchNormV2")
    .Input("x: T")
    .Input("scale: U")
    .Input("offset: U")
    .Input("mean: U")
    .Input("variance: U")
    .Output("y: T")
    .Output("batch_mean: U")
    .Output("batch_variance: U")
    .Output("reserve_space_1: U")
    .Output("reserve_space_2: U")
    .Attr("T: {half, bfloat16, float}")
    .Attr("U: {float}")
    .Attr("epsilon: float = 0.0001")
    .Attr("data_format: string = 'NHWC'")
    .Attr("is_training: bool = true")
    .SetShapeFn(shape_inference::FusedBatchNormShape);

REGISTER_OP("FusedBatchNormGrad")
    .Input("y_backprop: T")
    .Input("x: T")
    .Input("scale: T")
    .Input("reserve_space_1: T")
    .Input("reserve_space_2: T")
    .Output("x_backprop: T")
    .Output("scale_backprop: T")
    .Output("offset_backprop: T")
    .Output("reserve_space_3: T")
    .Output("reserve_space_4: T")
    .Attr("T: {float}")
    .Attr("epsilon: float = 0.0001")
    .Attr("data_format: string = 'NHWC'")
    .Attr("is_training: bool = true")
    .SetShapeFn(shape_inference::FusedBatchNormGradShape);

REGISTER_OP("FusedBatchNormGradV2")
    .Input("y_backprop: T")
    .Input("x: T")
    .Input("scale: float")
    .Input("reserve_space_1: U")
    .Input("reserve_space_2: U")
    .Output("x_backprop: T")
    .Output("scale_backprop: U")
    .Output("offset_backprop: U")
    .Output("reserve_space_3: U")
    .Output("reserve_space_4: U")
    .Attr("T: {half, bfloat16, float}")
    .Attr("U: {float}")
    .Attr("epsilon: float = 0.0001")
    .Attr("data_format: string = 'NHWC'")
    .Attr("is_training: bool = true")
    .SetShapeFn(shape_inference::FusedBatchNormGradShape);

// --------------------------------------------------------------------------

REGISTER_OP("BiasAdd")
    .Attr("T: numbertype")
    .Input("value: T")
    .Input("bias: T")
    .Attr(GetConvnetDataFormatAttrString())
    .Output("output: T")
    .SetShapeFn(shape_inference::BiasAddShape);
// --------------------------------------------------------------------------

REGISTER_OP("BiasAddGrad")
    .Attr("T: numbertype")
    .Input("out_backprop: T")
    .Attr(GetConvnetDataFormatAttrString())
    .Output("output: T")
    .SetShapeFn(shape_inference::BiasAddGradShape);
// --------------------------------------------------------------------------

REGISTER_OP("BiasAddV1")
    .Attr("T: numbertype")
    .Input("value: T")
    .Input("bias: T")
    .Output("output: T")
    .SetShapeFn(shape_inference::BiasAddShape);
// --------------------------------------------------------------------------

REGISTER_OP("Conv2D")
    .Input("input: T")
    .Input("filter: T")
    .Output("output: T")
    .Attr("T: {half, bfloat16, float, double}")
    .Attr("strides: list(int)")
    .Attr("use_cudnn_on_gpu: bool = true")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn(shape_inference::Conv2DShape);

REGISTER_OP("Conv2DBackpropInput")
    .Input("input_sizes: int32")
    .Input("filter: T")
    .Input("out_backprop: T")
    .Output("output: T")
    .Attr("T: {half, bfloat16, float, double}")
    .Attr("strides: list(int)")
    .Attr("use_cudnn_on_gpu: bool = true")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &s));
      TF_RETURN_IF_ERROR(c->WithRank(s, 4, &s));
      c->set_output(0, s);
      return Status::OK();
    });

// TODO(jeff): Instead of 'use_cudnn_for_gpu', maybe we should have a
// more general string attribute ('kernel_impl'?) that can be used to
// select among several possible implementations.
REGISTER_OP("Conv2DBackpropFilter")
    .Input("input: T")
    .Input("filter_sizes: int32")
    .Input("out_backprop: T")
    .Output("output: T")
    .Attr("T: {half, bfloat16, float, double}")
    .Attr("strides: list(int)")
    .Attr("use_cudnn_on_gpu: bool = true")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(1, &s));
      TF_RETURN_IF_ERROR(c->WithRank(s, 4, &s));
      c->set_output(0, s);
      return Status::OK();
    });

namespace {

Status CommonFusedConvCalculations(InferenceContext* c, bool has_resize) {
  ShapeHandle input;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input));

  ShapeHandle resized = input;
  int paddings_index = 1;
  int filter_index = 2;
  if (has_resize) {
    paddings_index = 2;
    filter_index = 3;

    ShapeHandle unused_size;
    TF_RETURN_IF_ERROR(c->Merge(c->input(1), c->Vector(2), &unused_size));

    const Tensor* size = c->input_tensor(1);
    DimensionHandle new_height = c->UnknownDim();
    DimensionHandle new_width = c->UnknownDim();
    if (size != nullptr) {
      new_height = c->MakeDim(size->flat<int32>()(0));
      new_width = c->MakeDim(size->flat<int32>()(1));
    }
    TF_RETURN_IF_ERROR(c->ReplaceDim(resized, 1, new_height, &resized));
    TF_RETURN_IF_ERROR(c->ReplaceDim(resized, 2, new_width, &resized));
  }

  ShapeHandle paddings;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(paddings_index), 2, &paddings));
  TF_RETURN_IF_ERROR(
      c->WithRank(resized, c->Value(c->Dim(paddings, 0)), &resized));
  TF_RETURN_IF_ERROR(
      c->Merge(paddings, c->Matrix(c->Rank(resized), 2), &paddings));

  const Tensor* paddings_t = c->input_tensor(paddings_index);
  ShapeHandle padded;
  if (paddings_t != nullptr) {
    std::vector<DimensionHandle> output_dims;
    for (int i = 0; i < 4; ++i) {
      DimensionHandle dim = c->Dim(resized, i);
      int64 p0 = static_cast<int64>(paddings_t->matrix<int32>()(i, 0));
      int64 p1 = static_cast<int64>(paddings_t->matrix<int32>()(i, 1));
      if (p0 < 0 || p1 < 0) {
        return errors::InvalidArgument("Paddings must be non-negative");
      }

      TF_RETURN_IF_ERROR(c->Add(dim, p0 + p1, &dim));
      output_dims.push_back(dim);
    }
    padded = c->MakeShape(output_dims);
  } else {
    padded = c->UnknownShapeOfRank(4);
  }

  // Work out the convolution's effect with 'padded' as the input.
  ShapeHandle filter;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(filter_index), 4, &filter));
  std::vector<int32> strides;
  TF_RETURN_IF_ERROR(c->GetAttr("strides", &strides));
  if (strides.size() != 4) {
    return errors::InvalidArgument(
        "Operation requires the stride attribute to contain 4 values, but ",
        "got: ", strides.size());
  }

  int32 stride_rows = strides[1];
  int32 stride_cols = strides[2];

  DimensionHandle batch_size_dim = c->Dim(padded, 0);
  DimensionHandle in_rows_dim = c->Dim(padded, 1);
  DimensionHandle in_cols_dim = c->Dim(padded, 2);
  DimensionHandle filter_rows_dim = c->Dim(filter, 0);
  DimensionHandle filter_cols_dim = c->Dim(filter, 1);
  DimensionHandle output_depth_dim = c->Dim(filter, 3);

  DimensionHandle unused;
  TF_RETURN_IF_ERROR(c->Merge(c->Dim(padded, 3), c->Dim(filter, 2), &unused));

  Padding padding;
  TF_RETURN_IF_ERROR(c->GetAttr("padding", &padding));

  DimensionHandle output_rows, output_cols;
  TF_RETURN_IF_ERROR(GetWindowedOutputSizeFromDims(
      c, in_rows_dim, filter_rows_dim, stride_rows, padding, &output_rows));
  TF_RETURN_IF_ERROR(GetWindowedOutputSizeFromDims(
      c, in_cols_dim, filter_cols_dim, stride_cols, padding, &output_cols));

  ShapeHandle output_shape = c->MakeShape(
      {batch_size_dim, output_rows, output_cols, output_depth_dim});
  c->set_output(0, output_shape);
  return Status::OK();
}

}  // namespace

REGISTER_OP("DataFormatDimMap")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {int32, int64} = DT_INT32")
    .Attr("src_format: string = 'NHWC'")
    .Attr("dst_format: string = 'NCHW'")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("DataFormatVecPermute")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {int32, int64} = DT_INT32")
    .Attr("src_format: string = 'NHWC'")
    .Attr("dst_format: string = 'NCHW'")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("FusedResizeAndPadConv2D")
    .Input("input: T")
    .Input("size: int32")
    .Input("paddings: int32")
    .Input("filter: T")
    .Output("output: T")
    .Attr("T: {float}")
    .Attr("resize_align_corners: bool = false")
    .Attr(GetMirrorPadModeAttrString())
    .Attr("strides: list(int)")
    .Attr(GetPaddingAttrString())
    .SetShapeFn([](InferenceContext* c) {
      return CommonFusedConvCalculations(c, true /* has_resize */);
    });

REGISTER_OP("FusedPadConv2D")
    .Input("input: T")
    .Input("paddings: int32")
    .Input("filter: T")
    .Output("output: T")
    .Attr("T: {float}")
    .Attr(GetMirrorPadModeAttrString())
    .Attr("strides: list(int)")
    .Attr(GetPaddingAttrString())
    .SetShapeFn([](InferenceContext* c) {
      return CommonFusedConvCalculations(c, false /* has_resize */);
    });

// --------------------------------------------------------------------------

REGISTER_OP("DepthwiseConv2dNative")
    .Input("input: T")
    .Input("filter: T")
    .Output("output: T")
    .Attr("T: {half, bfloat16, float, double}")
    .Attr("strides: list(int)")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn(shape_inference::DepthwiseConv2DNativeShape);

REGISTER_OP("DepthwiseConv2dNativeBackpropInput")
    .Input("input_sizes: int32")
    .Input("filter: T")
    .Input("out_backprop: T")
    .Output("output: T")
    .Attr("T: {half, bfloat16, float, double}")
    .Attr("strides: list(int)")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &s));
      TF_RETURN_IF_ERROR(c->WithRank(s, 4, &s));
      c->set_output(0, s);
      return Status::OK();
    });

REGISTER_OP("DepthwiseConv2dNativeBackpropFilter")
    .Input("input: T")
    .Input("filter_sizes: int32")
    .Input("out_backprop: T")
    .Output("output: T")
    .Attr("T: {half, bfloat16, float, double}")
    .Attr("strides: list(int)")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(1, &s));
      TF_RETURN_IF_ERROR(c->WithRank(s, 4, &s));
      c->set_output(0, s);
      return Status::OK();
    });

// --------------------------------------------------------------------------
REGISTER_OP("Conv3D")
    .Input("input: T")
    .Input("filter: T")
    .Output("output: T")
    .Attr("T: {half, bfloat16, float, double}")
    .Attr("strides: list(int) >= 5")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnet3dDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1, 1]")
    .SetShapeFn(shape_inference::Conv3DShape);

REGISTER_OP("Conv3DBackpropInput")
    .Input("input: T")
    .Input("filter: T")
    .Input("out_backprop: T")
    .Output("output: T")
    .Attr("T: {half, float, double}")
    .Attr("strides: list(int) >= 5")
    .Attr(GetPaddingAttrString())
    .Deprecated(10, "Use Conv3DBackpropInputV2")
    .Attr("dilations: list(int) = [1, 1, 1, 1, 1]")
    .SetShapeFn([](InferenceContext* c) {
      return UnchangedShapeWithRank(c, 5);
    });

REGISTER_OP("Conv3DBackpropFilter")
    .Input("input: T")
    .Input("filter: T")
    .Input("out_backprop: T")
    .Output("output: T")
    .Attr("T: {half, float, double}")
    .Attr("strides: list(int) >= 5")
    .Attr(GetPaddingAttrString())
    .Deprecated(10, "Use Conv3DBackpropFilterV2")
    .Attr("dilations: list(int) = [1, 1, 1, 1, 1]")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 5, &out));
      c->set_output(0, out);
      return Status::OK();
    });

REGISTER_OP("Conv3DBackpropInputV2")
    .Input("input_sizes: Tshape")
    .Input("filter: T")
    .Input("out_backprop: T")
    .Output("output: T")
    .Attr("T: {half, bfloat16, float, double}")
    .Attr("strides: list(int) >= 5")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnet3dDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1, 1]")
    .Attr("Tshape: {int32, int64} = DT_INT32")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &s));
      TF_RETURN_IF_ERROR(c->WithRank(s, 5, &s));
      c->set_output(0, s);
      return Status::OK();
    });

REGISTER_OP("Conv3DBackpropFilterV2")
    .Input("input: T")
    .Input("filter_sizes: int32")
    .Input("out_backprop: T")
    .Output("output: T")
    .Attr("T: {half, bfloat16, float, double}")
    .Attr("strides: list(int) >= 5")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnet3dDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1, 1]")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(1, &s));
      TF_RETURN_IF_ERROR(c->WithRank(s, 5, &s));
      c->set_output(0, s);
      return Status::OK();
    });

// --------------------------------------------------------------------------

REGISTER_OP("AvgPool3D")
    .Input("input: T")
    .Output("output: T")
    .Attr("ksize: list(int) >= 5")
    .Attr("strides: list(int) >= 5")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnet3dDataFormatAttrString())
    .Attr("T: {half, bfloat16, float, double}")
    .SetShapeFn(shape_inference::Pool3DShape);

REGISTER_OP("AvgPool3DGrad")
    .Input("orig_input_shape: int32")
    .Input("grad: T")
    .Output("output: T")
    .Attr("ksize: list(int) >= 5")
    .Attr("strides: list(int) >= 5")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnet3dDataFormatAttrString())
    .Attr("T: {half, bfloat16, float, double}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &s));
      TF_RETURN_IF_ERROR(c->WithRank(s, 5, &s));
      c->set_output(0, s);
      return Status::OK();
    });

// --------------------------------------------------------------------------

REGISTER_OP("MaxPool3D")
    .Input("input: T")
    .Output("output: T")
    .Attr("ksize: list(int) >= 5")
    .Attr("strides: list(int) >= 5")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnet3dDataFormatAttrString())
    .Attr("T: {half, bfloat16, float}")
    .SetShapeFn(shape_inference::Pool3DShape);

REGISTER_OP("MaxPool3DGrad")
    .Input("orig_input: TInput")
    .Input("orig_output: TInput")
    .Input("grad: T")
    .Output("output: T")
    .Attr("ksize: list(int) >= 5")
    .Attr("strides: list(int) >= 5")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnet3dDataFormatAttrString())
    .Attr("T: {half, bfloat16, float} = DT_FLOAT")
    .Attr("TInput: {half, bfloat16, float} = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) {
      return UnchangedShapeWithRank(c, 5);
    });

REGISTER_OP("MaxPool3DGradGrad")
    .Input("orig_input: T")
    .Input("orig_output: T")
    .Input("grad: T")
    .Output("output: T")
    .Attr("ksize: list(int) >= 5 ")
    .Attr("strides: list(int) >= 5")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnet3dDataFormatAttrString())
    .Attr("T: realnumbertype")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::Pool3DShape(c));
      ShapeHandle unused;
      // Validate 'orig_input' is the same shape as 'grad'
      TF_RETURN_IF_ERROR(c->Merge(c->input(0), c->input(2), &unused));
      // Validate 'orig_output' is same shape as 'output'
      TF_RETURN_IF_ERROR(c->Merge(c->input(1), c->output(0), &unused));
      return Status::OK();
    });

// --------------------------------------------------------------------------

REGISTER_OP("L2Loss")
    .Input("t: T")
    .Output("output: T")
    .Attr("T: {half, bfloat16, float, double}")
    .SetShapeFn(shape_inference::ScalarShape);

// --------------------------------------------------------------------------

REGISTER_OP("LRN")
    .Input("input: T")
    .Output("output: T")
    .Attr("depth_radius: int = 5")
    .Attr("bias: float = 1.0")
    .Attr("alpha: float = 1.0")
    .Attr("beta: float = 0.5")
    .Attr("T: {half, bfloat16, float} = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) {
      return UnchangedShapeWithRank(c, 4);
    });

REGISTER_OP("LRNGrad")
    .Input("input_grads: T")
    .Input("input_image: T")
    .Input("output_image: T")
    .Output("output: T")
    .Attr("depth_radius: int = 5")
    .Attr("bias: float = 1.0")
    .Attr("alpha: float = 1.0")
    .Attr("beta: float = 0.5")
    .Attr("T: {half, bfloat16, float} = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &s));  // input_grads
      TF_RETURN_IF_ERROR(c->Merge(s, c->input(1), &s));     // input_image
      TF_RETURN_IF_ERROR(c->Merge(s, c->input(2), &s));     // output_image
      c->set_output(0, s);
      return Status::OK();
    });

// --------------------------------------------------------------------------

REGISTER_OP("MaxPool")
    .Attr(
        "T: {half, bfloat16, float, double, int32, int64, uint8, int16, int8, "
        "uint16, qint8} = DT_FLOAT")
    .Attr("ksize: list(int) >= 4")
    .Attr("strides: list(int) >= 4")
    .Attr(GetPaddingAttrString())
    .Attr("data_format: {'NHWC', 'NCHW', 'NCHW_VECT_C'} = 'NHWC'")
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn(shape_inference::MaxPoolShape);

REGISTER_OP("MaxPoolV2")
    .Attr(
        "T: {half, bfloat16, float, double, int32, int64, uint8, int16, int8, "
        "uint16, qint8} = DT_FLOAT")
    .Attr(GetPaddingAttrString())
    .Attr("data_format: {'NHWC', 'NCHW', 'NCHW_VECT_C'} = 'NHWC'")
    .Input("input: T")
    .Input("ksize: int32")
    .Input("strides: int32")
    .Output("output: T")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::MaxPoolV2Shape(c, 3));
      return Status::OK();
    });

REGISTER_OP("MaxPoolGrad")
    .Attr("ksize: list(int) >= 4")
    .Attr("strides: list(int) >= 4")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Input("orig_input: T")
    .Input("orig_output: T")
    .Input("grad: T")
    .Output("output: T")
    .Attr("T: realnumbertype = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) {
      return UnchangedShapeWithRank(c, 4);
    });

REGISTER_OP("MaxPoolGradV2")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Input("orig_input: T")
    .Input("orig_output: T")
    .Input("grad: T")
    .Input("ksize: int32")
    .Input("strides: int32")
    .Output("output: T")
    .Attr("T: realnumbertype = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) {
      return UnchangedShapeWithRank(c, 4);
    });

REGISTER_OP("MaxPoolGradGrad")
    .Attr("ksize: list(int) >= 4")
    .Attr("strides: list(int) >= 4")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Input("orig_input: T")
    .Input("orig_output: T")
    .Input("grad: T")
    .Output("output: T")
    .Attr("T: realnumbertype")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::MaxPoolShape(c));
      ShapeHandle unused;
      // Validate 'orig_input' is the same shape as 'grad'
      TF_RETURN_IF_ERROR(c->Merge(c->input(0), c->input(2), &unused));
      // Validate 'orig_output' is same shape as 'output'
      TF_RETURN_IF_ERROR(c->Merge(c->input(1), c->output(0), &unused));
      return Status::OK();
    });

REGISTER_OP("MaxPoolGradGradV2")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Input("orig_input: T")
    .Input("orig_output: T")
    .Input("grad: T")
    .Input("ksize: int32")
    .Input("strides: int32")
    .Output("output: T")
    .Attr("T: realnumbertype")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::MaxPoolV2Shape(c, 5));
      ShapeHandle unused;
      // Validate 'orig_input' is the same shape as 'grad'
      TF_RETURN_IF_ERROR(c->Merge(c->input(0), c->input(2), &unused));
      // Validate 'orig_output' is same shape as 'output'
      TF_RETURN_IF_ERROR(c->Merge(c->input(1), c->output(0), &unused));
      return Status::OK();
    });

REGISTER_OP("MaxPoolWithArgmax")
    .Attr("ksize: list(int) >= 4")
    .Attr("strides: list(int) >= 4")
    .Attr("Targmax: {int32, int64} = DT_INT64")
    .Attr(GetPaddingAttrString())
    .Input("input: T")
    .Output("output: T")
    .Output("argmax: Targmax")
    .Attr("T: realnumbertype")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::MaxPoolShape(c));
      c->set_output(1, c->output(0));
      return Status::OK();
    });

REGISTER_OP("MaxPoolGradWithArgmax")
    .Attr("ksize: list(int) >= 4")
    .Attr("strides: list(int) >= 4")
    .Attr(GetPaddingAttrString())
    .Attr("Targmax: {int32, int64}")
    .Input("input: T")
    .Input("grad: T")
    .Input("argmax: Targmax")
    .Output("output: T")
    .Attr("T: realnumbertype")
    .SetShapeFn([](InferenceContext* c) {
      return UnchangedShapeWithRank(c, 4);
    });

REGISTER_OP("MaxPoolGradGradWithArgmax")
    .Attr("ksize: list(int) >= 4")
    .Attr("strides: list(int) >= 4")
    .Attr(GetPaddingAttrString())
    .Attr("Targmax: {int32, int64}")
    .Input("input: T")
    .Input("grad: T")
    .Input("argmax: Targmax")
    .Output("output: T")
    .Attr("T: realnumbertype")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::MaxPoolShape(c));
      ShapeHandle unused;
      // Validate 'orig_input' is the same shape as 'grad'
      TF_RETURN_IF_ERROR(c->Merge(c->input(0), c->input(1), &unused));
      // Validate 'argmax' is same shape as 'output'
      TF_RETURN_IF_ERROR(c->Merge(c->input(2), c->output(0), &unused));
      return Status::OK();
    });

// --------------------------------------------------------------------------

REGISTER_OP("Dilation2D")
    .Input("input: T")
    .Input("filter: T")
    .Output("output: T")
    .Attr("T: realnumbertype")
    .Attr("strides: list(int) >= 4")
    .Attr("rates: list(int) >= 4")
    .Attr(GetPaddingAttrString())
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input_shape));
      ShapeHandle filter_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &filter_shape));

      std::vector<int32> strides;
      TF_RETURN_IF_ERROR(c->GetAttr("strides", &strides));
      if (strides.size() != 4) {
        return errors::InvalidArgument(
            "Dilation2D requires the stride attribute to contain 4 values, but "
            "got: ",
            strides.size());
      }

      std::vector<int32> rates;
      TF_RETURN_IF_ERROR(c->GetAttr("rates", &rates));
      if (rates.size() != 4) {
        return errors::InvalidArgument(
            "Dilation2D requires the rates attribute to contain 4 values, but "
            "got: ",
            rates.size());
      }

      int32 stride_rows = strides[1];
      int32 stride_cols = strides[2];

      int32 rate_rows = rates[1];
      int32 rate_cols = rates[2];

      DimensionHandle batch_size_dim = c->Dim(input_shape, 0);
      DimensionHandle in_rows_dim = c->Dim(input_shape, 1);
      DimensionHandle in_cols_dim = c->Dim(input_shape, 2);
      DimensionHandle filter_rows_dim = c->Dim(filter_shape, 0);
      DimensionHandle filter_cols_dim = c->Dim(filter_shape, 1);
      DimensionHandle output_depth_dim = c->Dim(filter_shape, 2);

      if (!c->ValueKnown(in_rows_dim) || !c->ValueKnown(in_cols_dim) ||
          !c->ValueKnown(filter_rows_dim) || !c->ValueKnown(filter_cols_dim)) {
        ShapeHandle output_shape =
            c->MakeShape({batch_size_dim, InferenceContext::kUnknownDim,
                          InferenceContext::kUnknownDim, output_depth_dim});
        c->set_output(0, output_shape);
        return Status::OK();
      }
      DimensionHandle unused;
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(input_shape, 3), output_depth_dim, &unused));

      auto in_rows = c->Value(in_rows_dim);
      auto in_cols = c->Value(in_cols_dim);
      auto filter_rows = c->Value(filter_rows_dim);
      auto filter_cols = c->Value(filter_cols_dim);
      auto filter_rows_eff = filter_rows + (filter_rows - 1) * (rate_rows - 1);
      auto filter_cols_eff = filter_cols + (filter_cols - 1) * (rate_cols - 1);

      Padding padding;
      TF_RETURN_IF_ERROR(c->GetAttr("padding", &padding));

      int64 output_rows, output_cols;
      int64 padding_before, padding_after;
      TF_RETURN_IF_ERROR(GetWindowedOutputSizeVerbose(
          in_rows, filter_rows_eff, stride_rows, padding, &output_rows,
          &padding_before, &padding_after));
      TF_RETURN_IF_ERROR(GetWindowedOutputSizeVerbose(
          in_cols, filter_cols_eff, stride_cols, padding, &output_cols,
          &padding_before, &padding_after));

      ShapeHandle output_shape = c->MakeShape(
          {batch_size_dim, output_rows, output_cols, output_depth_dim});
      c->set_output(0, output_shape);
      return Status::OK();
    });

REGISTER_OP("Dilation2DBackpropInput")
    .Input("input: T")
    .Input("filter: T")
    .Input("out_backprop: T")
    .Output("in_backprop: T")
    .Attr("T: realnumbertype")
    .Attr("strides: list(int) >= 4")
    .Attr("rates: list(int) >= 4")
    .Attr(GetPaddingAttrString())
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("Dilation2DBackpropFilter")
    .Input("input: T")
    .Input("filter: T")
    .Input("out_backprop: T")
    .Output("filter_backprop: T")
    .Attr("T: realnumbertype")
    .Attr("strides: list(int) >= 4")
    .Attr("rates: list(int) >= 4")
    .Attr(GetPaddingAttrString())
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(1));
      return Status::OK();
    });

// --------------------------------------------------------------------------

REGISTER_OP("Relu")
    .Input("features: T")
    .Output("activations: T")
    .Attr("T: realnumbertype")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("ReluGrad")
    .Input("gradients: T")
    .Input("features: T")
    .Output("backprops: T")
    .Attr("T: realnumbertype")
    .SetShapeFn(shape_inference::MergeBothInputsShapeFn);

REGISTER_OP("Relu6")
    .Input("features: T")
    .Output("activations: T")
    .Attr("T: realnumbertype")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("Relu6Grad")
    .Input("gradients: T")
    .Input("features: T")
    .Output("backprops: T")
    .Attr("T: realnumbertype")
    .SetShapeFn(shape_inference::MergeBothInputsShapeFn);

REGISTER_OP("Elu")
    .Input("features: T")
    .Output("activations: T")
    .Attr("T: {half, bfloat16, float, double}")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("EluGrad")
    .Input("gradients: T")
    .Input("outputs: T")
    .Output("backprops: T")
    .Attr("T: {half, bfloat16, float, double}")
    .SetShapeFn(shape_inference::MergeBothInputsShapeFn);

REGISTER_OP("Selu")
    .Input("features: T")
    .Output("activations: T")
    .Attr("T: {half, bfloat16, float, double}")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("SeluGrad")
    .Input("gradients: T")
    .Input("outputs: T")
    .Output("backprops: T")
    .Attr("T: {half, bfloat16, float, double}")
    .SetShapeFn(shape_inference::MergeBothInputsShapeFn);

REGISTER_OP("Softplus")
    .Input("features: T")
    .Output("activations: T")
    .Attr("T: realnumbertype")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("SoftplusGrad")
    .Input("gradients: T")
    .Input("features: T")
    .Output("backprops: T")
    .Attr("T: realnumbertype")
    .SetShapeFn(shape_inference::MergeBothInputsShapeFn);

REGISTER_OP("Softsign")
    .Input("features: T")
    .Output("activations: T")
    .Attr("T: realnumbertype")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("SoftsignGrad")
    .Input("gradients: T")
    .Input("features: T")
    .Output("backprops: T")
    .Attr("T: realnumbertype")
    .SetShapeFn(shape_inference::MergeBothInputsShapeFn);

// --------------------------------------------------------------------------

REGISTER_OP("Softmax")
    .Input("logits: T")
    .Output("softmax: T")
    .Attr("T: {half, bfloat16, float, double}")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 1);
    });

// --------------------------------------------------------------------------

REGISTER_OP("LogSoftmax")
    .Input("logits: T")
    .Output("logsoftmax: T")
    .Attr("T: {half, bfloat16, float, double}")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 1);
    });

// --------------------------------------------------------------------------

REGISTER_OP("SoftmaxCrossEntropyWithLogits")
    .Input("features: T")
    .Input("labels: T")
    .Output("loss: T")
    .Output("backprop: T")
    .Attr("T: {half, bfloat16, float, double}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input;
      if (c->WithRank(c->input(0), 2, &input) == Status::OK() &&
          c->Merge(input, c->input(1), &input) == Status::OK()) {
        DimensionHandle batch_size = c->Dim(input, 0);
        c->set_output(0, c->Vector(batch_size));
        c->set_output(1, input);
        return Status::OK();
      }
      TF_RETURN_IF_ERROR(BroadcastBinaryOpOutputShapeFn(c, 1));

      if (!c->RankKnown(c->output(1))) {
        return errors::InvalidArgument(
            "Shape must be broadcasted with rank 2, but is rank is unknown.");
      }

      if (c->Rank(c->output(1)) != 2) {
        return errors::InvalidArgument(
            "Shape must be broadcasted with rank 2, but is rank ",
            c->Rank(c->output(1)));
      }
      DimensionHandle batch_size = c->Dim(c->output(1), 0);
      c->set_output(0, c->Vector(batch_size));
      return Status::OK();
    });

REGISTER_OP("SparseSoftmaxCrossEntropyWithLogits")
    .Input("features: T")
    .Input("labels: Tlabels")
    .Output("loss: T")
    .Output("backprop: T")
    .Attr("T: {half, bfloat16, float, double}")
    .Attr("Tlabels: {int32, int64} = DT_INT64")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle features;
      ShapeHandle labels;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &features));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &labels));

      DimensionHandle batch_size;
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(features, 0), c->Dim(labels, 0), &batch_size));
      TF_RETURN_IF_ERROR(c->ReplaceDim(features, 0, batch_size, &features));

      c->set_output(0, c->Vector(batch_size));
      c->set_output(1, features);
      return Status::OK();
    });

// --------------------------------------------------------------------------

REGISTER_OP("InTopK")
    .Input("predictions: float")
    .Input("targets: T")
    .Output("precision: bool")
    .Attr("k: int")
    .Attr("T: {int32, int64} = DT_INT32")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle predictions;
      ShapeHandle targets;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &predictions));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &targets));
      DimensionHandle batch_size;
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(predictions, 0), c->Dim(targets, 0), &batch_size));
      c->set_output(0, c->Vector(batch_size));
      return Status::OK();
    });

// This is the same as `InTopK`, but takes `k` as in input rather than an attr.
REGISTER_OP("InTopKV2")
    .Input("predictions: float")
    .Input("targets: T")
    .Input("k: T")
    .Output("precision: bool")
    .Attr("T: {int32, int64} = DT_INT32")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle predictions;
      ShapeHandle targets;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &predictions));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &targets));
      DimensionHandle batch_size;
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(predictions, 0), c->Dim(targets, 0), &batch_size));
      c->set_output(0, c->Vector(batch_size));
      return Status::OK();
    });

namespace {

Status TopKShapeFn(InferenceContext* c) {
  ShapeHandle input;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &input));

  // Get the k value, either from input tensor or attribute.
  DimensionHandle k_dim;
  if (c->num_inputs() >= 2) {
    TF_RETURN_IF_ERROR(c->MakeDimForScalarInput(1, &k_dim));
  } else {
    int32 k;
    TF_RETURN_IF_ERROR(c->GetAttr("k", &k));
    if (k < 0) {
      return errors::InvalidArgument("Need k >= 0, got ", k);
    }
    k_dim = c->MakeDim(k);
  }

  DimensionHandle last_dim = c->Dim(input, -1);
  if (c->ValueKnown(last_dim) && c->ValueKnown(k_dim) &&
      c->Value(last_dim) < c->Value(k_dim)) {
    return errors::InvalidArgument("input must have last dimension >= k = ",
                                   c->Value(k_dim), " but is ",
                                   c->Value(last_dim));
  }

  // Replace last_dim with k_dim.
  ShapeHandle s;
  TF_RETURN_IF_ERROR(c->Subshape(input, 0, -1, &s));
  TF_RETURN_IF_ERROR(c->Concatenate(s, c->Vector(k_dim), &s));
  c->set_output(0, s);
  c->set_output(1, s);
  return Status::OK();
}

}  // namespace

REGISTER_OP("TopK")
    .Input("input: T")
    .Output("values: T")
    .Output("indices: int32")
    .Attr("k: int >= 0")
    .Attr("sorted: bool = true")
    .Attr("T: realnumbertype")
    .Deprecated(7, "Use TopKV2 instead")
    .SetShapeFn(TopKShapeFn);

// This is the same as `TopK`, but takes `k` as in input rather than an attr.
REGISTER_OP("TopKV2")
    .Input("input: T")
    .Input("k: int32")
    .Output("values: T")
    .Output("indices: int32")
    .Attr("sorted: bool = true")
    .Attr("T: realnumbertype")
    .SetShapeFn(TopKShapeFn);

// --------------------------------------------------------------------------

REGISTER_OP("NthElement")
    .Input("input: T")
    .Input("n: int32")
    .Output("values: T")
    .Attr("reverse: bool = false")
    .Attr("T: realnumbertype")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &input));

      // Get the n value from input tensor, and make sure which is a scalar.
      DimensionHandle n_dim;
      TF_RETURN_IF_ERROR(c->MakeDimForScalarInput(1, &n_dim));

      // The last dimension of input tensor must be greater than N.
      DimensionHandle last_dim = c->Dim(input, -1);
      if (c->ValueKnown(last_dim) && c->ValueKnown(n_dim) &&
          c->Value(last_dim) <= c->Value(n_dim)) {
        return errors::InvalidArgument("Input must have last dimension > n = ",
                                       c->Value(n_dim), " but is ",
                                       c->Value(last_dim));
      }

      // Reduce last_dim for output tensor
      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->Subshape(input, 0, -1, &s));
      c->set_output(0, s);
      return Status::OK();
    });

// --------------------------------------------------------------------------

REGISTER_OP("FractionalMaxPool")
    .Input("value: T")
    .Output("output: T")
    .Output("row_pooling_sequence: int64")
    .Output("col_pooling_sequence: int64")
    .Attr("pooling_ratio: list(float) >=4")
    .Attr("pseudo_random: bool = false")
    .Attr("overlapping: bool = false")
    .Attr("deterministic: bool = false")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Attr("T: {float, double, int32, int64}")
    .SetShapeFn(FractionalPoolShapeFn);

REGISTER_OP("FractionalMaxPoolGrad")
    .Input("orig_input: T")
    .Input("orig_output: T")
    .Input("out_backprop: T")
    .Input("row_pooling_sequence: int64")
    .Input("col_pooling_sequence: int64")
    .Output("output: T")
    .Attr("overlapping: bool = false")
    .Attr("T: {float, double, int32, int64}")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRank(c, 4);
    });

// --------------------------------------------------------------------------

REGISTER_OP("FractionalAvgPool")
    .Input("value: T")
    .Output("output: T")
    .Output("row_pooling_sequence: int64")
    .Output("col_pooling_sequence: int64")
    .Attr("pooling_ratio: list(float) >=4")
    .Attr("pseudo_random: bool = false")
    .Attr("overlapping: bool = false")
    .Attr("deterministic: bool = false")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Attr("T: {float, double, int32, int64}")
    .SetShapeFn(FractionalPoolShapeFn);

REGISTER_OP("FractionalAvgPoolGrad")
    .Input("orig_input_tensor_shape: int64")
    .Input("out_backprop: T")
    .Input("row_pooling_sequence: int64")
    .Input("col_pooling_sequence: int64")
    .Output("output: T")
    .Attr("overlapping: bool = false")
    .Attr("T: {float, double, int32, int64}")
    .SetShapeFn([](InferenceContext* c) {
      if (c->input_tensor(0) != nullptr) {
        ShapeHandle out;
        TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &out));
        c->set_output(0, out);
      } else {
        c->set_output(0, c->UnknownShapeOfRank(4));
      }
      return Status::OK();
    });

REGISTER_OP("QuantizedAvgPool")
    .Input("input: T")
    .Input("min_input: float")
    .Input("max_input: float")
    .Output("output: T")
    .Output("min_output: float")
    .Output("max_output: float")
    .Attr("T: quantizedtype")
    .Attr("ksize: list(int)")
    .Attr("strides: list(int)")
    .Attr(GetPaddingAttrString())
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::AvgPoolShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("QuantizedBiasAdd")
    .Input("input: T1")
    .Input("bias: T2")
    .Input("min_input: float")
    .Input("max_input: float")
    .Input("min_bias: float")
    .Input("max_bias: float")
    .Output("output: out_type")
    .Output("min_out: float")
    .Output("max_out: float")
    .Attr("T1: quantizedtype")
    .Attr("T2: quantizedtype")
    .Attr("out_type: quantizedtype")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::BiasAddShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("QuantizedConv2D")
    .Input("input: Tinput")
    .Input("filter: Tfilter")
    .Input("min_input: float")
    .Input("max_input: float")
    .Input("min_filter: float")
    .Input("max_filter: float")
    .Output("output: out_type")
    .Output("min_output: float")
    .Output("max_output: float")
    .Attr("Tinput: quantizedtype")
    .Attr("Tfilter: quantizedtype")
    .Attr("out_type: quantizedtype = DT_QINT32")
    .Attr("strides: list(int)")
    .Attr(GetPaddingAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::Conv2DShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("QuantizedMaxPool")
    .Input("input: T")
    .Input("min_input: float")
    .Input("max_input: float")
    .Output("output: T")
    .Output("min_output: float")
    .Output("max_output: float")
    .Attr("T: quantizedtype")
    .Attr("ksize: list(int)")
    .Attr("strides: list(int)")
    .Attr(GetPaddingAttrString())
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::MaxPoolShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("QuantizedRelu")
    .Input("features: Tinput")
    .Input("min_features: float")
    .Input("max_features: float")
    .Output("activations: out_type")
    .Output("min_activations: float")
    .Output("max_activations: float")
    .Attr("Tinput: quantizedtype")
    .Attr("out_type: quantizedtype = DT_QUINT8")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::UnchangedShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("QuantizedRelu6")
    .Input("features: Tinput")
    .Input("min_features: float")
    .Input("max_features: float")
    .Output("activations: out_type")
    .Output("min_activations: float")
    .Output("max_activations: float")
    .Attr("Tinput: quantizedtype")
    .Attr("out_type: quantizedtype = DT_QUINT8")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::UnchangedShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("QuantizedReluX")
    .Input("features: Tinput")
    .Input("max_value: float")
    .Input("min_features: float")
    .Input("max_features: float")
    .Output("activations: out_type")
    .Output("min_activations: float")
    .Output("max_activations: float")
    .Attr("Tinput: quantizedtype")
    .Attr("out_type: quantizedtype = DT_QUINT8")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::UnchangedShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("QuantizedBatchNormWithGlobalNormalization")
    .Input("t: Tinput")
    .Input("t_min: float")
    .Input("t_max: float")
    .Input("m: Tinput")
    .Input("m_min: float")
    .Input("m_max: float")
    .Input("v: Tinput")
    .Input("v_min: float")
    .Input("v_max: float")
    .Input("beta: Tinput")
    .Input("beta_min: float")
    .Input("beta_max: float")
    .Input("gamma: Tinput")
    .Input("gamma_min: float")
    .Input("gamma_max: float")
    .Output("result: out_type")
    .Output("result_min: float")
    .Output("result_max: float")
    .Attr("Tinput: quantizedtype")
    .Attr("out_type: quantizedtype")
    .Attr("variance_epsilon: float")
    .Attr("scale_after_normalization: bool")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input));

      DimensionHandle last_dim = c->Dim(input, 3);
      for (int i = 1; i < 5; ++i) {  // covers m, v, beta, gamma
        ShapeHandle vec;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i * 3), 1, &vec));
        TF_RETURN_IF_ERROR(c->Merge(last_dim, c->Dim(vec, 0), &last_dim));
      }

      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->ReplaceDim(input, 3, last_dim, &out));
      c->set_output(0, out);
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());

      return Status::OK();
    });

#ifdef INTEL_MKL
REGISTER_OP("_MklConv2D")
    .Input("input: T")
    .Input("filter: T")
    .Input("mkl_input: uint8")
    .Input("mkl_filter: uint8")
    .Output("output: T")
    .Output("filter_output: T")
    .Output("mkl_output: uint8")
    .Output("mkl_filter_output: uint8")
    .Attr("T: {half, float, double}")
    .Attr("strides: list(int)")
    .Attr("use_cudnn_on_gpu: bool = true")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn(shape_inference::Conv2DShape)
    .Doc(R"doc(
MKL version of Conv2D operator. Uses MKL DNN APIs to perform 2D convolution.

NOTE Do not invoke this operator directly in Python. Graph rewrite pass is
expected to invoke these operators.
)doc");

REGISTER_OP("__MklDummyConv2DWithBias")
    .Input("input: T")
    .Input("filter: T")
    .Input("bias: T")
    .Output("output: T")
    .Attr("T: {half, float, double}")
    .Attr("strides: list(int)")
    .Attr("use_cudnn_on_gpu: bool = true")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn(shape_inference::Conv2DShape)
    .Doc(R"doc(
Dummy node that enables fusing Conv2D and BiasAdd operator for MKL. This node
does not perform anything. It is just created as an intermediate output of
merging Conv2D and BiasAdd.

NOTE Do not invoke this operator directly in Python. Graph rewrite pass is
expected to invoke these operators.
)doc");

REGISTER_OP("_MklConv2DWithBias")
    .Input("input: T")
    .Input("filter: T")
    .Input("bias: T")
    .Input("mkl_input: uint8")
    .Input("mkl_filter: uint8")
    .Input("mkl_bias: uint8")
    .Output("output: T")
    .Output("filter_output: T")
    .Output("mkl_output: uint8")
    .Output("mkl_filter_output: uint8")
    .Attr("T: {half, float, double}")
    .Attr("strides: list(int)")
    .Attr("use_cudnn_on_gpu: bool = true")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn(shape_inference::Conv2DShape)
    .Doc(R"doc(
MKL version of Conv2D and BiasAdd operator. Uses MKL DNN APIs to perform
2D convolution and add Bias to the output of convolution.

NOTE Do not invoke this operator directly in Python. Graph rewrite pass is
expected to invoke these operators.
)doc");

REGISTER_OP("_MklConv2DBackpropFilter")
    .Input("input: T")
    .Input("filter_sizes: int32")
    .Input("out_backprop: T")
    .Input("mkl_input: uint8")
    .Input("mkl_filter_size: uint8")
    .Input("mkl_out_backprop: uint8")
    .Output("output: T")
    .Output("mkl_output: uint8")
    .Attr("T: {half, float, double}")
    .Attr("strides: list(int)")
    .Attr("use_cudnn_on_gpu: bool = true")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(1, &s));
      TF_RETURN_IF_ERROR(c->WithRank(s, 4, &s));
      c->set_output(0, s);
      return Status::OK();
    })
    .Doc(R"doc(
MKL version of Conv2DBackpropFilter. Uses MKL DNN APIs to compute the
gradients of convolution with respect to the filter.

NOTE Do not invoke this operator directly in Python. Graph rewrite pass is
expected to invoke these operators.
)doc");

REGISTER_OP("__MklDummyConv2DBackpropFilterWithBias")
    .Input("input: T")
    .Input("filter_sizes: int32")
    .Input("out_backprop: T")
    .Output("output: T")
    .Output("bias_grad: T")
    .Attr("T: {half, float, double}")
    .Attr("strides: list(int)")
    .Attr("use_cudnn_on_gpu: bool = true")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input_shape;
      // Fetch the data_format attribute, which may not exist.
      string data_format;
      Status s = c->GetAttr("data_format", &data_format);

      if (s.ok() && data_format == "NCHW") {
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input_shape));
        c->set_output(1, c->Vector(c->Dim(input_shape, -3)));
      } else {
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input_shape));
        c->set_output(1, c->Vector(c->Dim(input_shape, -1)));
      }
      ShapeHandle sh;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(1, &sh));
      TF_RETURN_IF_ERROR(c->WithRank(sh, 4, &sh));
      c->set_output(0, sh);
      return Status::OK();
    })
    .Doc(R"doc(
Dummy node that enables fusing Conv2DBackpropFilter and BiasAddGrad operator
for MKL. This node does not perform anything. It is just created as an
intermediate output of merging Conv2DBackpropFilter and BiasAddGrad.

NOTE Do not invoke this operator directly in Python. Graph rewrite pass is
expected to invoke these operators.
)doc");

REGISTER_OP("_MklConv2DBackpropFilterWithBias")
    .Input("input: T")
    .Input("filter_sizes: int32")
    .Input("out_backprop: T")
    .Input("mkl_input: uint8")
    .Input("mkl_filter_size: uint8")
    .Input("mkl_out_backprop: uint8")
    .Output("output: T")
    .Output("bias_grad: T")
    .Output("mkl_output: uint8")
    .Output("mkl_bias_grad: uint8")
    .Attr("T: {half, float, double}")
    .Attr("strides: list(int)")
    .Attr("use_cudnn_on_gpu: bool = true")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input_shape;
      // Fetch the data_format attribute, which may not exist.
      string data_format;
      Status s = c->GetAttr("data_format", &data_format);

      if (s.ok() && data_format == "NCHW") {
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input_shape));
        c->set_output(1, c->Vector(c->Dim(input_shape, -3)));
      } else {
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input_shape));
        c->set_output(1, c->Vector(c->Dim(input_shape, -1)));
      }
      ShapeHandle sh;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(1, &sh));
      TF_RETURN_IF_ERROR(c->WithRank(sh, 4, &sh));
      c->set_output(0, sh);
      return Status::OK();
    })
    .Doc(R"doc(
MKL version of Conv2DBackpropFilterWithBias. Uses MKL DNN APIs to compute the
gradients of convolution with respect to the filter.

NOTE Do not invoke this operator directly in Python. Graph rewrite pass is
expected to invoke these operators.
)doc");

#ifdef INTEL_MKL_ML
REGISTER_OP("_MklConv2DWithBiasBackpropBias")
    .Input("out_backprop: T")
    .Input("mkl_out_backprop: uint8")
    .Output("output: T")
    .Output("mkl_output: uint8")
    .Attr("T: {half, float, double}")
    .Attr("strides: list(int)")
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .Doc(R"doc(
MKL version of Conv2DBackpropBias. Uses MKL DNN APIs to compute the
gradients of convolution with respect to the bias.

NOTE Do not invoke this operator directly in Python. Graph rewrite pass is
expected to invoke these operators.
)doc");
#endif

REGISTER_OP("_MklConv2DBackpropInput")
    .Input("input_sizes: int32")
    .Input("filter: T")
    .Input("out_backprop: T")
    .Input("mkl_input_sizes: uint8")
    .Input("mkl_filter: uint8")
    .Input("mkl_out_backprop: uint8")
    .Output("output: T")
    .Output("mkl_output: uint8")
    .Attr("T: {half, float, double}")
    .Attr("strides: list(int)")
    .Attr("use_cudnn_on_gpu: bool = true")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &s));
      TF_RETURN_IF_ERROR(c->WithRank(s, 4, &s));
      c->set_output(0, s);
      return Status::OK();
    })
    .Doc(R"doc(
MKL version of Convolution2D backward input. Uses MKL DNN APIs to compute the
gradients of convolution with respect to the input.

NOTE Do not invoke this operator directly in Python. Graph rewrite pass is
expected to invoke these operators.
)doc");

REGISTER_OP("_MklRelu")
    .Input("features: T")
    .Input("mkl_features: uint8")
    .Output("activations: T")
    .Output("mkl_activations: uint8")
    .Attr("T: realnumbertype")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
MKL version of Relu operator. Uses MKL DNN APIs to implement Relu operator.

NOTE Do not invoke this operator directly in Python. Graph rewrite pass is
expected to invoke these operators.
)doc");

REGISTER_OP("_MklReluGrad")
    .Input("gradients: T")
    .Input("features: T")
    .Input("mkl_gradients: uint8")
    .Input("mkl_features: uint8")
    .Output("backprops: T")
    .Output("mkl_backprops: uint8")
    .Attr("T: realnumbertype")
    .SetShapeFn(shape_inference::MergeBothInputsShapeFn)
    .Doc(R"doc(
MKL version of ReluGrad operator. Uses MKL DNN APIs to compute rectified
linear gradients for Relu operation.

NOTE Do not invoke this operator directly in Python. Graph rewrite pass is
expected to invoke these operators.
)doc");

REGISTER_OP("_MklElu")
    .Input("features: T")
    .Input("mkl_features: uint8")
    .Output("activations: T")
    .Output("mkl_activations: uint8")
    .Attr("T: realnumbertype")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
MKL version of Elu operator. Uses MKL DNN APIs to implement Elu operator.
NOTE Do not invoke this operator directly in Python. Graph rewrite pass is
expected to invoke these operators.
)doc");

REGISTER_OP("_MklEluGrad")
    .Input("gradients: T")
    .Input("features: T")
    .Input("mkl_gradients: uint8")
    .Input("mkl_features: uint8")
    .Output("backprops: T")
    .Output("mkl_backprops: uint8")
    .Attr("T: realnumbertype")
    .SetShapeFn(shape_inference::MergeBothInputsShapeFn)
    .Doc(R"doc(
MKL version of EluGrad operator. Uses MKL DNN APIs to compute Elu
gradients for Elu operation.
NOTE Do not invoke this operator directly in Python. Graph rewrite pass is
expected to invoke these operators.
)doc");

REGISTER_OP("_MklSoftmax")
    .Input("logits: T")
    .Input("mkl_logits: uint8")
    .Output("softmax: T")
    .Output("mkl_softmax: uint8")
    .Attr("T: {half, float, double}")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 1);
    })
    .Doc(R"doc(
MKL version of ReluGrad operator. Uses MKL DNN APIs to compute rectified
linear gradients for Relu operation.
)doc");

REGISTER_OP("_MklTanh")
    .Input("features: T")
    .Input("mkl_features: uint8")
    .Output("activations: T")
    .Output("mkl_activations: uint8")
    .Attr("T: realnumbertype")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
MKL version of Tanh operator. Uses MKL DNN APIs to implement Tanh operator.
NOTE Do not invoke this operator directly in Python. Graph rewrite pass is
expected to invoke these operators.
)doc");

REGISTER_OP("_MklTanhGrad")
    .Input("gradients: T")
    .Input("features: T")
    .Input("mkl_gradients: uint8")
    .Input("mkl_features: uint8")
    .Output("backprops: T")
    .Output("mkl_backprops: uint8")
    .Attr("T: realnumbertype")
    .SetShapeFn(shape_inference::MergeBothInputsShapeFn)
    .Doc(R"doc(
MKL version of TanhGrad operator. Uses MKL DNN APIs to compute tanh
gradients for Tanh operation.
NOTE Do not invoke this operator directly in Python. Graph rewrite pass is
expected to invoke these operators.
)doc");

REGISTER_OP("_MklMaxPool")
    .Attr("T: {float, half} = DT_FLOAT")
    .Attr("ksize: list(int) >= 4")
    .Attr("strides: list(int) >= 4")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("workspace_enabled: bool = false")
    .Input("input: T")
    .Input("mkl_input: uint8")
    .Output("output: T")
#ifdef INTEL_MKL_ML
    .Output("workspace: T")
#else
    .Output("workspace: uint8")
#endif
    .Output("mkl_output: uint8")
    .Output("mkl_workspace: uint8")
    .SetShapeFn(shape_inference::MaxPoolShape)
    .Doc(R"doc(
MKL version of MaxPool operator. Uses MKL DNN APIs to perform max pooling
on the input.

NOTE Do not invoke this operator directly in Python. Graph rewrite pass is
expected to invoke these operators.
)doc");

REGISTER_OP("_MklMaxPoolGrad")
    .Attr("T: {float, half} = DT_FLOAT")
    .Attr("ksize: list(int) >= 4")
    .Attr("strides: list(int) >= 4")
    .Attr("workspace_enabled: bool = false")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Input("orig_input: T")
    .Input("orig_output: T")
    .Input("grad: T")
#ifdef INTEL_MKL_ML
    .Input("workspace: T")
#else
    .Input("workspace: uint8")
#endif
    .Input("mkl_orig_input: uint8")
    .Input("mkl_orig_output: uint8")
    .Input("mkl_grad: uint8")
    .Input("mkl_workspace: uint8")
    .Output("output: T")
    .Output("mkl_output: uint8")
    .SetShapeFn([](InferenceContext* c) {
      return UnchangedShapeWithRank(c, 4);
    })
    .Doc(R"doc(
MKL version of MaxPoolGrad. Uses MKL DNN APIs to compute gradients of
MaxPool operator.

NOTE Do not invoke this operator directly in Python. Graph rewrite pass is
expected to invoke these operators.
)doc");

REGISTER_OP("_MklAvgPool")
    .Input("value: T")
    .Input("mkl_input: uint8")
    .Output("output: T")
    .Output("mkl_output: uint8")
    .Attr("ksize: list(int) >= 4")
    .Attr("strides: list(int) >= 4")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("T: {float, half, double}")
    .SetShapeFn(shape_inference::AvgPoolShape)
    .Doc(R"doc(
MKL version of AvgPool operator. Uses MKL DNN APIs to perform average pooling
on the input.

NOTE Do not invoke this operator directly in Python. Graph rewrite pass is
expected to invoke these operators.
)doc");

REGISTER_OP("_MklAvgPoolGrad")
    .Input("orig_input_shape: int32")
    .Input("grad: T")
    .Input("mkl_orig_input: uint8")
    .Input("mkl_grad: uint8")
    .Output("output: T")
    .Output("mkl_output: uint8")
    .Attr("ksize: list(int) >= 4")
    .Attr("strides: list(int) >= 4")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("T: {float, half, double}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &s));
      TF_RETURN_IF_ERROR(c->WithRank(s, 4, &s));
      c->set_output(0, s);
      return Status::OK();
    })
    .Doc(R"doc(
MKL version of AvgPoolGrad operator. Uses MKL DNN APIs to compute gradients
of AvgPool function.

NOTE Do not invoke this operator directly in Python. Graph rewrite pass is
expected to invoke these operators.
)doc");

REGISTER_OP("_MklLRN")
    .Input("input: T")
    .Input("mkl_input: uint8")
    .Output("output: T")
#ifdef INTEL_MKL_ML
    .Output("workspace: T")
#else
    .Output("workspace: uint8")
#endif
    .Output("mkl_output: uint8")
    .Output("mkl_workspace: uint8")
    .Attr("depth_radius: int = 5")
    .Attr("bias: float = 1.0")
    .Attr("alpha: float = 1.0")
    .Attr("beta: float = 0.5")
    .Attr("workspace_enabled: bool = false")
    .Attr("T: {float, half} = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) {
      return UnchangedShapeWithRank(c, 4);
    })
    .Doc(R"doc(
MKL version of LRN operator. Uses MKL DNN APIs to perform local response
normalization.

NOTE Do not invoke this operator directly in Python. Graph rewrite pass is
expected to invoke these operators.
)doc");

REGISTER_OP("_MklLRNGrad")
    .Input("input_grads: T")
    .Input("input_image: T")
    .Input("output_image: T")
#ifdef INTEL_MKL_ML
    .Input("workspace: T")
#else
    .Input("workspace: uint8")
#endif
    .Input("mkl_input_grads: uint8")
    .Input("mkl_input_image: uint8")
    .Input("mkl_output_image: uint8")
    .Input("mkl_workspace: uint8")
    .Output("output: T")
    .Output("mkl_output: uint8")
    .Attr("depth_radius: int = 5")
    .Attr("bias: float = 1.0")
    .Attr("alpha: float = 1.0")
    .Attr("beta: float = 0.5")
    .Attr("workspace_enabled: bool = false")
    .Attr("T: {float, half} = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &s));  // input_grads
      TF_RETURN_IF_ERROR(c->Merge(s, c->input(1), &s));     // input_image
      TF_RETURN_IF_ERROR(c->Merge(s, c->input(2), &s));     // output_image
      c->set_output(0, s);
      return Status::OK();
    })
    .Doc(R"doc(
MKL version of LRNGrad operator. Uses MKL DNN APIs to compute gradient for
local response normalization.

NOTE Do not invoke this operator directly in Python. Graph rewrite pass is
expected to invoke these operators.
)doc");

REGISTER_OP("_MklFusedBatchNorm")
    .Input("x: T")
    .Input("scale: T")
    .Input("offset: T")
    .Input("mean: T")
    .Input("variance: T")
    .Input("mkl_x: uint8")
    .Input("mkl_scale: uint8")
    .Input("mkl_offset: uint8")
    .Input("mkl_mean: uint8")
    .Input("mkl_variance: uint8")
    .Output("y: T")
    .Output("batch_mean: T")
    .Output("batch_variance: T")
    .Output("reserve_space_1: T")
    .Output("reserve_space_2: T")
    .Output("mkl_y: uint8")
    .Output("mkl_batch_mean: uint8")
    .Output("mkl_batch_variance: uint8")
    .Output("mkl_reserve_space_1: uint8")
    .Output("mkl_reserve_space_2: uint8")
    .Attr("T: numbertype")
    .Attr("epsilon: float = 0.0001")
    .Attr("data_format: string = 'NHWC'")
    .Attr("is_training: bool = true")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle x;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &x));

      bool is_training;
      TF_RETURN_IF_ERROR(c->GetAttr("is_training", &is_training));
      int number_inputs = (is_training) ? 3 : 5;
      string data_format;
      TF_RETURN_IF_ERROR(c->GetAttr("data_format", &data_format));
      DimensionHandle channel_dim =
          (data_format == "NHWC") ? c->Dim(x, 3) : c->Dim(x, 1);

      // covers scale, offset, and if is_training is false, mean, variance
      for (int i = 1; i < number_inputs; ++i) {
        ShapeHandle vec;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 1, &vec));
        TF_RETURN_IF_ERROR(c->Merge(channel_dim, c->Dim(vec, 0), &channel_dim));
      }

      ShapeHandle y;
      if (data_format == "NHWC") {
        TF_RETURN_IF_ERROR(c->ReplaceDim(x, 3, channel_dim, &y));
      } else {
        TF_RETURN_IF_ERROR(c->ReplaceDim(x, 1, channel_dim, &y));
      }
      c->set_output(0, y);
      ShapeHandle vector_shape = c->Vector(channel_dim);
      c->set_output(1, vector_shape);
      c->set_output(2, vector_shape);
      c->set_output(3, vector_shape);
      c->set_output(4, vector_shape);
      return Status::OK();
    })
    .Doc(R"doc(
MKL version of FusedBatchNorm operator. Uses MKL DNN APIs to perform fused
batch normalization.

NOTE Do not invoke this operator directly in Python. Graph rewrite pass is
expected to invoke these operators.
)doc");

REGISTER_OP("_MklFusedBatchNormGrad")
    .Input("y_backprop: T")
    .Input("x: T")
    .Input("scale: T")
    .Input("reserve_space_1: T")
    .Input("reserve_space_2: T")
    .Input("mkl_y_backprop: uint8")
    .Input("mkl_x: uint8")
    .Input("mkl_scale: uint8")
    .Input("mkl_reserve_space_1: uint8")
    .Input("mkl_reserve_space_2: uint8")
    .Output("x_backprop: T")
    .Output("scale_backprop: T")
    .Output("offset_backprop: T")
    .Output("reserve_space_3: T")
    .Output("reserve_space_4: T")
    .Output("mkl_x_backprop: uint8")
    .Output("mkl_scale_backprop: uint8")
    .Output("mkl_offset_backprop: uint8")
    .Output("mkl_reserve_space_3: uint8")
    .Output("mkl_reserve_space_4: uint8")
    .Attr("T: numbertype")
    .Attr("epsilon: float = 0.0001")
    .Attr("data_format: string = 'NHWC'")
    .Attr("is_training: bool = true")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle y_backprop;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &y_backprop));
      ShapeHandle x;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &x));

      bool is_training;
      string data_format;
      TF_RETURN_IF_ERROR(c->GetAttr("is_training", &is_training));
      TF_RETURN_IF_ERROR(c->GetAttr("data_format", &data_format));
      DimensionHandle channel_dim = (data_format == "NHWC")
                                        ? c->Dim(y_backprop, 3)
                                        : c->Dim(y_backprop, 1);
      if (data_format == "NHWC") {
        TF_RETURN_IF_ERROR(c->Merge(channel_dim, c->Dim(x, 3), &channel_dim));
      } else {
        TF_RETURN_IF_ERROR(c->Merge(channel_dim, c->Dim(x, 1), &channel_dim));
      }

      // covers scale, mean (reserve_space_1), variance (reserve_space_2)
      for (int i = 2; i < 5; ++i) {
        ShapeHandle vec;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 1, &vec));
        TF_RETURN_IF_ERROR(c->Merge(channel_dim, c->Dim(vec, 0), &channel_dim));
      }

      ShapeHandle x_backprop;
      if (data_format == "NHWC") {
        TF_RETURN_IF_ERROR(
            c->ReplaceDim(y_backprop, 3, channel_dim, &x_backprop));
      } else {
        TF_RETURN_IF_ERROR(
            c->ReplaceDim(y_backprop, 1, channel_dim, &x_backprop));
      }
      c->set_output(0, x_backprop);
      c->set_output(1, c->Vector(channel_dim));
      c->set_output(2, c->Vector(channel_dim));
      // Set the correct shapes for reserve_spaces
      // so that gradients can be performed when
      // the op is in a symbolic condition.
      if (is_training) {
        c->set_output(3, c->Vector(0));
        c->set_output(4, c->Vector(0));
      } else {
        c->set_output(3, c->Vector(channel_dim));
        c->set_output(4, c->Vector(channel_dim));
      }
      return Status::OK();
    })
    .Doc(R"doc(
MKL version of FusedBatchNormGrad operator. Uses MKL DNN APIs to compute
gradients for fused batch normalization.

NOTE Do not invoke this operator directly in Python. Graph rewrite pass is
expected to invoke these operators.
)doc");

REGISTER_OP("_MklToTf")
    .Input("input: T")
    .Input("mkl_input: uint8")
    .Output("output: T")
    .Attr("T: {half, float, double}")
    .Attr(GetConvnetDataFormatAttrString())
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
MKL operator to convert a tensor from MKL layout to TensorFlow layout.

NOTE Do not invoke this operator directly in Python. Graph rewrite pass is
expected to invoke these operators.
)doc");

REGISTER_OP("_MklInputConversion")
    .Input("input_0: T")
    .Input("input_1: T")
    .Input("mkl_input_0: uint8")
    .Input("mkl_input_1: uint8")
    .Output("output_0: T")
    .Output("output_1: T")
    .Output("mkl_output_0: uint8")
    .Output("mkl_output_1: uint8")
    // All datatypes supported by element-wise ops
    .Attr(
        "T: {half, float, double, uint8, int8, uint16, int16, int32, int64, "
        "complex64, complex128}")
    .Attr(GetConvnetDataFormatAttrString())
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
MKL operator to process the inputs to an elementwise MKL op. Both inputs
need to be either in TF or in MKL format. This op is added before every
element-wise MKL op.

NOTE Do not invoke this operator directly in Python. Graph rewrite pass is
expected to invoke these operators.
)doc");
#endif  // INTEL_MKL

}  // namespace tensorflow
