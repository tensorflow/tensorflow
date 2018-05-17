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
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("RegexReplace")
    .Input("input: string")
    .Input("pattern: string")
    .Input("rewrite: string")
    .Output("output: string")
    .Attr("replace_global: bool = true")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      c->set_output(0, c->input(0));
      return Status::OK();
    });

REGISTER_OP("RegexFullMatch")
    .Input("input: string")
    .Input("pattern: string")
    .Output("output: bool")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      c->set_output(0, c->input(0));
      return Status::OK();
    });

REGISTER_OP("StringToHashBucketFast")
    .Input("input: string")
    .Output("output: int64")
    .Attr("num_buckets: int >= 1")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("StringToHashBucketStrong")
    .Input("input: string")
    .Output("output: int64")
    .Attr("num_buckets: int >= 1")
    .Attr("key: list(int)")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("StringToHashBucket")
    .Input("string_tensor: string")
    .Output("output: int64")
    .Attr("num_buckets: int >= 1")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("ReduceJoin")
    .Input("inputs: string")
    .Input("reduction_indices: int32")
    .Attr("keep_dims: bool = false")
    .Attr("separator: string = ''")
    .Output("output: string")
    .SetShapeFn(shape_inference::ReductionShape);

REGISTER_OP("AsString")
    .Input("input: T")
    .Output("output: string")
    .Attr("T: {int32, int64, complex64, float, double, bool, int8}")
    .Attr("precision: int = -1")
    .Attr("scientific: bool = false")
    .Attr("shortest: bool = false")
    .Attr("width: int = -1")
    .Attr("fill: string = ''")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("StringJoin")
    .Input("inputs: N * string")
    .Attr("N: int")
    .Attr("separator: string = ''")
    .Output("output: string")
    .SetShapeFn([](InferenceContext* c) {
      // If all inputs are scalars, then return a scalar.
      bool all_scalar = true;
      for (int i = 0; i < c->num_inputs(); ++i) {
        if (c->Rank(c->input(i)) != 0) all_scalar = false;
      }
      if (all_scalar) {
        c->set_output(0, c->Scalar());
        return Status::OK();
      }

      // At least one input is unknown or a scalar.
      // Merge the non-scalars to find the output shape.
      // Don't merge inputs with unknown rank, as they can actually be scalars
      // or the output shape.
      ShapeHandle out = c->UnknownShape();
      for (int i = 0; i < c->num_inputs(); ++i) {
        if (c->RankKnown(c->input(i)) && c->Rank(c->input(i)) != 0) {
          TF_RETURN_IF_ERROR(c->Merge(out, c->input(i), &out));
        }
      }
      c->set_output(0, out);
      return Status::OK();
    });

REGISTER_OP("StringSplit")
    .Input("input: string")
    .Input("delimiter: string")
    .Output("indices: int64")
    .Output("values: string")
    .Output("shape: int64")
    .Attr("skip_empty: bool = true")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));

      c->set_output(0, c->Matrix(InferenceContext::kUnknownDim, 2));
      c->set_output(1, c->Vector(InferenceContext::kUnknownDim));
      c->set_output(2, c->Vector(2));
      return Status::OK();
    });

REGISTER_OP("StringStrip")
    .Input("input: string")
    .Output("output: string")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("EncodeBase64")
    .Input("input: string")
    .Output("output: string")
    .Attr("pad: bool = false")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("DecodeBase64")
    .Input("input: string")
    .Output("output: string")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("Substr")
    .Input("input: string")
    .Input("pos: T")
    .Input("len: T")
    .Output("output: string")
    .Attr("T: {int32, int64}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle pos_shape = c->input(1);
      ShapeHandle len_shape = c->input(2);
      ShapeHandle unused;
      // Check that pos/len have same rank
      TF_RETURN_IF_ERROR(c->WithRank(pos_shape, c->Rank(len_shape), &unused));
      // Check that dimensions are equal
      for (int32 i = 0; i < c->Rank(pos_shape); ++i) {
        DimensionHandle pos_dim = c->Dim(pos_shape, i);
        DimensionHandle len_dim = c->Dim(len_shape, i);
        if (c->Value(pos_dim) != c->Value(len_dim)) {
          return errors::InvalidArgument(
              "pos and len shapes must match: ", c->DebugString(pos_shape),
              " vs. ", c->DebugString(len_shape));
        }
      }
      // c->input(0) is the ShapeHandle to input strings
      // BroadcastBinaryOpShapeFn infers shape from c->input(0) and c->input(1).
      return shape_inference::BroadcastBinaryOpShapeFn(c);
    });

}  // namespace tensorflow
