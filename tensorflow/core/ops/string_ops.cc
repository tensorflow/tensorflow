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

#include <string>
#include <vector>

#include "absl/strings/str_split.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace shape_inference {
class InferenceContext;
}  // namespace shape_inference

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

REGISTER_OP("StaticRegexReplace")
    .Input("input: string")
    .Attr("pattern: string")
    .Attr("rewrite: string")
    .Output("output: string")
    .Attr("replace_global: bool = true")
    .SetShapeFn(shape_inference::UnchangedShape);

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

REGISTER_OP("StaticRegexFullMatch")
    .Input("input: string")
    .Attr("pattern: string")
    .Output("output: bool")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("StringToHashBucketFast")
    .Input("input: string")
    .Output("output: int64")
    .Attr("num_buckets: int >= 1")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("_TensorToHashBucketFast")
    .Input("input: T")
    .Output("output: int64")
    .Attr("T: {int8, uint8, int16, uint16, int32, uint32, int64, uint64}")
    .Attr("num_buckets: int >= 1")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Internal operation which is a composition of converting the tensor to a string
tensor (AsString) and then calling hash functions (StringToHashBucketFast):
reserved for internal use.

Do not invoke this operator directly in Python. A fusion optimization is
expected to create these operators.
)doc");

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

REGISTER_OP("UnsortedSegmentJoin")
    .Input("inputs: string")
    .Input("segment_ids: Tindices")
    .Input("num_segments: Tnumsegments")
    .Attr("separator: string = ''")
    .Attr("Tindices: {int32,int64}")
    .Attr("Tnumsegments: {int32,int64} = DT_INT32")
    .Output("output: string")
    .SetShapeFn(shape_inference::UnsortedSegmentReductionShapeFn);

REGISTER_OP("AsString")
    .Input("input: T")
    .Output("output: string")
    .Attr("T: {realnumbertype, complex64, complex128, bool, variant}")
    .Attr("precision: int = -1")
    .Attr("scientific: bool = false")
    .Attr("shortest: bool = false")
    .Attr("width: int = -1")
    .Attr("fill: string = ''")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("StringFormat")
    .Input("inputs: T")
    .Output("output: string")
    .Attr("T: list(type) >= 0")
    .Attr("template: string = '%s'")
    .Attr("placeholder: string = '%s'")
    .Attr("summarize: int = 3")
    .SetShapeFn([](InferenceContext* c) {
      string template_;
      string placeholder;
      TF_RETURN_IF_ERROR(c->GetAttr("template", &template_));
      TF_RETURN_IF_ERROR(c->GetAttr("placeholder", &placeholder));

      std::vector<std::string> split_template;
      split_template = absl::StrSplit(template_, placeholder);
      int64_t num_placeholders = split_template.size() - 1;
      if (c->num_inputs() != num_placeholders) {
        return errors::InvalidArgument(strings::StrCat(
            "num placeholders in template and num inputs must match: ",
            num_placeholders, " vs. ", c->num_inputs()));
      }

      c->set_output(0, c->Scalar());
      return Status::OK();
    });

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

REGISTER_OP("StringSplitV2")
    .Input("input: string")
    .Input("sep: string")
    .Output("indices: int64")
    .Output("values: string")
    .Output("shape: int64")
    .Attr("maxsplit: int = -1")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));

      c->set_output(0, c->Matrix(InferenceContext::kUnknownDim, 2));
      c->set_output(1, c->Vector(InferenceContext::kUnknownDim));
      c->set_output(2, c->Vector(2));
      return Status::OK();
    });

REGISTER_OP("StringLower")
    .Input("input: string")
    .Output("output: string")
    .Attr("encoding: string =''")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("StringUpper")
    .Input("input: string")
    .Output("output: string")
    .Attr("encoding: string =''")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("StringStrip")
    .Input("input: string")
    .Output("output: string")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("StringLength")
    .Input("input: string")
    .Output("output: int32")
    .Attr("unit: {'BYTE', 'UTF8_CHAR'} = 'BYTE'")
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
    .Attr("unit: {'BYTE', 'UTF8_CHAR'} = 'BYTE'")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle pos_shape = c->input(1);
      ShapeHandle len_shape = c->input(2);
      ShapeHandle unused;
      // If len rank is known, check that pos and len have the same rank
      if (c->RankKnown(len_shape)) {
        TF_RETURN_IF_ERROR(c->WithRank(pos_shape, c->Rank(len_shape), &unused));
      }
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

REGISTER_OP("UnicodeScript")
    .Input("input: int32")
    .Output("output: int32")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("UnicodeEncode")
    .Input("input_values: int32")
    .Input("input_splits: Tsplits")
    .Attr("errors: {'ignore', 'replace', 'strict'} = 'replace'")
    .Attr("output_encoding: {'UTF-8', 'UTF-16-BE', 'UTF-32-BE'}")
    .Attr("replacement_char: int = 65533")  // 0xFFFD unicode replacement char
    .Attr("Tsplits: {int32, int64} = DT_INT64")
    .Output("output: string")
    .SetShapeFn([](InferenceContext* c) {
      // Check rank of inner values
      ShapeHandle input_inner_values_shape = c->input(0);
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(input_inner_values_shape, 1, &unused));

      // Check rank of input_splits
      ShapeHandle splits_shape = c->input(1);
      TF_RETURN_IF_ERROR(c->WithRank(splits_shape, 1, &unused));

      // Output shape is a 1-D tensor with size equal to number of splits.
      std::vector<DimensionHandle> dims(1);
      TF_RETURN_IF_ERROR(c->Subtract(c->Dim(splits_shape, 0), 1, &dims[0]));
      c->set_output(0, c->MakeShape(dims));

      return Status::OK();
    });

REGISTER_OP("UnicodeTranscode")
    .Input("input: string")
    .Output("output: string")
    .Attr("input_encoding: string")
    .Attr("output_encoding: {'UTF-8', 'UTF-16-BE', 'UTF-32-BE'}")
    .Attr("errors: {'strict', 'replace', 'ignore'} = 'replace'")
    .Attr("replacement_char: int = 65533")  // 0xFFFD unicode replacement char
    .Attr("replace_control_characters: bool = false")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("UnicodeDecode")
    .Input("input: string")
    .Output("row_splits: Tsplits")
    .Output("char_values: int32")
    .Attr("input_encoding: string")
    .Attr("errors: {'strict', 'replace', 'ignore'} = 'replace'")
    .Attr("replacement_char: int = 65533")  // 0xFFFD unicode replacement char
    .Attr("replace_control_characters: bool = false")
    .Attr("Tsplits: {int32, int64} = DT_INT64")
    .SetShapeFn([](InferenceContext* c) {
      // row_splits.shape == [input.size() + 1]
      DimensionHandle num_row_splits;
      DimensionHandle input_size = c->NumElements(c->input(0));
      TF_RETURN_IF_ERROR(c->Add(input_size, 1, &num_row_splits));
      c->set_output(0, c->Vector(num_row_splits));

      // char_values.shape == [num_chars]
      DimensionHandle num_chars = c->UnknownDim();
      c->set_output(1, c->Vector(num_chars));
      return Status::OK();
    });

REGISTER_OP("UnicodeDecodeWithOffsets")
    .Input("input: string")
    .Output("row_splits: Tsplits")
    .Output("char_values: int32")
    .Output("char_to_byte_starts: int64")
    .Attr("input_encoding: string")
    .Attr("errors: {'strict', 'replace', 'ignore'} = 'replace'")
    .Attr("replacement_char: int = 65533")  // 0xFFFD unicode replacement char
    .Attr("replace_control_characters: bool = false")
    .Attr("Tsplits: {int32, int64} = DT_INT64")
    .SetShapeFn([](InferenceContext* c) {
      // row_splits.shape == [input.size() + 1]
      DimensionHandle num_row_splits;
      DimensionHandle input_size = c->NumElements(c->input(0));
      TF_RETURN_IF_ERROR(c->Add(input_size, 1, &num_row_splits));
      c->set_output(0, c->Vector(num_row_splits));

      // char_values.shape == offset_values.shape == [num_chars]
      DimensionHandle num_chars = c->UnknownDim();
      c->set_output(1, c->Vector(num_chars));
      c->set_output(2, c->Vector(num_chars));
      return Status::OK();
    });

REGISTER_OP("StringNGrams")
    .Attr("separator: string")
    .Attr("ngram_widths: list(int) >= 0")
    .Attr("left_pad: string")
    .Attr("right_pad: string")
    .Attr("pad_width: int")
    .Attr("preserve_short_sequences: bool")
    .Attr("Tsplits: {int32, int64} = DT_INT64")
    .Input("data: string")
    .Input("data_splits: Tsplits")
    .Output("ngrams: string")
    .Output("ngrams_splits: Tsplits")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->UnknownShapeOfRank(1));
      ShapeHandle data = c->input(0);
      TF_RETURN_IF_ERROR(c->WithRank(data, 1, &data));
      ShapeHandle data_splits = c->input(1);
      TF_RETURN_IF_ERROR(c->WithRank(data_splits, 1, &data_splits));
      c->set_output(1, data_splits);
      return Status::OK();
    });

}  // namespace tensorflow
