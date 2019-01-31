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
#include "tensorflow/core/util/example_proto_helper.h"

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("DecodeRaw")
    .Input("bytes: string")
    .Output("output: out_type")
    .Attr(
        "out_type: "
        "{half,float,double,int32,uint16,uint8,int16,int8,int64,complex64,"
        "complex128}")
    .Attr("little_endian: bool = true")
    .SetShapeFn([](InferenceContext* c) {
      // Note: last dimension is data dependent.
      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->Concatenate(
          c->input(0), c->Vector(InferenceContext::kUnknownDim), &out));
      c->set_output(0, out);
      return Status::OK();
    });

REGISTER_OP("DecodeCompressed")
    .Input("bytes: string")
    .Output("output: string")
    .Attr("compression_type: string = ''")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("ParseExample")
    .Input("serialized: string")
    .Input("names: string")
    .Input("sparse_keys: Nsparse * string")
    .Input("dense_keys: Ndense * string")
    .Input("dense_defaults: Tdense")
    .Output("sparse_indices: Nsparse * int64")
    .Output("sparse_values: sparse_types")
    .Output("sparse_shapes: Nsparse * int64")
    .Output("dense_values: Tdense")
    .Attr("Nsparse: int >= 0")  // Inferred from sparse_keys
    .Attr("Ndense: int >= 0")   // Inferred from dense_keys
    .Attr("sparse_types: list({float,int64,string}) >= 0")
    .Attr("Tdense: list({float,int64,string}) >= 0")
    .Attr("dense_shapes: list(shape) >= 0")
    .SetShapeFn([](InferenceContext* c) {
      ParseExampleAttrs attrs;
      TF_RETURN_IF_ERROR(attrs.Init(c));

      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));  // names

      // Output sparse_indices, sparse_values, and sparse_shapes.
      int output_idx = 0;
      for (int i = 0; i < attrs.num_sparse; ++i) {
        c->set_output(output_idx++, c->Matrix(c->UnknownDim(), 2));
      }
      for (int i = 0; i < attrs.num_sparse; ++i) {
        c->set_output(output_idx++, c->Vector(c->UnknownDim()));
      }
      for (int i = 0; i < attrs.num_sparse; ++i) {
        c->set_output(output_idx++, c->Vector(2));
      }

      // Output dense_shapes.
      for (int i = 0; i < attrs.num_dense; ++i) {
        ShapeHandle dense;
        TF_RETURN_IF_ERROR(
            c->MakeShapeFromPartialTensorShape(attrs.dense_shapes[i], &dense));
        TF_RETURN_IF_ERROR(c->Concatenate(input, dense, &dense));
        c->set_output(output_idx++, dense);
      }
      return Status::OK();
    });

REGISTER_OP("ParseSingleExample")
    .Input("serialized: string")
    .Input("dense_defaults: Tdense")
    .Output("sparse_indices: num_sparse * int64")
    .Output("sparse_values: sparse_types")
    .Output("sparse_shapes: num_sparse * int64")
    .Output("dense_values: Tdense")
    .Attr("num_sparse: int >= 0")
    .Attr("sparse_keys: list(string) >= 0")
    .Attr("dense_keys: list(string) >= 0")
    .Attr("sparse_types: list({float,int64,string}) >= 0")
    .Attr("Tdense: list({float,int64,string}) >= 0")
    .Attr("dense_shapes: list(shape) >= 0")
    .SetShapeFn([](InferenceContext* c) {
      ParseSingleExampleAttrs attrs;
      TF_RETURN_IF_ERROR(attrs.Init(c));

      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &input));

      // Output sparse_indices, sparse_values, and sparse_shapes.
      int output_idx = 0;
      for (int i = 0; i < attrs.sparse_keys.size(); ++i) {
        c->set_output(output_idx++, c->Matrix(c->UnknownDim(), 1));
      }
      for (int i = 0; i < attrs.sparse_keys.size(); ++i) {
        c->set_output(output_idx++, c->Vector(c->UnknownDim()));
      }
      for (int i = 0; i < attrs.sparse_keys.size(); ++i) {
        c->set_output(output_idx++, c->Vector(1));
      }

      // Output dense_shapes.
      for (int i = 0; i < attrs.dense_keys.size(); ++i) {
        ShapeHandle dense;
        TF_RETURN_IF_ERROR(
            c->MakeShapeFromPartialTensorShape(attrs.dense_shapes[i], &dense));
        c->set_output(output_idx++, dense);
      }
      return Status::OK();
    });

REGISTER_OP("ParseSequenceExample")
    .Input("serialized: string")
    .Input("debug_name: string")
    .Input("context_dense_defaults: Tcontext_dense")
    .Output("context_sparse_indices: Ncontext_sparse * int64")
    .Output("context_sparse_values: context_sparse_types")
    .Output("context_sparse_shapes: Ncontext_sparse * int64")
    .Output("context_dense_values: Tcontext_dense")
    .Output("feature_list_sparse_indices: Nfeature_list_sparse * int64")
    .Output("feature_list_sparse_values: feature_list_sparse_types")
    .Output("feature_list_sparse_shapes: Nfeature_list_sparse * int64")
    .Output("feature_list_dense_values: feature_list_dense_types")
    .Output("feature_list_dense_lengths: Nfeature_list_dense * int64")
    .Attr("feature_list_dense_missing_assumed_empty: list(string) >= 0")
    .Attr("context_sparse_keys: list(string) >= 0")
    .Attr("context_dense_keys: list(string) >= 0")
    .Attr("feature_list_sparse_keys: list(string) >= 0")
    .Attr("feature_list_dense_keys: list(string) >= 0")
    .Attr("Ncontext_sparse: int >= 0 = 0")
    .Attr("Ncontext_dense: int >= 0 = 0")
    .Attr("Nfeature_list_sparse: int >= 0 = 0")
    .Attr("Nfeature_list_dense: int >= 0 = 0")
    .Attr("context_sparse_types: list({float,int64,string}) >= 0 = []")
    .Attr("Tcontext_dense: list({float,int64,string}) >= 0 = []")
    .Attr("feature_list_dense_types: list({float,int64,string}) >= 0 = []")
    .Attr("context_dense_shapes: list(shape) >= 0 = []")
    .Attr("feature_list_sparse_types: list({float,int64,string}) >= 0 = []")
    .Attr("feature_list_dense_shapes: list(shape) >= 0 = []")
    .SetShapeFn([](InferenceContext* c) {
      ParseSequenceExampleAttrs attrs;
      TF_RETURN_IF_ERROR(attrs.Init(c));

      // Verify that the input is a vector, and carry the shape if known.
      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input));
      shape_inference::DimensionHandle num_examples = c->Dim(input, 0);

      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));  // debug_name

      int output_idx = 0;

      // Output context_sparse_indices, context_sparse_values, and
      // context_sparse_shapes.
      for (int i = 0; i < attrs.num_context_sparse; ++i) {
        c->set_output(output_idx++, c->Matrix(c->UnknownDim(), 2));
      }
      for (int i = 0; i < attrs.num_context_sparse; ++i) {
        c->set_output(output_idx++, c->Vector(c->UnknownDim()));
      }
      for (int i = 0; i < attrs.num_context_sparse; ++i) {
        c->set_output(output_idx++, c->Vector(2));
      }

      // Output context_dense_values.
      for (int i = 0; i < attrs.num_context_dense; ++i) {
        ShapeHandle s;
        TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(
            attrs.context_dense_shapes[i], &s));
        TF_RETURN_IF_ERROR(c->Concatenate(c->Vector(num_examples), s, &s));
        c->set_output(output_idx++, s);
      }

      // Output feature_list_sparse_indices, feature_list_sparse_values,
      // feature_list_sparse_shapes.
      for (int i = 0; i < attrs.num_feature_list_sparse; ++i) {
        c->set_output(output_idx++, c->Matrix(c->UnknownDim(), 3));
      }
      for (int i = 0; i < attrs.num_feature_list_sparse; ++i) {
        c->set_output(output_idx++, c->Vector(c->UnknownDim()));
      }
      for (int i = 0; i < attrs.num_feature_list_sparse; ++i) {
        c->set_output(output_idx++, c->Vector(3));
      }

      // Output feature_list_dense_shapes.
      for (int i = 0; i < attrs.num_feature_list_dense; ++i) {
        ShapeHandle s;
        TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(
            attrs.feature_list_dense_shapes[i], &s));
        TF_RETURN_IF_ERROR(
            c->Concatenate(c->Matrix(num_examples, c->UnknownDim()), s, &s));
        c->set_output(output_idx++, s);
      }

      // Output feature_list_dense_lengths.
      for (int i = 0; i < attrs.num_feature_list_dense; ++i) {
        c->set_output(output_idx++, c->Vector(num_examples));
      }

      return Status::OK();
    });

REGISTER_OP("ParseSingleSequenceExample")
    .Input("serialized: string")
    .Input("feature_list_dense_missing_assumed_empty: string")
    .Input("context_sparse_keys: Ncontext_sparse * string")
    .Input("context_dense_keys: Ncontext_dense * string")
    .Input("feature_list_sparse_keys: Nfeature_list_sparse * string")
    .Input("feature_list_dense_keys: Nfeature_list_dense * string")
    .Input("context_dense_defaults: Tcontext_dense")
    .Input("debug_name: string")
    .Output("context_sparse_indices: Ncontext_sparse * int64")
    .Output("context_sparse_values: context_sparse_types")
    .Output("context_sparse_shapes: Ncontext_sparse * int64")
    .Output("context_dense_values: Tcontext_dense")
    .Output("feature_list_sparse_indices: Nfeature_list_sparse * int64")
    .Output("feature_list_sparse_values: feature_list_sparse_types")
    .Output("feature_list_sparse_shapes: Nfeature_list_sparse * int64")
    .Output("feature_list_dense_values: feature_list_dense_types")
    // Infer from context_sparse_keys
    .Attr("Ncontext_sparse: int >= 0 = 0")
    // Infer from context_dense_keys
    .Attr("Ncontext_dense: int >= 0 = 0")
    // Infer from feature_list_sparse_keys
    .Attr("Nfeature_list_sparse: int >= 0 = 0")
    // Infer from feature_list_dense_keys
    .Attr("Nfeature_list_dense: int >= 0 = 0")
    .Attr("context_sparse_types: list({float,int64,string}) >= 0 = []")
    .Attr("Tcontext_dense: list({float,int64,string}) >= 0 = []")
    .Attr("feature_list_dense_types: list({float,int64,string}) >= 0 = []")
    .Attr("context_dense_shapes: list(shape) >= 0 = []")
    .Attr("feature_list_sparse_types: list({float,int64,string}) >= 0 = []")
    .Attr("feature_list_dense_shapes: list(shape) >= 0 = []")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      ParseSingleSequenceExampleAttrs attrs;
      TF_RETURN_IF_ERROR(attrs.Init(c));

      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &input));

      // feature_list_dense_missing_assumed_empty
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));

      int output_idx = 0;

      // Output context_sparse_indices, context_sparse_values, and
      // context_sparse_shapes.
      for (int i = 0; i < attrs.num_context_sparse; ++i) {
        c->set_output(output_idx++, c->Matrix(c->UnknownDim(), 1));
      }
      for (int i = 0; i < attrs.num_context_sparse; ++i) {
        c->set_output(output_idx++, c->Vector(c->UnknownDim()));
      }
      for (int i = 0; i < attrs.num_context_sparse; ++i) {
        c->set_output(output_idx++, c->Vector(1));
      }

      // Output context_dense_shapes.
      for (int i = 0; i < attrs.num_context_dense; ++i) {
        ShapeHandle s;
        TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(
            attrs.context_dense_shapes[i], &s));
        c->set_output(output_idx++, s);
      }

      // Output feature_list_sparse_indices, feature_list_sparse_values,
      // feature_list_sparse_shapes.
      for (int i = 0; i < attrs.num_feature_list_sparse; ++i) {
        c->set_output(output_idx++, c->Matrix(c->UnknownDim(), 2));
      }
      for (int i = 0; i < attrs.num_feature_list_sparse; ++i) {
        c->set_output(output_idx++, c->Vector(c->UnknownDim()));
      }
      for (int i = 0; i < attrs.num_feature_list_sparse; ++i) {
        c->set_output(output_idx++, c->Vector(2));
      }

      // Output feature_list_dense_shapes.
      for (int i = 0; i < attrs.num_feature_list_dense; ++i) {
        ShapeHandle s;
        TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(
            attrs.feature_list_dense_shapes[i], &s));
        TF_RETURN_IF_ERROR(
            c->Concatenate(c->Vector(InferenceContext::kUnknownDim), s, &s));
        c->set_output(output_idx++, s);
      }
      return Status::OK();
    });

REGISTER_OP("ParseTensor")
    .Input("serialized: string")
    .Output("output: out_type")
    .Attr("out_type: type")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("SerializeTensor")
    .Input("tensor: T")
    .Output("serialized: string")
    .Attr("T: type")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("DecodeJSONExample")
    .Input("json_examples: string")
    .Output("binary_examples: string")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("DecodeCSV")
    .Input("records: string")
    .Input("record_defaults: OUT_TYPE")
    .Output("output: OUT_TYPE")
    .Attr("OUT_TYPE: list({float,double,int32,int64,string})")
    .Attr("field_delim: string = ','")
    .Attr("use_quote_delim: bool = true")
    .Attr("na_value: string = ''")
    .Attr("select_cols: list(int) = []")
    .SetShapeFn([](InferenceContext* c) {
      // Validate the record_defaults inputs.
      for (int i = 1; i < c->num_inputs(); ++i) {
        ShapeHandle v;
        TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(i), 1, &v));
        if (c->Rank(c->input(i)) == 1 && c->Value(c->Dim(v, 0)) > 1) {
          return errors::InvalidArgument(
              "Shape of a default must be a length-0 or length-1 vector, or a "
              "scalar.");
        }
      }

      // Propagate shape of the records input.
      for (int i = 0; i < c->num_outputs(); ++i) c->set_output(i, c->input(0));
      return Status::OK();
    });

REGISTER_OP("StringToNumber")
    .Input("string_tensor: string")
    .Output("output: out_type")
    .Attr("out_type: {float, double, int32, int64} = DT_FLOAT")
    .SetShapeFn(shape_inference::UnchangedShape);

}  // namespace tensorflow
