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

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("DecodeRaw")
    .Input("bytes: string")
    .Output("output: out_type")
    .Attr("out_type: {float,double,int32,uint8,int16,int8,int64}")
    .Attr("little_endian: bool = true")
    .SetShapeFn([](InferenceContext* c) {
      // Note: last dimension is data dependent.
      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->Concatenate(
          c->input(0), c->Vector(InferenceContext::kUnknownDim), &out));
      c->set_output(0, out);
      return Status::OK();
    })
    .Doc(R"doc(
Reinterpret the bytes of a string as a vector of numbers.

bytes: All the elements must have the same length.
little_endian: Whether the input `bytes` are in little-endian order.
  Ignored for `out_type` values that are stored in a single byte like
  `uint8`.
output: A Tensor with one more dimension than the input `bytes`.  The
  added dimension will have size equal to the length of the elements
  of `bytes` divided by the number of bytes to represent `out_type`.
)doc");

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
      ParseSingleExampleAttrs attrs;
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
      TensorShapeProto shape_proto;
      for (int i = 0; i < attrs.num_dense; ++i) {
        attrs.dense_shapes[i].AsProto(&shape_proto);
        ShapeHandle dense;
        TF_RETURN_IF_ERROR(c->MakeShapeFromShapeProto(shape_proto, &dense));
        TF_RETURN_IF_ERROR(c->Concatenate(input, dense, &dense));
        c->set_output(output_idx++, dense);
      }
      return Status::OK();
    })
    .Doc(R"doc(
Transforms a vector of brain.Example protos (as strings) into typed tensors.

serialized: A vector containing a batch of binary serialized Example protos.
names: A vector containing the names of the serialized protos.
  May contain, for example, table key (descriptive) names for the
  corresponding serialized protos.  These are purely useful for debugging
  purposes, and the presence of values here has no effect on the output.
  May also be an empty vector if no names are available.
  If non-empty, this vector must be the same length as "serialized".
dense_keys: A list of Ndense string Tensors (scalars).
  The keys expected in the Examples' features associated with dense values.
dense_defaults: A list of Ndense Tensors (some may be empty).
  dense_defaults[j] provides default values
  when the example's feature_map lacks dense_key[j].  If an empty Tensor is
  provided for dense_defaults[j], then the Feature dense_keys[j] is required.
  The input type is inferred from dense_defaults[j], even when it's empty.
  If dense_defaults[j] is not empty, its shape must match dense_shapes[j].
dense_shapes: A list of Ndense shapes; the shapes of data in each Feature
  given in dense_keys.
  The number of elements in the Feature corresponding to dense_key[j]
  must always equal dense_shapes[j].NumEntries().
  If dense_shapes[j] == (D0, D1, ..., DN) then the shape of output
  Tensor dense_values[j] will be (|serialized|, D0, D1, ..., DN):
  The dense outputs are just the inputs row-stacked by batch.
sparse_keys: A list of Nsparse string Tensors (scalars).
  The keys expected in the Examples' features associated with sparse values.
sparse_types: A list of Nsparse types; the data types of data in each Feature
  given in sparse_keys.
  Currently the ParseExample supports DT_FLOAT (FloatList),
  DT_INT64 (Int64List), and DT_STRING (BytesList).
)doc");

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
      TensorShapeProto shape_proto;
      for (int i = 0; i < attrs.num_context_dense; ++i) {
        attrs.context_dense_shapes[i].AsProto(&shape_proto);
        ShapeHandle s;
        TF_RETURN_IF_ERROR(c->MakeShapeFromShapeProto(shape_proto, &s));
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
        attrs.feature_list_dense_shapes[i].AsProto(&shape_proto);
        ShapeHandle s;
        TF_RETURN_IF_ERROR(c->MakeShapeFromShapeProto(shape_proto, &s));
        TF_RETURN_IF_ERROR(
            c->Concatenate(c->Vector(InferenceContext::kUnknownDim), s, &s));
        c->set_output(output_idx++, s);
      }
      return Status::OK();
    })
    .Doc(R"doc(
Transforms a scalar brain.SequenceExample proto (as strings) into typed tensors.

serialized: A scalar containing a binary serialized SequenceExample proto.
feature_list_dense_missing_assumed_empty: A vector listing the
  FeatureList keys which may be missing from the SequenceExample.  If the
  associated FeatureList is missing, it is treated as empty.  By default,
  any FeatureList not listed in this vector must exist in the SequenceExample.
context_dense_keys: A list of Ncontext_dense string Tensors (scalars).
  The keys expected in the SequenceExamples' context features associated with
  dense values.
feature_list_dense_keys: A list of Nfeature_list_dense string Tensors (scalars).
  The keys expected in the SequenceExamples' feature_lists associated
  with lists of dense values.
context_dense_defaults: A list of Ncontext_dense Tensors (some may be empty).
  context_dense_defaults[j] provides default values
  when the SequenceExample's context map lacks context_dense_key[j].
  If an empty Tensor is provided for context_dense_defaults[j],
  then the Feature context_dense_keys[j] is required.
  The input type is inferred from context_dense_defaults[j], even when it's
  empty.  If context_dense_defaults[j] is not empty, its shape must match
  context_dense_shapes[j].
debug_name: A scalar containing the name of the serialized proto.
  May contain, for example, table key (descriptive) name for the
  corresponding serialized proto.  This is purely useful for debugging
  purposes, and the presence of values here has no effect on the output.
  May also be an empty scalar if no name is available.
context_dense_shapes: A list of Ncontext_dense shapes; the shapes of data in
  each context Feature given in context_dense_keys.
  The number of elements in the Feature corresponding to context_dense_key[j]
  must always equal context_dense_shapes[j].NumEntries().
  The shape of context_dense_values[j] will match context_dense_shapes[j].
feature_list_dense_shapes: A list of Nfeature_list_dense shapes; the shapes of
  data in each FeatureList given in feature_list_dense_keys.
  The shape of each Feature in the FeatureList corresponding to
  feature_list_dense_key[j] must always equal
  feature_list_dense_shapes[j].NumEntries().
context_sparse_keys: A list of Ncontext_sparse string Tensors (scalars).
  The keys expected in the Examples' features associated with context_sparse
  values.
context_sparse_types: A list of Ncontext_sparse types; the data types of data in
  each context Feature given in context_sparse_keys.
  Currently the ParseSingleSequenceExample supports DT_FLOAT (FloatList),
  DT_INT64 (Int64List), and DT_STRING (BytesList).
feature_list_sparse_keys: A list of Nfeature_list_sparse string Tensors
  (scalars).  The keys expected in the FeatureLists associated with sparse
  values.
feature_list_sparse_types: A list of Nfeature_list_sparse types; the data types
  of data in each FeatureList given in feature_list_sparse_keys.
  Currently the ParseSingleSequenceExample supports DT_FLOAT (FloatList),
  DT_INT64 (Int64List), and DT_STRING (BytesList).
)doc");

REGISTER_OP("ParseTensor")
    .Input("serialized: string")
    .Output("output: out_type")
    .Attr("out_type: type")
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Transforms a serialized tensorflow.TensorProto proto into a Tensor.

serialized: A scalar string containing a serialized TensorProto proto.
out_type: The type of the serialized tensor.  The provided type must match the
  type of the serialized tensor and no implicit conversion will take place.
output: A Tensor of type `out_type`.
)doc");

REGISTER_OP("DecodeJSONExample")
    .Input("json_examples: string")
    .Output("binary_examples: string")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Convert JSON-encoded Example records to binary protocol buffer strings.

This op translates a tensor containing Example records, encoded using
the [standard JSON
mapping](https://developers.google.com/protocol-buffers/docs/proto3#json),
into a tensor containing the same records encoded as binary protocol
buffers. The resulting tensor can then be fed to any of the other
Example-parsing ops.

json_examples: Each string is a JSON object serialized according to the JSON
  mapping of the Example proto.
binary_examples: Each string is a binary Example protocol buffer corresponding
  to the respective element of `json_examples`.
)doc");

REGISTER_OP("DecodeCSV")
    .Input("records: string")
    .Input("record_defaults: OUT_TYPE")
    .Output("output: OUT_TYPE")
    .Attr("OUT_TYPE: list({float,int32,int64,string})")
    .Attr("field_delim: string = ','")
    .SetShapeFn([](InferenceContext* c) {
      // Validate the record_defaults inputs.
      for (int i = 1; i < c->num_inputs(); ++i) {
        ShapeHandle v;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 1, &v));
        if (c->Value(c->Dim(v, 0)) > 1) {
          return errors::InvalidArgument(
              "Shape of a default must be a length-0 or length-1 vector");
        }
      }

      // Propagate shape of the records input.
      for (int i = 0; i < c->num_outputs(); ++i) c->set_output(i, c->input(0));
      return Status::OK();
    })
    .Doc(R"doc(
Convert CSV records to tensors. Each column maps to one tensor.

RFC 4180 format is expected for the CSV records.
(https://tools.ietf.org/html/rfc4180)
Note that we allow leading and trailing spaces with int or float field.

records: Each string is a record/row in the csv and all records should have
  the same format.
record_defaults: One tensor per column of the input record, with either a
  scalar default value for that column or empty if the column is required.
field_delim: delimiter to separate fields in a record.
output: Each tensor will have the same shape as records.
)doc");

REGISTER_OP("StringToNumber")
    .Input("string_tensor: string")
    .Output("output: out_type")
    .Attr("out_type: {float, int32} = DT_FLOAT")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Converts each string in the input Tensor to the specified numeric type.

(Note that int32 overflow results in an error while float overflow
results in a rounded value.)

out_type: The numeric type to interpret each string in `string_tensor` as.
output: A Tensor of the same shape as the input `string_tensor`.
)doc");

}  // namespace tensorflow
