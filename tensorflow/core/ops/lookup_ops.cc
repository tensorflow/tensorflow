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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/dataset_stateful_op_whitelist.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeAndType;
using shape_inference::ShapeHandle;

// --------------------------------------------------------------------------

namespace {
Status TwoElementVectorInputsAndScalarOutputs(InferenceContext* c) {
  ShapeHandle handle;
  DimensionHandle unused_handle;
  for (int i = 0; i < c->num_inputs(); ++i) {
    TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 1, &handle));
    TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_handle));
  }
  for (int i = 0; i < c->num_outputs(); ++i) {
    c->set_output(i, c->Scalar());
  }
  return Status::OK();
}

Status ScalarAndTwoElementVectorInputsAndScalarOutputs(InferenceContext* c) {
  ShapeHandle handle;
  DimensionHandle unused_handle;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle));
  for (int i = 1; i < c->num_inputs(); ++i) {
    TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 1, &handle));
    TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_handle));
  }
  for (int i = 0; i < c->num_outputs(); ++i) {
    c->set_output(i, c->Scalar());
  }
  return Status::OK();
}

Status TwoElementOutput(InferenceContext* c) {
  c->set_output(0, c->Vector(2));
  return Status::OK();
}

Status ScalarOutput(InferenceContext* c) {
  c->set_output(0, c->Scalar());
  return Status::OK();
}
}  // namespace

REGISTER_OP("LookupTableFind")
    .Input("table_handle: Ref(string)")
    .Input("keys: Tin")
    .Input("default_value: Tout")
    .Output("values: Tout")
    .Attr("Tin: type")
    .Attr("Tout: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));

      // Default value must be scalar or vector.
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(2), 1, &unused));
      c->set_output(0, c->UnknownShape());
      return Status::OK();
    });

Status ValidateTableResourceHandle(InferenceContext* c, ShapeHandle keys,
                                   const string& key_dtype_attr,
                                   const string& value_dtype_attr,
                                   bool is_lookup,
                                   ShapeAndType* output_shape_and_type) {
  auto* handle_data = c->input_handle_shapes_and_types(0);
  if (handle_data == nullptr || handle_data->size() != 2) {
    output_shape_and_type->shape = c->UnknownShape();
    output_shape_and_type->dtype = DT_INVALID;
  } else {
    const ShapeAndType& key_shape_and_type = (*handle_data)[0];
    const ShapeAndType& value_shape_and_type = (*handle_data)[1];
    DataType key_dtype;
    TF_RETURN_IF_ERROR(c->GetAttr(key_dtype_attr, &key_dtype));
    if (key_shape_and_type.dtype != key_dtype) {
      return errors::InvalidArgument(
          "Trying to read value with wrong dtype. "
          "Expected ",
          DataTypeString(key_shape_and_type.dtype), " got ",
          DataTypeString(key_dtype));
    }
    DataType value_dtype;
    TF_RETURN_IF_ERROR(c->GetAttr(value_dtype_attr, &value_dtype));
    if (value_shape_and_type.dtype != value_dtype) {
      return errors::InvalidArgument(
          "Trying to read value with wrong dtype. "
          "Expected ",
          DataTypeString(value_shape_and_type.dtype), " got ",
          DataTypeString(value_dtype));
    }
    output_shape_and_type->dtype = value_shape_and_type.dtype;

    if (is_lookup) {
      if (c->RankKnown(key_shape_and_type.shape) && c->RankKnown(keys)) {
        int keys_rank = c->Rank(keys);
        int key_suffix_rank = c->Rank(key_shape_and_type.shape);
        if (keys_rank < key_suffix_rank) {
          return errors::InvalidArgument(
              "Expected keys to have suffix ",
              c->DebugString(key_shape_and_type.shape),
              " but saw shape: ", c->DebugString(keys));
        }
        for (int d = 0; d < key_suffix_rank; d++) {
          // Ensure the suffix of keys match what's in the Table.
          DimensionHandle dim = c->Dim(key_shape_and_type.shape, d);
          TF_RETURN_IF_ERROR(
              c->ReplaceDim(keys, keys_rank - key_suffix_rank + d, dim, &keys));
        }
        std::vector<DimensionHandle> keys_prefix_vec;
        keys_prefix_vec.reserve(keys_rank - key_suffix_rank);
        for (int d = 0; d < keys_rank - key_suffix_rank; ++d) {
          keys_prefix_vec.push_back(c->Dim(keys, d));
        }
        ShapeHandle keys_prefix = c->MakeShape(keys_prefix_vec);
        TF_RETURN_IF_ERROR(c->Concatenate(keys_prefix,
                                          value_shape_and_type.shape,
                                          &output_shape_and_type->shape));
      } else {
        output_shape_and_type->shape = c->UnknownShape();
      }
    } else {
      TF_RETURN_IF_ERROR(c->Concatenate(keys, value_shape_and_type.shape,
                                        &output_shape_and_type->shape));
    }
  }
  return Status::OK();
}

REGISTER_OP("LookupTableFindV2")
    .Input("table_handle: resource")
    .Input("keys: Tin")
    .Input("default_value: Tout")
    .Output("values: Tout")
    .Attr("Tin: type")
    .Attr("Tout: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle));

      // Default value must be scalar or vector.
      ShapeHandle keys;
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(2), 1, &keys));

      ShapeAndType value_shape_and_type;
      TF_RETURN_IF_ERROR(ValidateTableResourceHandle(
          c,
          /*keys=*/c->input(1),
          /*key_dtype_attr=*/"Tin",
          /*value_dtype_attr=*/"Tout",
          /*is_lookup=*/true, &value_shape_and_type));
      c->set_output(0, value_shape_and_type.shape);

      return Status::OK();
    });
WHITELIST_STATEFUL_OP_FOR_DATASET_FUNCTIONS("LookupTableFindV2");
// TODO(b/72710477): Update this.

REGISTER_OP("LookupTableInsert")
    .Input("table_handle: Ref(string)")
    .Input("keys: Tin")
    .Input("values: Tout")
    .Attr("Tin: type")
    .Attr("Tout: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));

      // TODO(ebrevdo): Validate keys and values shape.
      return Status::OK();
    });

REGISTER_OP("LookupTableInsertV2")
    .Input("table_handle: resource")
    .Input("keys: Tin")
    .Input("values: Tout")
    .Attr("Tin: type")
    .Attr("Tout: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle));

      // TODO: Validate keys and values shape.
      return Status::OK();
    });

REGISTER_OP("LookupTableRemoveV2")
    .Input("table_handle: resource")
    .Input("keys: Tin")
    .Attr("Tin: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle));
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 1, &handle));

      // TODO(turboale): Validate keys shape.
      return Status::OK();
    });

REGISTER_OP("LookupTableSize")
    .Input("table_handle: Ref(string)")
    .Output("size: int64")
    .SetShapeFn(TwoElementVectorInputsAndScalarOutputs);
WHITELIST_STATEFUL_OP_FOR_DATASET_FUNCTIONS("LookupTableSize");

REGISTER_OP("LookupTableSizeV2")
    .Input("table_handle: resource")
    .Output("size: int64")
    .SetShapeFn(ScalarAndTwoElementVectorInputsAndScalarOutputs);
WHITELIST_STATEFUL_OP_FOR_DATASET_FUNCTIONS("LookupTableSizeV2");

REGISTER_OP("LookupTableExport")
    .Input("table_handle: Ref(string)")
    .Output("keys: Tkeys")
    .Output("values: Tvalues")
    .Attr("Tkeys: type")
    .Attr("Tvalues: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));

      ShapeHandle values = c->UnknownShape();
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(values, 1, &values));
      ShapeHandle keys = c->Vector(c->Dim(values, 0));
      c->set_output(0, keys);
      c->set_output(1, values);
      return Status::OK();
    });

REGISTER_OP("LookupTableExportV2")
    .Input("table_handle: resource")
    .Output("keys: Tkeys")
    .Output("values: Tvalues")
    .Attr("Tkeys: type")
    .Attr("Tvalues: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle));
      ShapeHandle keys = c->UnknownShapeOfRank(1);
      ShapeAndType value_shape_and_type;
      TF_RETURN_IF_ERROR(ValidateTableResourceHandle(
          c,
          /*keys=*/keys,
          /*key_dtype_attr=*/"Tkeys",
          /*value_dtype_attr=*/"Tvalues",
          /*is_lookup=*/false, &value_shape_and_type));
      c->set_output(0, keys);
      c->set_output(1, value_shape_and_type.shape);
      return Status::OK();
    });

REGISTER_OP("LookupTableImport")
    .Input("table_handle: Ref(string)")
    .Input("keys: Tin")
    .Input("values: Tout")
    .Attr("Tin: type")
    .Attr("Tout: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));

      // TODO(ebrevdo): Validate keys and values shape.
      return Status::OK();
    });

REGISTER_OP("LookupTableImportV2")
    .Input("table_handle: resource")
    .Input("keys: Tin")
    .Input("values: Tout")
    .Attr("Tin: type")
    .Attr("Tout: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle));

      ShapeHandle keys;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &keys));
      TF_RETURN_IF_ERROR(c->Merge(keys, c->input(2), &keys));
      return Status::OK();
    });

Status MutableHashTableShape(InferenceContext* c, const ShapeHandle& key,
                             const ShapeHandle& value) {
  c->set_output(0, c->Scalar());

  ShapeHandle key_s;
  TF_RETURN_IF_ERROR(c->WithRankAtMost(key, 1, &key_s));

  DataType key_t;
  TF_RETURN_IF_ERROR(c->GetAttr("key_dtype", &key_t));

  DataType value_t;
  TF_RETURN_IF_ERROR(c->GetAttr("value_dtype", &value_t));

  // ShapeAndType vector for {key, value}.
  c->set_output_handle_shapes_and_types(
      0, std::vector<ShapeAndType>{{key_s, key_t}, {value, value_t}});

  return Status::OK();
}

REGISTER_OP("HashTable")
    .Output("table_handle: Ref(string)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("use_node_name_sharing: bool = false")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .SetIsStateful()
    .SetShapeFn(TwoElementOutput);

REGISTER_OP("HashTableV2")
    .Output("table_handle: resource")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("use_node_name_sharing: bool = false")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .SetIsStateful()
    .SetShapeFn(ScalarOutput);

REGISTER_OP("MutableHashTable")
    .Output("table_handle: Ref(string)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("use_node_name_sharing: bool = false")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .SetIsStateful()
    .SetShapeFn(TwoElementOutput);

REGISTER_OP("MutableHashTableV2")
    .Output("table_handle: resource")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("use_node_name_sharing: bool = false")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      return MutableHashTableShape(c, /*key=*/c->Scalar(),
                                   /*value=*/c->Scalar());
    });

REGISTER_OP("MutableHashTableOfTensors")
    .Output("table_handle: Ref(string)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("use_node_name_sharing: bool = false")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .Attr("value_shape: shape = {}")
    .SetIsStateful()
    .SetShapeFn(TwoElementOutput);

REGISTER_OP("MutableHashTableOfTensorsV2")
    .Output("table_handle: resource")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("use_node_name_sharing: bool = false")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .Attr("value_shape: shape = {}")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      PartialTensorShape value_p;
      TF_RETURN_IF_ERROR(c->GetAttr("value_shape", &value_p));
      ShapeHandle value_s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(value_p, &value_s));
      return MutableHashTableShape(c, /*key=*/c->Scalar(), /*value=*/value_s);
    });

REGISTER_OP("MutableDenseHashTable")
    .Input("empty_key: key_dtype")
    .Output("table_handle: Ref(string)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("use_node_name_sharing: bool = false")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .Attr("value_shape: shape = {}")
    .Attr("initial_num_buckets: int = 131072")  // 2^17
    .Attr("max_load_factor: float = 0.8")
    .SetIsStateful()
    .SetShapeFn(TwoElementOutput);

REGISTER_OP("MutableDenseHashTableV2")
    .Input("empty_key: key_dtype")
    .Input("deleted_key: key_dtype")
    .Output("table_handle: resource")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("use_node_name_sharing: bool = false")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .Attr("value_shape: shape = {}")
    .Attr("initial_num_buckets: int = 131072")  // 2^17
    .Attr("max_load_factor: float = 0.8")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      PartialTensorShape value_p;
      TF_RETURN_IF_ERROR(c->GetAttr("value_shape", &value_p));
      ShapeHandle value_s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(value_p, &value_s));
      return MutableHashTableShape(c, /*key=*/c->input(0), /*value=*/value_s);
    });

REGISTER_OP("InitializeTable")
    .Input("table_handle: Ref(string)")
    .Input("keys: Tkey")
    .Input("values: Tval")
    .Attr("Tkey: type")
    .Attr("Tval: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));

      ShapeHandle keys;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &keys));
      TF_RETURN_IF_ERROR(c->Merge(keys, c->input(2), &keys));
      return Status::OK();
    });

REGISTER_OP("InitializeTableV2")
    .Input("table_handle: resource")
    .Input("keys: Tkey")
    .Input("values: Tval")
    .Attr("Tkey: type")
    .Attr("Tval: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle));

      ShapeHandle keys;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &keys));
      TF_RETURN_IF_ERROR(c->Merge(keys, c->input(2), &keys));
      return Status::OK();
    });

REGISTER_OP("InitializeTableFromTextFile")
    .Input("table_handle: Ref(string)")
    .Input("filename: string")
    .Attr("key_index: int >= -2")
    .Attr("value_index: int >= -2")
    .Attr("vocab_size: int >= -1 = -1")
    .Attr("delimiter: string = '\t'")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));

      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &handle));
      return Status::OK();
    });

REGISTER_OP("InitializeTableFromTextFileV2")
    .Input("table_handle: resource")
    .Input("filename: string")
    .Attr("key_index: int >= -2")
    .Attr("value_index: int >= -2")
    .Attr("vocab_size: int >= -1 = -1")
    .Attr("delimiter: string = '\t'")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle));

      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &handle));
      return Status::OK();
    });

}  // namespace tensorflow
