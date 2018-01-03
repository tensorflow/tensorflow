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
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
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
    })
    .Doc(R"doc(
Looks up keys in a table, outputs the corresponding values.

The tensor `keys` must of the same type as the keys of the table.
The output `values` is of the type of the table values.

The scalar `default_value` is the value output for keys not present in the
table. It must also be of the same type as the table values.

table_handle: Handle to the table.
keys:  Any shape.  Keys to look up.
values: Same shape as `keys`.  Values found in the table, or `default_values`
   for missing keys.
)doc");

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
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(2), 1, &unused));
      c->set_output(0, c->UnknownShape());
      return Status::OK();
    })
    .Doc(R"doc(
Looks up keys in a table, outputs the corresponding values.

The tensor `keys` must of the same type as the keys of the table.
The output `values` is of the type of the table values.

The scalar `default_value` is the value output for keys not present in the
table. It must also be of the same type as the table values.

table_handle: Handle to the table.
keys:  Any shape.  Keys to look up.
values: Same shape as `keys`.  Values found in the table, or `default_values`
   for missing keys.
)doc");

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
    })
    .Doc(R"doc(
Updates the table to associates keys with values.

The tensor `keys` must be of the same type as the keys of the table.
The tensor `values` must be of the type of the table values.

table_handle: Handle to the table.
keys:  Any shape.  Keys to look up.
values: Values to associate with keys.
)doc");

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
    })
    .Doc(R"doc(
Updates the table to associates keys with values.

The tensor `keys` must be of the same type as the keys of the table.
The tensor `values` must be of the type of the table values.

table_handle: Handle to the table.
keys:  Any shape.  Keys to look up.
values: Values to associate with keys.
)doc");

REGISTER_OP("LookupTableSize")
    .Input("table_handle: Ref(string)")
    .Output("size: int64")
    .SetShapeFn(TwoElementVectorInputsAndScalarOutputs)
    .Doc(R"doc(
Computes the number of elements in the given table.

table_handle: Handle to the table.
size: Scalar that contains number of elements in the table.
)doc");

REGISTER_OP("LookupTableSizeV2")
    .Input("table_handle: resource")
    .Output("size: int64")
    .SetShapeFn(ScalarAndTwoElementVectorInputsAndScalarOutputs)
    .Doc(R"doc(
Computes the number of elements in the given table.

table_handle: Handle to the table.
size: Scalar that contains number of elements in the table.
)doc");

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
    })
    .Doc(R"doc(
Outputs all keys and values in the table.

table_handle: Handle to the table.
keys: Vector of all keys present in the table.
values: Tensor of all values in the table. Indexed in parallel with `keys`.
)doc");

REGISTER_OP("LookupTableExportV2")
    .Input("table_handle: resource")
    .Output("keys: Tkeys")
    .Output("values: Tvalues")
    .Attr("Tkeys: type")
    .Attr("Tvalues: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle));

      ShapeHandle values = c->UnknownShape();
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(values, 1, &values));
      ShapeHandle keys = c->Vector(c->Dim(values, 0));
      c->set_output(0, keys);
      c->set_output(1, values);
      return Status::OK();
    })
    .Doc(R"doc(
Outputs all keys and values in the table.

table_handle: Handle to the table.
keys: Vector of all keys present in the table.
values: Tensor of all values in the table. Indexed in parallel with `keys`.
)doc");

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
    })
    .Doc(R"doc(
Replaces the contents of the table with the specified keys and values.

The tensor `keys` must be of the same type as the keys of the table.
The tensor `values` must be of the type of the table values.

table_handle: Handle to the table.
keys:  Any shape.  Keys to look up.
values: Values to associate with keys.
)doc");

REGISTER_OP("LookupTableImportV2")
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
    })
    .Doc(R"doc(
Replaces the contents of the table with the specified keys and values.

The tensor `keys` must be of the same type as the keys of the table.
The tensor `values` must be of the type of the table values.

table_handle: Handle to the table.
keys:  Any shape.  Keys to look up.
values: Values to associate with keys.
)doc");

REGISTER_OP("HashTable")
    .Output("table_handle: Ref(string)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("use_node_name_sharing: bool = false")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .SetIsStateful()
    .SetShapeFn(TwoElementOutput)
    .Doc(R"doc(
Creates a non-initialized hash table.

This op creates a hash table, specifying the type of its keys and values.
Before using the table you will have to initialize it.  After initialization the
table will be immutable.

table_handle: Handle to a table.
container: If non-empty, this table is placed in the given container.
  Otherwise, a default container is used.
shared_name: If non-empty, this table is shared under the given name across
  multiple sessions.
use_node_name_sharing: If true and shared_name is empty, the table is shared
  using the node name.
key_dtype: Type of the table keys.
value_dtype: Type of the table values.
)doc");

REGISTER_OP("HashTableV2")
    .Output("table_handle: resource")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("use_node_name_sharing: bool = false")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .SetIsStateful()
    .SetShapeFn(ScalarOutput)
    .Doc(R"doc(
Creates a non-initialized hash table.

This op creates a hash table, specifying the type of its keys and values.
Before using the table you will have to initialize it.  After initialization the
table will be immutable.

table_handle: Handle to a table.
container: If non-empty, this table is placed in the given container.
  Otherwise, a default container is used.
shared_name: If non-empty, this table is shared under the given name across
  multiple sessions.
use_node_name_sharing: If true and shared_name is empty, the table is shared
  using the node name.
key_dtype: Type of the table keys.
value_dtype: Type of the table values.
)doc");

REGISTER_OP("MutableHashTable")
    .Output("table_handle: Ref(string)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("use_node_name_sharing: bool = false")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .SetIsStateful()
    .SetShapeFn(TwoElementOutput)
    .Doc(R"doc(
Creates an empty hash table.

This op creates a mutable hash table, specifying the type of its keys and
values. Each value must be a scalar. Data can be inserted into the table using
the insert operations. It does not support the initialization operation.

table_handle: Handle to a table.
container: If non-empty, this table is placed in the given container.
  Otherwise, a default container is used.
shared_name: If non-empty, this table is shared under the given name across
  multiple sessions.
use_node_name_sharing: If true and shared_name is empty, the table is shared
  using the node name.
key_dtype: Type of the table keys.
value_dtype: Type of the table values.
)doc");

REGISTER_OP("MutableHashTableV2")
    .Output("table_handle: resource")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("use_node_name_sharing: bool = false")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .SetIsStateful()
    .SetShapeFn(ScalarOutput)
    .Doc(R"doc(
Creates an empty hash table.

This op creates a mutable hash table, specifying the type of its keys and
values. Each value must be a scalar. Data can be inserted into the table using
the insert operations. It does not support the initialization operation.

table_handle: Handle to a table.
container: If non-empty, this table is placed in the given container.
  Otherwise, a default container is used.
shared_name: If non-empty, this table is shared under the given name across
  multiple sessions.
use_node_name_sharing: If true and shared_name is empty, the table is shared
  using the node name.
key_dtype: Type of the table keys.
value_dtype: Type of the table values.
)doc");

REGISTER_OP("MutableHashTableOfTensors")
    .Output("table_handle: Ref(string)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("use_node_name_sharing: bool = false")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .Attr("value_shape: shape = {}")
    .SetIsStateful()
    .SetShapeFn(TwoElementOutput)
    .Doc(R"doc(
Creates an empty hash table.

This op creates a mutable hash table, specifying the type of its keys and
values. Each value must be a vector. Data can be inserted into the table using
the insert operations. It does not support the initialization operation.

table_handle: Handle to a table.
container: If non-empty, this table is placed in the given container.
  Otherwise, a default container is used.
shared_name: If non-empty, this table is shared under the given name across
  multiple sessions.
key_dtype: Type of the table keys.
value_dtype: Type of the table values.
)doc");

REGISTER_OP("MutableHashTableOfTensorsV2")
    .Output("table_handle: resource")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("use_node_name_sharing: bool = false")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .Attr("value_shape: shape = {}")
    .SetIsStateful()
    .SetShapeFn(ScalarOutput)
    .Doc(R"doc(
Creates an empty hash table.

This op creates a mutable hash table, specifying the type of its keys and
values. Each value must be a vector. Data can be inserted into the table using
the insert operations. It does not support the initialization operation.

table_handle: Handle to a table.
container: If non-empty, this table is placed in the given container.
  Otherwise, a default container is used.
shared_name: If non-empty, this table is shared under the given name across
  multiple sessions.
key_dtype: Type of the table keys.
value_dtype: Type of the table values.
)doc");

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
    .SetShapeFn(TwoElementOutput)
    .Doc(R"doc(
Creates an empty hash table that uses tensors as the backing store.

It uses "open addressing" with quadratic reprobing to resolve
collisions.

This op creates a mutable hash table, specifying the type of its keys and
values. Each value must be a scalar. Data can be inserted into the table using
the insert operations. It does not support the initialization operation.

empty_key: The key used to represent empty key buckets internally. Must not
  be used in insert or lookup operations.
table_handle: Handle to a table.
container: If non-empty, this table is placed in the given container.
  Otherwise, a default container is used.
shared_name: If non-empty, this table is shared under the given name across
  multiple sessions.
key_dtype: Type of the table keys.
value_dtype: Type of the table values.
value_shape: The shape of each value.
initial_num_buckets: The initial number of hash table buckets. Must be a power
  to 2.
max_load_factor: The maximum ratio between number of entries and number of
  buckets before growing the table. Must be between 0 and 1.
)doc");

REGISTER_OP("MutableDenseHashTableV2")
    .Input("empty_key: key_dtype")
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
    .SetShapeFn(ScalarOutput)
    .Doc(R"doc(
Creates an empty hash table that uses tensors as the backing store.

It uses "open addressing" with quadratic reprobing to resolve
collisions.

This op creates a mutable hash table, specifying the type of its keys and
values. Each value must be a scalar. Data can be inserted into the table using
the insert operations. It does not support the initialization operation.

empty_key: The key used to represent empty key buckets internally. Must not
  be used in insert or lookup operations.
table_handle: Handle to a table.
container: If non-empty, this table is placed in the given container.
  Otherwise, a default container is used.
shared_name: If non-empty, this table is shared under the given name across
  multiple sessions.
key_dtype: Type of the table keys.
value_dtype: Type of the table values.
value_shape: The shape of each value.
initial_num_buckets: The initial number of hash table buckets. Must be a power
  to 2.
max_load_factor: The maximum ratio between number of entries and number of
  buckets before growing the table. Must be between 0 and 1.
)doc");

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
    })
    .Doc(R"doc(
Table initializer that takes two tensors for keys and values respectively.

table_handle: Handle to a table which will be initialized.
keys: Keys of type Tkey.
values: Values of type Tval.
)doc");

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
    })
    .Doc(R"doc(
Table initializer that takes two tensors for keys and values respectively.

table_handle: Handle to a table which will be initialized.
keys: Keys of type Tkey.
values: Values of type Tval.
)doc");

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
    })
    .Doc(R"doc(
Initializes a table from a text file.

It inserts one key-value pair into the table for each line of the file.
The key and value is extracted from the whole line content, elements from the
split line based on `delimiter` or the line number (starting from zero).
Where to extract the key and value from a line is specified by `key_index` and
`value_index`.

- A value of -1 means use the line number(starting from zero), expects `int64`.
- A value of -2 means use the whole line content, expects `string`.
- A value >= 0 means use the index (starting at zero) of the split line based
  on `delimiter`.

table_handle: Handle to a table which will be initialized.
filename: Filename of a vocabulary text file.
key_index: Column index in a line to get the table `key` values from.
value_index: Column index that represents information of a line to get the table
  `value` values from.
vocab_size: Number of elements of the file, use -1 if unknown.
delimiter: Delimiter to separate fields in a line.
)doc");

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
    })
    .Doc(R"doc(
Initializes a table from a text file.

It inserts one key-value pair into the table for each line of the file.
The key and value is extracted from the whole line content, elements from the
split line based on `delimiter` or the line number (starting from zero).
Where to extract the key and value from a line is specified by `key_index` and
`value_index`.

- A value of -1 means use the line number(starting from zero), expects `int64`.
- A value of -2 means use the whole line content, expects `string`.
- A value >= 0 means use the index (starting at zero) of the split line based
  on `delimiter`.

table_handle: Handle to a table which will be initialized.
filename: Filename of a vocabulary text file.
key_index: Column index in a line to get the table `key` values from.
value_index: Column index that represents information of a line to get the table
  `value` values from.
vocab_size: Number of elements of the file, use -1 if unknown.
delimiter: Delimiter to separate fields in a line.
)doc");

}  // namespace tensorflow
