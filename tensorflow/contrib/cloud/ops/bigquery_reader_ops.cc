/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

/* This file registers Bigquery reader ops. */

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
namespace tensorflow {

using shape_inference::InferenceContext;

REGISTER_OP("BigQueryReader")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("project_id: string")
    .Attr("dataset_id: string")
    .Attr("table_id: string")
    .Attr("columns: list(string)")
    .Attr("timestamp_millis: int")
    .Attr("test_end_point: string = ''")
    .Output("reader_handle: Ref(string)")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Vector(2));
      return Status::OK();
    })
    .Doc(R"doc(
A Reader that outputs rows from a BigQuery table as tensorflow Examples.

container: If non-empty, this reader is placed in the given container.
           Otherwise, a default container is used.
shared_name: If non-empty, this reader is named in the given bucket
             with this shared_name. Otherwise, the node name is used instead.
project_id: GCP project ID.
dataset_id: BigQuery Dataset ID.
table_id: Table to read.
columns: List of columns to read. Leave empty to read all columns.
timestamp_millis: Table snapshot timestamp in millis since epoch. Relative
(negative or zero) snapshot times are not allowed. For more details, see
'Table Decorators' in BigQuery docs.
test_end_point: Do not use. For testing purposes only.
reader_handle: The handle to reference the Reader.
)doc");

REGISTER_OP("GenerateBigQueryReaderPartitions")
    .Attr("project_id: string")
    .Attr("dataset_id: string")
    .Attr("table_id: string")
    .Attr("columns: list(string)")
    .Attr("timestamp_millis: int")
    .Attr("num_partitions: int")
    .Attr("test_end_point: string = ''")
    .Output("partitions: string")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Vector(InferenceContext::kUnknownDim));
      return Status::OK();
    })
    .Doc(R"doc(
Generates serialized partition messages suitable for batch reads.

This op should not be used directly by clients. Instead, the
bigquery_reader_ops.py file defines a clean interface to the reader.

project_id: GCP project ID.
dataset_id: BigQuery Dataset ID.
table_id: Table to read.
columns: List of columns to read. Leave empty to read all columns.
timestamp_millis: Table snapshot timestamp in millis since epoch. Relative
(negative or zero) snapshot times are not allowed. For more details, see
'Table Decorators' in BigQuery docs.
num_partitions: Number of partitions to split the table into.
test_end_point: Do not use. For testing purposes only.
partitions: Serialized table partitions.
)doc");

}  // namespace tensorflow
