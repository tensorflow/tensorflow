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

namespace tensorflow {

REGISTER_OP("ExperimentalDirectedInterleaveDataset")
    .Input("selector_input_dataset: variant")
    .Input("data_input_datasets: N * variant")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .Attr("N: int >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("ExperimentalCSVDataset")
    .Input("filenames: string")
    .Input("compression_type: string")
    .Input("buffer_size: int64")
    .Input("header: bool")
    .Input("field_delim: string")
    .Input("use_quote_delim: bool")
    .Input("na_value: string")
    .Input("select_cols: int64")
    .Input("record_defaults: output_types")
    .Output("handle: variant")
    .Attr("output_types: list({float,double,int32,int64,string}) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetIsStateful()  // TODO(b/65524810): Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // `filenames` must be a scalar or a vector.
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(0), 1, &unused));
      // `compression_type`, `buffer_size`, `header`, `field_delim`,
      // `use_quote_delim`, `na_value` must be scalars
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &unused));
      // `select_cols` must be a vector
      TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 1, &unused));
      // `record_defaults` must be lists of scalars
      for (size_t i = 8; i < c->num_inputs(); ++i) {
        shape_inference::ShapeHandle v;
        TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(i), 1, &v));
        if (c->Rank(c->input(i)) == 1 && c->Value(c->Dim(v, 0)) > 1) {
          return errors::InvalidArgument(
              "Shape of a default must be a length-0 or length-1 vector, or a "
              "scalar.");
        }
      }
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("ExperimentalIgnoreErrorsDataset")
    .Input("input_dataset: variant")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("ExperimentalMapDataset")
    .Input("input_dataset: variant")
    .Input("other_arguments: Targuments")
    .Output("handle: variant")
    .Attr("f: func")
    .Attr("Targuments: list(type) >= 0")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .Attr("use_inter_op_parallelism: bool = true")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("ExperimentalNonSerializableDataset")
    .Input("input_dataset: variant")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("ExperimentalSleepDataset")
    .Input("input_dataset: variant")
    .Input("sleep_microseconds: int64")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // Both inputs are scalar.
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(0), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(1), 0, &unused));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("ExperimentalUniqueDataset")
    .Input("input_dataset: variant")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("ExperimentalIteratorGetDevice")
    .Input("resource: resource")
    .Output("device: string")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("ExperimentalFunctionBufferingResource")
    .Input("string_arg: string")
    .Input("target_device: string")
    .Output("resource: resource")
    .Attr("shared_name: string")
    .Attr("container: string")
    .Attr("f: func")
    .Attr("buffer_size: int")
    .Attr("output_types: list(type)")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("ExperimentalFunctionBufferingResourceGetNext")
    .Input("function_buffer_resource: resource")
    .Attr("output_types: list(type)")
    .Output("output: output_types")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("ExperimentalFunctionBufferingResourceReset")
    .Input("function_buffer_resource: resource")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("ExperimentalThreadPoolDataset")
    .Input("input_dataset: variant")
    .Input("thread_pool: resource")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("ExperimentalThreadPoolHandle")
    .Output("handle: resource")
    .SetShapeFn(shape_inference::ScalarShape)
    .Attr("num_threads: int")
    .Attr("max_intra_op_parallelism: int = 1")
    .Attr("display_name: string")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''");

REGISTER_OP("ExperimentalAssertNextDataset")
    .Input("input_dataset: variant")
    .Input("transformations: string")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // transformations should be a vector.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("ExperimentalNumaMapAndBatchDataset")
    .Input("input_dataset: variant")
    .Input("other_arguments: Targuments")
    .Input("batch_size: int64")
    .Input("num_parallel_calls: int64")
    .Input("drop_remainder: bool")
    .Output("handle: variant")
    .Attr("f: func")
    .Attr("Targuments: list(type) >= 0")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      // Use index from the end to retrieve the Input shapes,
      // so that to avoid guessing the length of "other_arguments".
      // batch_size, num_parallel_batches, and drop_remainder are 0-D scalars.
      shape_inference::ShapeHandle unused;
      TF_RETURN_IF_ERROR(
          c->WithRank(c->input(c->num_inputs() - 3), 0, &unused));
      TF_RETURN_IF_ERROR(
          c->WithRank(c->input(c->num_inputs() - 2), 0, &unused));
      TF_RETURN_IF_ERROR(
          c->WithRank(c->input(c->num_inputs() - 1), 0, &unused));

      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("ExperimentalLMDBDataset")
    .Input("filenames: string")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetIsStateful()  // TODO(b/65524810): Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("ExperimentalIdentityIndexedDataset")
    .Input("size: uint64")
    .Output("handle: variant")
    .SetIsStateful()
    .SetShapeFn(
        shape_inference::ScalarShape);  // TODO(saeta): check input shapes.

///////////////////////////////////////////////////////////////////////////////
//     IndexedDataset Internals
///////////////////////////////////////////////////////////////////////////////

// Creates the handle.
REGISTER_OP("ExperimentalMaterializedIndexDatasetHandle")
    .Output("handle: resource")
    .Attr("container: string")
    .Attr("shared_name: string")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

// Actually materialize the materialize handle.
REGISTER_OP("ExperimentalIndexedDatasetMaterialize")
    .Input("dataset: variant")
    .Input("materialized: resource")
    .SetShapeFn(shape_inference::NoOutputs);

namespace {

Status GetShapeFn(shape_inference::InferenceContext* c) {
  shape_inference::ShapeHandle unused;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
  std::vector<PartialTensorShape> output_shapes;
  TF_RETURN_IF_ERROR(c->GetAttr("output_shapes", &output_shapes));
  if (output_shapes.size() != c->num_outputs()) {
    return errors::InvalidArgument(
        "`output_shapes` must be the same length as `output_types` (",
        output_shapes.size(), " vs. ", c->num_outputs());
  }
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    shape_inference::ShapeHandle output_shape_handle;
    TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(
        output_shapes[i], &output_shape_handle));
    c->set_output(static_cast<int>(i), output_shape_handle);
  }
  return Status::OK();
}

}  // namespace

REGISTER_OP("ExperimentalIndexedDatasetGet")
    .Input("materialized: resource")
    .Input("index: uint64")
    .Output("components: output_types")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(GetShapeFn);

}  // namespace tensorflow
