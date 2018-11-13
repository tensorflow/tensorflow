/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

// --------------------------------------------------------------------------

// The ops in this section can be composed to define an input
// pipeline. Each op produces a DT_VARIANT tensor that represents
// a DAG of "dataset" objects. An "dataset" object can be converted
// to a stateful "iterator" by passing the "dataset" to the
// "MakeIterator" op.
//
// TODO(b/65524810): DT_VARIANT tensors that represent "dataset" objects are
// not presently serializable. To avoid issues with constant folding, ensure
// that any "source dataset" ops (i.e. ops that output a dataset and do not
// take one as input) are marked "stateful".

REGISTER_OP("TensorDataset")
    .Input("components: Toutput_types")
    .Output("handle: variant")
    .Attr("Toutput_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetIsStateful()  // TODO(b/65524810): Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn(shape_inference::ScalarShape);  // TODO(mrry): Validate that
                                                // `components` have shapes
                                                // compatible with
                                                // `output_shapes`.

REGISTER_OP("TensorSliceDataset")
    .Input("components: Toutput_types")
    .Output("handle: variant")
    .Attr("Toutput_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetIsStateful()  // TODO(b/65524810): Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn(shape_inference::ScalarShape);  // TODO(mrry): Validate that the
                                                // dim-0 slices of `components`
                                                // have shapes compatible with
                                                // `output_shapes`.

REGISTER_OP("SparseTensorSliceDataset")
    .Input("indices: int64")
    .Input("values: Tvalues")
    .Input("dense_shape: int64")
    .Output("handle: variant")
    .Attr("Tvalues: type")
    .SetIsStateful()  // TODO(b/65524810): Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("GeneratorDataset")
    .Input("init_func_other_args: Tinit_func_args")
    .Input("next_func_other_args: Tnext_func_args")
    .Input("finalize_func_other_args: Tfinalize_func_args")
    .Output("handle: variant")
    .Attr("init_func: func")
    .Attr("next_func: func")
    .Attr("finalize_func: func")
    .Attr("Tinit_func_args: list(type) >= 0")
    .Attr("Tnext_func_args: list(type) >= 0")
    .Attr("Tfinalize_func_args: list(type) >= 0")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetIsStateful()  // TODO(b/65524810): Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("UnbatchDataset")
    .Input("input_dataset: variant")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("ZipDataset")
    .Input("input_datasets: N * variant")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .Attr("N: int >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("ConcatenateDataset")
    .Input("input_dataset: variant")
    .Input("another_dataset: variant")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("RepeatDataset")
    .Input("input_dataset: variant")
    .Input("count: int64")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle count_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &count_shape));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("TakeDataset")
    .Input("input_dataset: variant")
    .Input("count: int64")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle count_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &count_shape));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("SkipDataset")
    .Input("input_dataset: variant")
    .Input("count: int64")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle count_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &count_shape));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("BytesProducedStatsDataset")
    .Input("input_dataset: variant")
    .Input("tag: string")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle tag_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &tag_shape));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("LatencyStatsDataset")
    .Input("input_dataset: variant")
    .Input("tag: string")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle tag_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &tag_shape));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("ParseExampleDataset")
    .Input("input_dataset: variant")
    .Input("num_parallel_calls: int64")
    .Input("dense_defaults: Tdense")
    .Output("handle: variant")
    .Attr("sparse_keys: list(string) >= 0")
    .Attr("dense_keys: list(string) >= 0")
    .Attr("sparse_types: list({float,int64,string}) >= 0")
    .Attr("Tdense: list({float,int64,string}) >= 0")
    .Attr("dense_shapes: list(shape) >= 0")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")  // Output components will be
                                              // sorted by key (dense_keys and
                                              // sparse_keys combined) here.
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("SetStatsAggregatorDataset")
    .Input("input_dataset: variant")
    .Input("stats_aggregator: resource")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("MapDataset")
    .Input("input_dataset: variant")
    .Input("other_arguments: Targuments")
    .Output("handle: variant")
    .Attr("f: func")
    .Attr("Targuments: list(type) >= 0")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("ParallelMapDataset")
    .Input("input_dataset: variant")
    .Input("other_arguments: Targuments")
    .Input("num_parallel_calls: int32")
    .Output("handle: variant")
    .Attr("f: func")
    .Attr("Targuments: list(type) >= 0")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("MapAndBatchDataset")
    .Input("input_dataset: variant")
    .Input("other_arguments: Targuments")
    .Input("batch_size: int64")
    .Input("num_parallel_batches: int64")
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

REGISTER_OP("MapAndBatchDatasetV2")
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
      // batch_size, num_parallel_calls, and drop_remainder are 0-D scalars.
      shape_inference::ShapeHandle unused;
      TF_RETURN_IF_ERROR(
          c->WithRank(c->input(c->num_inputs() - 3), 0, &unused));
      TF_RETURN_IF_ERROR(
          c->WithRank(c->input(c->num_inputs() - 2), 0, &unused));
      TF_RETURN_IF_ERROR(
          c->WithRank(c->input(c->num_inputs() - 1), 0, &unused));

      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("PrefetchDataset")
    .Input("input_dataset: variant")
    .Input("buffer_size: int64")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // buffer_size should be a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("ScanDataset")
    .Input("input_dataset: variant")
    .Input("initial_state: Tstate")
    .Input("other_arguments: Targuments")
    .Output("handle: variant")
    .Attr("f: func")
    .Attr("Tstate: list(type) >= 1")
    .Attr("Targuments: list(type) >= 0")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("FlatMapDataset")
    .Input("input_dataset: variant")
    .Input("other_arguments: Targuments")
    .Output("handle: variant")
    .Attr("f: func")
    .Attr("Targuments: list(type) >= 0")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("InterleaveDataset")
    .Input("input_dataset: variant")
    .Input("other_arguments: Targuments")
    .Input("cycle_length: int64")
    .Input("block_length: int64")
    .Output("handle: variant")
    .Attr("f: func")
    .Attr("Targuments: list(type) >= 0")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("ParallelInterleaveDataset")
    .Input("input_dataset: variant")
    .Input("other_arguments: Targuments")
    .Input("cycle_length: int64")
    .Input("block_length: int64")
    .Input("sloppy: bool")
    .Input("buffer_output_elements: int64")
    .Input("prefetch_input_elements: int64")
    .Output("handle: variant")
    .Attr("f: func")
    .Attr("Targuments: list(type) >= 0")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("GroupByReducerDataset")
    .Input("input_dataset: variant")
    .Input("key_func_other_arguments: Tkey_func_other_arguments")
    .Input("init_func_other_arguments: Tinit_func_other_arguments")
    .Input("reduce_func_other_arguments: Treduce_func_other_arguments")
    .Input("finalize_func_other_arguments: Tfinalize_func_other_arguments")
    .Output("handle: variant")
    .Attr("key_func: func")
    .Attr("init_func: func")
    .Attr("reduce_func: func")
    .Attr("finalize_func: func")
    .Attr("Tkey_func_other_arguments: list(type) >= 0")
    .Attr("Tinit_func_other_arguments: list(type) >= 0")
    .Attr("Treduce_func_other_arguments: list(type) >= 0")
    .Attr("Tfinalize_func_other_arguments: list(type) >= 0")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("GroupByWindowDataset")
    .Input("input_dataset: variant")
    .Input("key_func_other_arguments: Tkey_func_other_arguments")
    .Input("reduce_func_other_arguments: Treduce_func_other_arguments")
    .Input(
        "window_size_func_other_arguments: Twindow_size_func_other_arguments")
    .Output("handle: variant")
    .Attr("key_func: func")
    .Attr("reduce_func: func")
    .Attr("window_size_func: func")
    .Attr("Tkey_func_other_arguments: list(type) >= 0")
    .Attr("Treduce_func_other_arguments: list(type) >= 0")
    .Attr("Twindow_size_func_other_arguments: list(type) >= 0")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("FilterDataset")
    .Input("input_dataset: variant")
    .Input("other_arguments: Targuments")
    .Output("handle: variant")
    .Attr("predicate: func")
    .Attr("Targuments: list(type) >= 0")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("FilterByLastComponentDataset")
    .Input("input_dataset: variant")
    .Output("output: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("WindowDataset")
    .Input("input_dataset: variant")
    .Input("window_size: int64")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // batch_size should be a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("BatchDataset")
    .Input("input_dataset: variant")
    .Input("batch_size: int64")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // batch_size should be a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("BatchDatasetV2")
    .Input("input_dataset: variant")
    .Input("batch_size: int64")
    .Input("drop_remainder: bool")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // batch_size should be a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      // drop_remainder should be a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("SlideDataset")
    .Input("input_dataset: variant")
    .Input("window_size: int64")
    .Input("window_shift: int64")
    .Input("window_stride: int64")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // window_size, window_shift, and window_stride should be scalars.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      return shape_inference::ScalarShape(c);
    });

// TODO(mrry): Validate that `padded_shapes` are all vectors, the lengths of
// `output_types` and `output_shapes` are `N` the `output_shapes` are (as far as
// possible to tell statically) compatible with `padded_shapes`, and that
// `padding_values` are all scalars.
REGISTER_OP("PaddedBatchDataset")
    .Input("input_dataset: variant")
    .Input("batch_size: int64")
    .Input("padded_shapes: N * int64")
    .Input("padding_values: Toutput_types")
    .Output("handle: variant")
    .Attr("Toutput_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .Attr("N: int >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // batch_size should be a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("PaddedBatchDatasetV2")
    .Input("input_dataset: variant")
    .Input("batch_size: int64")
    .Input("padded_shapes: N * int64")
    .Input("padding_values: Toutput_types")
    .Input("drop_remainder: bool")
    .Output("handle: variant")
    .Attr("Toutput_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .Attr("N: int >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // batch_size should be a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      // drop_remainder should be a scalar.
      TF_RETURN_IF_ERROR(
          c->WithRank(c->input(c->num_inputs() - 1), 0, &unused));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("DenseToSparseBatchDataset")
    .Input("input_dataset: variant")
    .Input("batch_size: int64")
    .Input("row_shape: int64")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // batch_size should be a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      // row_shape should be a 1-D vector.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("RangeDataset")
    .Input("start: int64")
    .Input("stop: int64")
    .Input("step: int64")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetIsStateful()  // TODO(b/65524810): Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // start, stop, and step should be scalars.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("RandomDataset")
    .Input("seed: int64")
    .Input("seed2: int64")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetIsStateful()  // TODO(b/65524810): Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // buffer_size, seed, and seed2 should be scalars.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("ShuffleDataset")
    .Input("input_dataset: variant")
    .Input("buffer_size: int64")
    .Input("seed: int64")
    .Input("seed2: int64")
    .Output("handle: variant")
    .Attr("reshuffle_each_iteration: bool = true")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // buffer_size, seed, and seed2 should be scalars.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("ShuffleAndRepeatDataset")
    .Input("input_dataset: variant")
    .Input("buffer_size: int64")
    .Input("seed: int64")
    .Input("seed2: int64")
    .Input("count: int64")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // buffer_size, seed, seed2, and count should be scalars.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("CacheDataset")
    .Input("input_dataset: variant")
    .Input("filename: string")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // filename should be a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("TextLineDataset")
    .Input("filenames: string")
    .Input("compression_type: string")
    .Input("buffer_size: int64")
    .Output("handle: variant")
    .SetIsStateful()  // TODO(b/65524810): Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // `filenames` must be a scalar or a vector.
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(0), 1, &unused));
      // `compression_type` could only be a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      // `buffer_size` could only be a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("SqlDataset")
    .Input("driver_name: string")
    .Input("data_source_name: string")
    .Input("query: string")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetIsStateful()  // TODO(b/65524810): Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // driver_name, data_source_name, and query should be scalars.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("FixedLengthRecordDataset")
    .Input("filenames: string")
    .Input("header_bytes: int64")
    .Input("record_bytes: int64")
    .Input("footer_bytes: int64")
    .Input("buffer_size: int64")
    .Output("handle: variant")
    .SetIsStateful()  // TODO(b/65524810): Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // `filenames` must be a scalar or a vector.
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(0), 1, &unused));
      // header_bytes, record_bytes, footer_bytes, buffer_size should be
      // scalars.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("TFRecordDataset")
    .Input("filenames: string")
    .Input("compression_type: string")
    .Input("buffer_size: int64")
    .Output("handle: variant")
    .SetIsStateful()  // TODO(b/65524810): Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // `filenames` must be a scalar or a vector.
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(0), 1, &unused));
      // `compression_type` could only be a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      // `buffer_size` could only be a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("Iterator")
    .Output("handle: resource")
    .Attr("shared_name: string")
    .Attr("container: string")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("IteratorV2")
    .Output("handle: resource")
    .Attr("shared_name: string")
    .Attr("container: string")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("AnonymousIterator")
    .Output("handle: resource")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("MakeIterator")
    .Input("dataset: variant")
    .Input("iterator: resource")
    .SetShapeFn(shape_inference::NoOutputs);

REGISTER_OP("OneShotIterator")
    .Output("handle: resource")
    .Attr("dataset_factory: func")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

namespace {

Status IteratorGetNextShapeFn(shape_inference::InferenceContext* c) {
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

REGISTER_OP("IteratorGetNext")
    .Input("iterator: resource")
    .Output("components: output_types")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(IteratorGetNextShapeFn);

REGISTER_OP("IteratorGetNextSync")
    .Input("iterator: resource")
    .Output("components: output_types")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(IteratorGetNextShapeFn);

REGISTER_OP("DatasetToSingleElement")
    .Input("dataset: variant")
    .Output("components: output_types")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(IteratorGetNextShapeFn);

REGISTER_OP("IteratorToStringHandle")
    .Input("resource_handle: resource")
    .Output("string_handle: string")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("IteratorFromStringHandle")
    .Input("string_handle: string")
    .Output("resource_handle: resource")
    .Attr("output_types: list(type) >= 0 = []")
    .Attr("output_shapes: list(shape) >= 0 = []")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("IteratorFromStringHandleV2")
    .Input("string_handle: string")
    .Output("resource_handle: resource")
    .Attr("output_types: list(type) >= 0 = []")
    .Attr("output_shapes: list(shape) >= 0 = []")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("SerializeIterator")
    .Input("resource_handle: resource")
    .Output("serialized: variant")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("DeserializeIterator")
    .Input("resource_handle: resource")
    .Input("serialized: variant")
    .SetShapeFn(shape_inference::NoOutputs);

REGISTER_OP("StatsAggregatorHandle")
    .Output("handle: resource")
    .SetShapeFn(shape_inference::ScalarShape)
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''");

REGISTER_OP("StatsAggregatorSummary")
    .Input("iterator: resource")
    .Output("summary: string")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("PrependFromQueueAndPaddedBatchDataset")
    .Input("input_dataset: variant")
    .Input("batch_size: int64")
    .Input("padded_shapes: N * int64")
    .Input("padding_values: Toutput_types")
    .Output("handle: variant")
    .Attr("Toutput_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .Attr("N: int >= 1")
    // TODO(ebrevdo): Validate that `padded_shapes` are all vectors, the lengths
    // of `Toutput_types` and `output_shapes` are `N`, that the
    // length of `output_types` is `N`, the `output_shapes` are
    // (as far as possible to tell statically) compatible with `padded_shapes`,
    // and that `padding_values` are all scalars.
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // batch_size should be a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("EnqueueInQueueDataset")
    .Input("queue: variant")
    .Input("components: Tcomponents")
    .Attr("Tcomponents: list(type) >= 1")
    .SetIsStateful()  // To avoid CSE on multiple calls to Enqueue.
    // TODO(ebrevdo): SetShapeFn to test input dtypes and shapes by
    // reading from queue handle (is that even possible?).
    .SetShapeFn(shape_inference::NoOutputs);

REGISTER_OP("DatasetToTFRecord")
    .Input("input_dataset: variant")
    .Input("filename: string")
    .Input("compression_type: string")
    .SetShapeFn(shape_inference::NoOutputs);

REGISTER_OP("DatasetToGraph")
    .Input("input_dataset: variant")
    .Output("graph: string")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("SinkDataset")
    .Input("input_dataset: variant")
    .Output("handle: variant")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("OptimizeDataset")
    .Input("input_dataset: variant")
    .Input("optimizations: string")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("OptionalFromValue")
    .Input("components: Toutput_types")
    .Output("optional: variant")
    .Attr("Toutput_types: list(type) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("OptionalNone")
    .Output("optional: variant")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("OptionalHasValue")
    .Input("optional: variant")
    .Output("has_value: bool")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("OptionalGetValue")
    .Input("optional: variant")
    .Output("components: output_types")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(IteratorGetNextShapeFn);

REGISTER_OP("IteratorGetNextAsOptional")
    .Input("iterator: resource")
    .Output("optional: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("MapDefun")
    .Input("arguments: Targuments")
    .Output("output: output_types")
    .Attr("Targuments: list(type) >= 1")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .Attr("f: func")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      std::vector<TensorShape> output_shapes;
      TF_RETURN_IF_ERROR(c->GetAttr("output_shapes", &output_shapes));
      if (output_shapes.size() != c->num_outputs()) {
        return errors::InvalidArgument(
            "`output_shapes` must be the same length as `output_types` (",
            output_shapes.size(), " vs. ", c->num_outputs(), ")");
      }

      int64 dim_zero = -1;
      for (size_t i = 0; i < static_cast<size_t>(c->num_inputs()); ++i) {
        auto dim_handle = c->Dim(c->input(i), 0);
        if (c->ValueKnown(dim_handle)) {
          if (dim_zero == -1) {
            dim_zero = c->Value(dim_handle);
          } else if (c->Value(dim_handle) != dim_zero) {
            return errors::InvalidArgument(
                "Inputs must have the same dimension 0.");
          }
        }
      }

      for (size_t i = 0; i < output_shapes.size(); ++i) {
        PartialTensorShape s({});
        s = s.Concatenate(dim_zero);
        s = s.Concatenate(output_shapes[i]);
        shape_inference::ShapeHandle output_shape_handle;

        TF_RETURN_IF_ERROR(
            c->MakeShapeFromPartialTensorShape(s, &output_shape_handle));
        c->set_output(static_cast<int>(i), output_shape_handle);
      }
      return Status::OK();
    });

}  // namespace tensorflow
