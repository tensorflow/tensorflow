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
// TODO(b/123753214): DT_VARIANT tensors that represent "dataset" objects are
// not presently serializable. To avoid issues with constant folding, ensure
// that any "source dataset" ops (i.e. ops that output a dataset and do not
// take one as input) are marked "stateful".

REGISTER_OP("TensorDataset")
    .Input("components: Toutput_types")
    .Output("handle: variant")
    .Attr("Toutput_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetIsStateful()  // TODO(b/123753214): Source dataset ops must be marked
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
    .SetIsStateful()  // TODO(b/123753214): Source dataset ops must be marked
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
    .SetIsStateful()  // TODO(b/123753214): Source dataset ops must be marked
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
    .SetIsStateful()  // TODO(b/123753214): Source dataset ops must be marked
                      // stateful to inhibit constant folding.
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

REGISTER_OP("MapDataset")
    .Input("input_dataset: variant")
    .Input("other_arguments: Targuments")
    .Output("handle: variant")
    .Attr("f: func")
    .Attr("Targuments: list(type) >= 0")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .Attr("use_inter_op_parallelism: bool = true")
    .Attr("preserve_cardinality: bool = false")
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
    .Attr("use_inter_op_parallelism: bool = true")
    .Attr("sloppy: bool = false")
    .Attr("preserve_cardinality: bool = false")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("PrefetchDataset")
    .Input("input_dataset: variant")
    .Input("buffer_size: int64")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .Attr("slack_period: int = 0")
    .Attr("legacy_autotune: bool = true")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // buffer_size should be a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      return shape_inference::ScalarShape(c);
    });

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

REGISTER_OP("ParallelInterleaveDatasetV2")
    .Input("input_dataset: variant")
    .Input("other_arguments: Targuments")
    .Input("cycle_length: int64")
    .Input("block_length: int64")
    .Input("num_parallel_calls: int64")
    .Output("handle: variant")
    .Attr("f: func")
    .Attr("Targuments: list(type) >= 0")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .Attr("sloppy: bool = false")
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

// This op is no longer supported.
REGISTER_OP("FilterByLastComponentDataset")
    .Input("input_dataset: variant")
    .Output("output: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("WindowDataset")
    .Input("input_dataset: variant")
    .Input("size: int64")
    .Input("shift: int64")
    .Input("stride: int64")
    .Input("drop_remainder: bool")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // size, shift, stride, and drop_remainder should be scalars.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));
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
    .Attr("parallel_copy: bool = false")
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

REGISTER_OP("ShardDataset")
    .Input("input_dataset: variant")
    .Input("num_shards: int64")
    .Input("index: int64")
    .Output("handle: variant")
    .Attr("require_non_empty: bool = false")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // num_shards should be a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      // index should be a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
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
    .Attr("parallel_copy: bool = false")
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

REGISTER_OP("RangeDataset")
    .Input("start: int64")
    .Input("stop: int64")
    .Input("step: int64")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetIsStateful()  // TODO(b/123753214): Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // start, stop, and step should be scalars.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("AnonymousRandomSeedGenerator")
    .Input("seed: int64")
    .Input("seed2: int64")
    .Output("handle: resource")
    .Output("deleter: variant")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      c->set_output(1, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("DeleteRandomSeedGenerator")
    .Input("handle: resource")
    .Input("deleter: variant")
    .SetShapeFn(shape_inference::NoOutputs);

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

REGISTER_OP("ShuffleDatasetV2")
    .Input("input_dataset: variant")
    .Input("buffer_size: int64")
    .Input("seed_generator: resource")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // buffer_size, seed, and seed2 should be scalars.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
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

REGISTER_OP("AnonymousMemoryCache")
    .Output("handle: resource")
    .Output("deleter: variant")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      c->set_output(1, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("DeleteMemoryCache")
    .Input("handle: resource")
    .Input("deleter: variant")
    .SetShapeFn(shape_inference::NoOutputs);

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

REGISTER_OP("CacheDatasetV2")
    .Input("input_dataset: variant")
    .Input("filename: string")
    .Input("cache: resource")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // filename should be a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      // cache should be a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("TextLineDataset")
    .Input("filenames: string")
    .Input("compression_type: string")
    .Input("buffer_size: int64")
    .Output("handle: variant")
    .SetIsStateful()  // TODO(b/123753214): Source dataset ops must be marked
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

REGISTER_OP("FixedLengthRecordDataset")
    .Input("filenames: string")
    .Input("header_bytes: int64")
    .Input("record_bytes: int64")
    .Input("footer_bytes: int64")
    .Input("buffer_size: int64")
    .Output("handle: variant")
    .SetIsStateful()  // TODO(b/123753214): Source dataset ops must be marked
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

REGISTER_OP("FixedLengthRecordDatasetV2")
    .Input("filenames: string")
    .Input("header_bytes: int64")
    .Input("record_bytes: int64")
    .Input("footer_bytes: int64")
    .Input("buffer_size: int64")
    .Input("compression_type: string")
    .Output("handle: variant")
    .SetIsStateful()  // TODO(b/123753214): Source dataset ops must be marked
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
    .SetIsStateful()  // TODO(b/123753214): Source dataset ops must be marked
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

REGISTER_OP("AnonymousIteratorV2")
    .Output("handle: resource")
    .Output("deleter: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      c->set_output(1, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("DeleteIterator")
    .Input("handle: resource")
    .Input("deleter: variant")
    .SetShapeFn(shape_inference::NoOutputs);

REGISTER_OP("DeleteMultiDeviceIterator")
    .Input("multi_device_iterator: resource")
    .Input("iterators: N * resource")
    .Input("deleter: variant")
    .Attr("N: int >= 0")
    .SetShapeFn(shape_inference::NoOutputs);

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

// TODO(b/124308596): Instead of conservatively marking this op as stateful,
// implement a mechanism to determine whether `dataset` has a side-effect
// and use it to decide whether to use a stateless or stateful version of this
// op.
REGISTER_OP("DatasetToSingleElement")
    .Input("dataset: variant")
    .Output("components: output_types")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetIsStateful()
    .SetShapeFn(IteratorGetNextShapeFn);

// TODO(b/124308596): Instead of conservatively marking this op as stateful,
// implement a mechanism to determine whether `dataset` has a side-effect
// and use it to decide whether to use a stateless or stateful version of this
// op.
REGISTER_OP("ReduceDataset")
    .Input("input_dataset: variant")
    .Input("initial_state: Tstate")
    .Input("other_arguments: Targuments")
    .Output("components: output_types")
    .Attr("f: func")
    .Attr("Tstate: list(type) >= 1")
    .Attr("Targuments: list(type) >= 0")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .Attr("use_inter_op_parallelism: bool = true")
    .SetIsStateful()
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

REGISTER_OP("DatasetToGraph")
    .Input("input_dataset: variant")
    .Attr("stateful_whitelist: list(string) >= 0 = []")
    .Attr("allow_stateful: bool = false")
    .Attr("strip_device_assignment: bool = false")
    .Output("graph: string")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("DatasetToGraphV2")
    .Input("input_dataset: variant")
    .Attr("external_state_policy: int = 0")
    .Attr("strip_device_assignment: bool = false")
    .Output("graph: string")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("OptimizeDataset")
    .Input("input_dataset: variant")
    .Input("optimizations: string")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .Attr("optimization_configs: list(string) = []")
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

REGISTER_OP("ModelDataset")
    .Input("input_dataset: variant")
    .Output("handle: variant")
    .Attr("algorithm: int = 0")
    .Attr("cpu_budget: int = 0")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

// TODO(b/124308749): Add a stateful version of MapDefun and use it when `f`
// is stateful.
REGISTER_OP("MapDefun")
    .Input("arguments: Targuments")
    .Input("captured_inputs: Tcaptured")
    .Output("output: output_types")
    .Attr("Targuments: list(type) >= 1")
    .Attr("Tcaptured: list(type) >= 0 = []")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .Attr("f: func")
    .Attr("max_intra_op_parallelism: int = 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      std::vector<PartialTensorShape> output_shapes;
      TF_RETURN_IF_ERROR(c->GetAttr("output_shapes", &output_shapes));
      DataTypeVector t_args;
      TF_RETURN_IF_ERROR(c->GetAttr("Targuments", &t_args));
      if (output_shapes.size() != c->num_outputs()) {
        return errors::InvalidArgument(
            "`output_shapes` must be the same length as `output_types` (",
            output_shapes.size(), " vs. ", c->num_outputs(), ")");
      }

      int64 dim_zero = -1;
      for (size_t i = 0; i < t_args.size(); ++i) {
        if (c->Rank(c->input(i)) == 0) {
          return errors::InvalidArgument(
              "Arguments must have rank at least 1. Input ", i,
              " has rank of 0.");
        }
        auto dim_handle = c->Dim(c->input(i), 0);
        if (c->ValueKnown(dim_handle)) {
          if (dim_zero == -1) {
            dim_zero = c->Value(dim_handle);
          } else if (c->Value(dim_handle) != dim_zero) {
            return errors::InvalidArgument(
                "Arguments must have the same dimension 0.");
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

REGISTER_OP("WrapDatasetVariant")
    .Input("input_handle: variant")
    .Output("output_handle: variant")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("UnwrapDatasetVariant")
    .Input("input_handle: variant")
    .Output("output_handle: variant")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("AnonymousMultiDeviceIterator")
    .Output("handle: resource")
    .Output("deleter: variant")
    .Attr("devices: list(string) >= 1")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      c->set_output(1, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("MultiDeviceIterator")
    .Output("handle: resource")
    .Attr("devices: list(string) >= 1")
    .Attr("shared_name: string")
    .Attr("container: string")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("MultiDeviceIteratorInit")
    .Input("dataset: variant")
    .Input("multi_device_iterator: resource")
    .Input("max_buffer_size: int64")
    .Output("incarnation_id: int64")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("MultiDeviceIteratorGetNextFromShard")
    .Input("multi_device_iterator: resource")
    .Input("shard_num: int32")
    .Input("incarnation_id: int64")
    .Output("components: output_types")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(IteratorGetNextShapeFn);

REGISTER_OP("MultiDeviceIteratorToStringHandle")
    .Input("multi_device_iterator: resource")
    .Output("string_handle: string")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("MultiDeviceIteratorFromStringHandle")
    .Input("string_handle: string")
    .Output("multi_device_iterator: resource")
    .Attr("output_types: list(type) >= 0 = []")
    .Attr("output_shapes: list(shape) >= 0 = []")
    .SetShapeFn(shape_inference::ScalarShape);

}  // namespace tensorflow
