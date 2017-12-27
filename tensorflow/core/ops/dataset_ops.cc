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
    .SetShapeFn(shape_inference::ScalarShape)  // TODO(mrry): Validate that
                                               // `components` have shapes
                                               // compatible with
                                               // `output_shapes`.
    .Doc(R"doc(
Creates a dataset that emits `components` as a tuple of tensors once.
)doc");

REGISTER_OP("TensorSliceDataset")
    .Input("components: Toutput_types")
    .Output("handle: variant")
    .Attr("Toutput_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetIsStateful()  // TODO(b/65524810): Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn(shape_inference::ScalarShape)  // TODO(mrry): Validate that the
                                               // dim-0 slices of `components`
                                               // have shapes compatible with
                                               // `output_shapes`.
    .Doc(R"doc(
Creates a dataset that emits each dim-0 slice of `components` once.
)doc");

REGISTER_OP("SparseTensorSliceDataset")
    .Input("indices: int64")
    .Input("values: Tvalues")
    .Input("dense_shape: int64")
    .Output("handle: variant")
    .Attr("Tvalues: type")
    .SetIsStateful()  // TODO(b/65524810): Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Creates a dataset that splits a SparseTensor into elements row-wise.
)doc");

REGISTER_OP("ZipDataset")
    .Input("input_datasets: N * variant")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .Attr("N: int >= 1")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Creates a dataset that zips together `input_datasets`.
)doc");

REGISTER_OP("ConcatenateDataset")
    .Input("input_dataset: variant")
    .Input("another_dataset: variant")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Creates a dataset that concatenates `input_dataset` with `another_dataset`.
)doc");

REGISTER_OP("RepeatDataset")
    .Input("input_dataset: variant")
    .Input("count: int64")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape)  // TODO(mrry): Validate the shape
                                               // of `count`.
    .Doc(R"doc(
Creates a dataset that emits the outputs of `input_dataset` `count` times.

count: A scalar representing the number of times that `input_dataset` should
  be repeated. A value of `-1` indicates that it should be repeated infinitely.
)doc");

REGISTER_OP("TakeDataset")
    .Input("input_dataset: variant")
    .Input("count: int64")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Creates a dataset that contains `count` elements from the `input_dataset`.

count: A scalar representing the number of elements from the `input_dataset`
  that should be taken. A value of `-1` indicates that all of `input_dataset`
  is taken.
)doc");

REGISTER_OP("SkipDataset")
    .Input("input_dataset: variant")
    .Input("count: int64")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Creates a dataset that skips `count` elements from the `input_dataset`.

count: A scalar representing the number of elements from the `input_dataset`
  that should be skipped.  If count is -1, skips everything.
)doc");

REGISTER_OP("IgnoreErrorsDataset")
    .Input("input_dataset: variant")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Creates a dataset that contains the elements of `input_dataset` ignoring errors.
)doc");

REGISTER_OP("BytesProducedStatsDataset")
    .Input("input_dataset: variant")
    .Input("tag: string")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Records the bytes size of each element of `input_dataset` in a StatsAggregator.
)doc");

REGISTER_OP("LatencyStatsDataset")
    .Input("input_dataset: variant")
    .Input("tag: string")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Records the latency of producing `input_dataset` elements in a StatsAggregator.
)doc");

REGISTER_OP("MapDataset")
    .Input("input_dataset: variant")
    .Input("other_arguments: Targuments")
    .Output("handle: variant")
    .Attr("f: func")
    .Attr("Targuments: list(type) >= 0")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Creates a dataset that applies `f` to the outputs of `input_dataset`.
)doc");

REGISTER_OP("ParallelMapDataset")
    .Input("input_dataset: variant")
    .Input("other_arguments: Targuments")
    .Input("num_parallel_calls: int32")
    .Output("handle: variant")
    .Attr("f: func")
    .Attr("Targuments: list(type) >= 0")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Creates a dataset that applies `f` to the outputs of `input_dataset`.

Unlike a "MapDataset", which applies `f` sequentially, this dataset invokes up
to `num_parallel_calls` copies of `f` in parallel.

num_parallel_calls: The number of concurrent invocations of `f` that process
  elements from `input_dataset` in parallel.
)doc");

REGISTER_OP("MapAndBatchDataset")
    .Input("input_dataset: variant")
    .Input("other_arguments: Targuments")
    .Input("batch_size: int64")
    .Input("num_parallel_batches: int64")
    .Output("handle: variant")
    .Attr("f: func")
    .Attr("Targuments: list(type) >= 0")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Creates a dataset that applies `f` to the outputs of `input_dataset` and then
batches `batch_size` of them.

Unlike a "MapDataset", which applies `f` sequentially, this dataset invokes up
to `batch_size * num_parallel_batches` copies of `f` in parallel.

batch_size: A scalar representing the number of elements to accumulate in a
  batch. It determines the number of concurrent invocations of `f` that process
  elements from `input_dataset` in parallel.
num_parallel_batches: A scalar representing the number of batches to create in
  parallel. Processing multiple batches in parallel benefits workloads prone to
  stragglers.
)doc");

REGISTER_OP("PrefetchDataset")
    .Input("input_dataset: variant")
    .Input("buffer_size: int64")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Creates a dataset that asynchronously prefetches elements from `input_dataset`.

buffer_size: The maximum number of elements to buffer in an iterator over
  this dataset.
)doc");

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
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Creates a dataset successively reduces `f` over the elements of `input_dataset`.
)doc");

REGISTER_OP("FlatMapDataset")
    .Input("input_dataset: variant")
    .Input("other_arguments: Targuments")
    .Output("handle: variant")
    .Attr("f: func")
    .Attr("Targuments: list(type) >= 0")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Creates a dataset that applies `f` to the outputs of `input_dataset`.

Unlike MapDataset, the `f` in FlatMapDataset is expected to return a
Dataset variant, and FlatMapDataset will flatten successive results
into a single Dataset.

f: A function mapping elements of `input_dataset`, concatenated with
  `other_arguments`, to a Dataset variant that contains elements matching
  `output_types` and `output_shapes`.
)doc");

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
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Creates a dataset that applies `f` to the outputs of `input_dataset`.

Unlike MapDataset, the `f` in InterleaveDataset is expected to return
a Dataset variant, and InterleaveDataset will flatten successive
results into a single Dataset. Unlike FlatMapDataset,
InterleaveDataset will interleave sequences of up to `block_length`
consecutive elements from `cycle_length` input elements.

f: A function mapping elements of `input_dataset`, concatenated with
  `other_arguments`, to a Dataset variant that contains elements matching
  `output_types` and `output_shapes`.
)doc");

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
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Creates a dataset that applies `f` to the outputs of `input_dataset`.

The resulting dataset is similar to the `InterleaveDataset`, with the exception
that if retrieving the next value from a dataset would cause the requester to
block, it will skip that input dataset. This dataset is especially useful
when loading data from a variable-latency datastores (e.g. HDFS, GCS), as it
allows the training step to proceed so long as some data is available.

!! WARNING !! This dataset is not deterministic!

f: A function mapping elements of `input_dataset`, concatenated with
   `other_arguments`, to a Dataset variant that contains elements matching
   `output_types` and `output_shapes`.
)doc");

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
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Creates a dataset that computes a windowed group-by on `input_dataset`.

// TODO(mrry): Support non-int64 keys.

key_func: A function mapping an element of `input_dataset`, concatenated
  with `key_func_other_arguments` to a scalar value of type DT_INT64.
)doc");

REGISTER_OP("FilterDataset")
    .Input("input_dataset: variant")
    .Input("other_arguments: Targuments")
    .Output("handle: variant")
    .Attr("predicate: func")
    .Attr("Targuments: list(type) >= 0")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Creates a dataset containing elements of `input_dataset` matching `predicate`.

The `predicate` function must return a scalar boolean and accept the
following arguments:

* One tensor for each component of an element of `input_dataset`.
* One tensor for each value in `other_arguments`.

predicate: A function returning a scalar boolean.
other_arguments: A list of tensors, typically values that were captured when
  building a closure for `predicate`.
)doc");

REGISTER_OP("BatchDataset")
    .Input("input_dataset: variant")
    .Input("batch_size: int64")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Creates a dataset that batches `batch_size` elements from `input_dataset`.

batch_size: A scalar representing the number of elements to accumulate in a
  batch.
)doc");

REGISTER_OP("PaddedBatchDataset")
    .Input("input_dataset: variant")
    .Input("batch_size: int64")
    .Input("padded_shapes: N * int64")
    .Input("padding_values: Toutput_types")
    .Output("handle: variant")
    .Attr("Toutput_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .Attr("N: int >= 1")
    .SetShapeFn(shape_inference::ScalarShape)  // TODO(mrry): Validate that
                                               // `padded_shapes` are all
                                               // vectors, the lengths of
                                               // `output_types` and
                                               // `output_shapes` are `N`,
                                               // the `output_shapes` are (as
                                               // far as possible to tell
                                               // statically) compatible with
                                               // `padded_shapes`, and
                                               // that `padding_values` are
                                               // all scalars.
    .Doc(R"doc(
Creates a dataset that batches and pads `batch_size` elements from the input.

batch_size: A scalar representing the number of elements to accumulate in a
  batch.
padded_shapes: A list of int64 tensors representing the desired padded shapes
  of the corresponding output components. These shapes may be partially
  specified, using `-1` to indicate that a particular dimension should be
  padded to the maximum size of all batch elements.
padding_values: A list of scalars containing the padding value to use for
  each of the outputs.
)doc");

REGISTER_OP("DenseToSparseBatchDataset")
    .Input("input_dataset: variant")
    .Input("batch_size: int64")
    .Input("row_shape: int64")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Creates a dataset that batches input elements into a SparseTensor.

input_dataset: A handle to an input dataset. Must have a single component.
batch_size: A scalar representing the number of elements to accumulate in a
  batch.
row_shape: A vector representing the dense shape of each row in the produced
  SparseTensor. The shape may be partially specified, using `-1` to indicate
  that a particular dimension should use the maximum size of all batch elements.
)doc");

REGISTER_OP("RangeDataset")
    .Input("start: int64")
    .Input("stop: int64")
    .Input("step: int64")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetIsStateful()  // TODO(b/65524810): Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Creates a dataset with a range of values. Corresponds to python's xrange.

start: corresponds to start in python's xrange().
stop: corresponds to stop in python's xrange().
step: corresponds to step in python's xrange().
)doc");

REGISTER_OP("RandomDataset")
    .Input("seed: int64")
    .Input("seed2: int64")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetIsStateful()  // TODO(b/65524810): Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Creates a Dataset that returns pseudorandom numbers.

seed: A scalar seed for the random number generator. If either seed or
  seed2 is set to be non-zero, the random number generator is seeded
  by the given seed.  Otherwise, a random seed is used.
seed2: A second scalar seed to avoid seed collision.
)doc");

REGISTER_OP("ShuffleDataset")
    .Input("input_dataset: variant")
    .Input("buffer_size: int64")
    .Input("seed: int64")
    .Input("seed2: int64")
    .Output("handle: variant")
    .Attr("reshuffle_each_iteration: bool = true")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Creates a dataset that shuffles elements from `input_dataset` pseudorandomly.

buffer_size: The number of output elements to buffer in an iterator over
  this dataset. Compare with the `min_after_dequeue` attr when creating a
  `RandomShuffleQueue`.
reshuffle_each_iteration: If true, each iterator over this dataset will be given
  a different pseudorandomly generated seed, based on a sequence seeded by the
  `seed` and `seed2` inputs. If false, each iterator will be given the same
  seed, and repeated iteration over this dataset will yield the exact same
  sequence of results.
seed: A scalar seed for the random number generator. If either `seed` or
  `seed2` is set to be non-zero, the random number generator is seeded
  by the given seed.  Otherwise, a random seed is used.
seed2: A second scalar seed to avoid seed collision.
)doc");

REGISTER_OP("ShuffleAndRepeatDataset")
    .Input("input_dataset: variant")
    .Input("buffer_size: int64")
    .Input("seed: int64")
    .Input("seed2: int64")
    .Input("count: int64")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Creates a dataset that shuffles and repeats elements from `input_dataset`
pseudorandomly.

buffer_size: The number of output elements to buffer in an iterator over
  this dataset. Compare with the `min_after_dequeue` attr when creating a
  `RandomShuffleQueue`.
count: A scalar representing the number of times the underlying dataset
  should be repeated. The default is `-1`, which results in infinite repetition.
seed: A scalar seed for the random number generator. If either `seed` or
  `seed2` is set to be non-zero, the random number generator is seeded
  by the given seed.  Otherwise, a random seed is used.
seed2: A second scalar seed to avoid seed collision.
)doc");

REGISTER_OP("CacheDataset")
    .Input("input_dataset: variant")
    .Input("filename: string")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Creates a dataset that caches elements from `input_dataset`.

A CacheDataset will iterate over the input_dataset, and store tensors. If the
cache already exists, the cache will be used. If the cache is inappropriate
(e.g. cannot be opened, contains tensors of the wrong shape / size), an error
will the returned when used.

filename: A path on the filesystem where we should cache the dataset. Note: this
  will be a directory.
)doc");

REGISTER_OP("UniqueDataset")
    .Input("input_dataset: variant")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Creates a dataset that contains the unique elements of `input_dataset`.
)doc");

REGISTER_OP("TextLineDataset")
    .Input("filenames: string")
    .Input("compression_type: string")
    .Input("buffer_size: int64")
    .Output("handle: variant")
    .SetIsStateful()  // TODO(b/65524810): Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn(shape_inference::ScalarShape)  // TODO(mrry): validate
                                               // that `filenames` is
                                               // a scalar or a
                                               // vector.
    .Doc(R"doc(
Creates a dataset that emits the lines of one or more text files.

filenames: A scalar or a vector containing the name(s) of the file(s) to be
  read.
compression_type: A scalar containing either (i) the empty string (no
  compression), (ii) "ZLIB", or (iii) "GZIP".
buffer_size: A scalar containing the number of bytes to buffer.
)doc");

REGISTER_OP("SqlDataset")
    .Input("driver_name: string")
    .Input("data_source_name: string")
    .Input("query: string")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetIsStateful()  // TODO(b/65524810): Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Creates a dataset that executes a SQL query and emits rows of the result set.

driver_name: The database type. Currently, the only supported type is 'sqlite'.
data_source_name: A connection string to connect to the database.
query: A SQL query to execute.
)doc");

REGISTER_OP("FixedLengthRecordDataset")
    .Input("filenames: string")
    .Input("header_bytes: int64")
    .Input("record_bytes: int64")
    .Input("footer_bytes: int64")
    .Input("buffer_size: int64")
    .Output("handle: variant")
    .SetIsStateful()  // TODO(b/65524810): Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Creates a dataset that emits the records from one or more binary files.

filenames: A scalar or a vector containing the name(s) of the file(s) to be
  read.
header_bytes: A scalar representing the number of bytes to skip at the
  beginning of a file.
record_bytes: A scalar representing the number of bytes in each record.
footer_bytes: A scalar representing the number of bytes to skip at the end
  of a file.
buffer_size: A scalar representing the number of bytes to buffer. Must be > 0.
)doc");

REGISTER_OP("TFRecordDataset")
    .Input("filenames: string")
    .Input("compression_type: string")
    .Input("buffer_size: int64")
    .Output("handle: variant")
    .SetIsStateful()  // TODO(b/65524810): Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Creates a dataset that emits the records from one or more TFRecord files.

filenames: A scalar or vector containing the name(s) of the file(s) to be
  read.
compression_type: A scalar containing either (i) the empty string (no
  compression), (ii) "ZLIB", or (iii) "GZIP".
buffer_size: A scalar representing the number of bytes to buffer. A value of
  0 means no buffering will be performed.
)doc");

REGISTER_OP("Iterator")
    .Output("handle: resource")
    .Attr("shared_name: string")
    .Attr("container: string")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
A container for an iterator resource.

handle: A handle to the iterator that can be passed to a "MakeIterator"
  or "IteratorGetNext" op.
)doc");

REGISTER_OP("MakeIterator")
    .Input("dataset: variant")
    .Input("iterator: resource")
    .SetShapeFn(shape_inference::NoOutputs)
    .Doc(R"doc(
Makes a new iterator from the given `dataset` and stores it in `iterator`.

This operation may be executed multiple times. Each execution will reset the
iterator in `iterator` to the first element of `dataset`.
)doc");

REGISTER_OP("OneShotIterator")
    .Output("handle: resource")
    .Attr("dataset_factory: func")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Makes a "one-shot" iterator that can be iterated only once.

A one-shot iterator bundles the logic for defining the dataset and
the state of the iterator in a single op, which allows simple input
pipelines to be defined without an additional initialization
("MakeIterator") step.

One-shot iterators have the following limitations:

* They do not support parameterization: all logic for creating the underlying
  dataset must be bundled in the `dataset_factory` function.
* They are not resettable. Once a one-shot iterator reaches the end of its
  underlying dataset, subsequent "IteratorGetNext" operations on that
  iterator will always produce an `OutOfRange` error.

For greater flexibility, use "Iterator" and "MakeIterator" to define
an iterator using an arbitrary subgraph, which may capture tensors
(including fed values) as parameters, and which may be reset multiple
times by rerunning "MakeIterator".

handle: A handle to the iterator that can be passed to an "IteratorGetNext"
  op.
dataset_factory: A function of type `() -> DT_VARIANT`, where the returned
  DT_VARIANT is a dataset.
)doc");

REGISTER_OP("IteratorGetNext")
    .Input("iterator: resource")
    .Output("components: output_types")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
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
    })
    .Doc(R"doc(
Gets the next output from the given iterator.
)doc");

REGISTER_OP("DatasetToSingleElement")
    .Input("dataset: variant")
    .Output("components: output_types")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
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
    })
    .Doc(R"doc(
Outputs the single element from the given dataset.

dataset: A handle to a dataset that contains a single element.
components: The components of the single element of `input`.
)doc");

REGISTER_OP("IteratorToStringHandle")
    .Input("resource_handle: resource")
    .Output("string_handle: string")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Converts the given `resource_handle` representing an iterator to a string.

resource_handle: A handle to an iterator resource.
string_handle: A string representation of the given handle.
)doc");

REGISTER_OP("IteratorFromStringHandle")
    .Input("string_handle: string")
    .Output("resource_handle: resource")
    .Attr("output_types: list(type) >= 0 = []")
    .Attr("output_shapes: list(shape) >= 0 = []")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Converts the given string representing a handle to an iterator to a resource.

string_handle: A string representation of the given handle.
resource_handle: A handle to an iterator resource.
output_types: If specified, defines the type of each tuple component in an
  element produced by the resulting iterator.
output_shapes: If specified, defines the shape of each tuple component in an
  element produced by the resulting iterator.
)doc");

REGISTER_OP("SerializeIterator")
    .Input("resource_handle: resource")
    .Output("serialized: variant")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Converts the given `resource_handle` representing an iterator to a variant tensor.

resource_handle: A handle to an iterator resource.
serialized: A variant tensor storing the state of the iterator contained in the
  resource.
)doc");

REGISTER_OP("DeserializeIterator")
    .Input("resource_handle: resource")
    .Input("serialized: variant")
    .SetShapeFn(shape_inference::NoOutputs)
    .Doc(R"doc(
Converts the given variant tensor to an iterator and stores it in the given resource.

resource_handle: A handle to an iterator resource.
serialized: A variant tensor storing the state of the iterator contained in the
  resource.
)doc");

REGISTER_OP("StatsAggregatorHandle")
    .Output("handle: resource")
    .SetShapeFn(shape_inference::ScalarShape)
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Doc(R"doc(
Creates a statistics manager resource.
)doc");

REGISTER_OP("IteratorSetStatsAggregator")
    .Input("iterator_handle: resource")
    .Input("stats_aggregator_handle: resource")
    .SetShapeFn(shape_inference::NoOutputs)
    .Doc(R"doc(
Associates the given iterator with the given statistics aggregator.
)doc");

REGISTER_OP("StatsAggregatorSummary")
    .Input("iterator: resource")
    .Output("summary: string")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Produces a summary of any statistics recorded by the given statistics manager.
)doc");

}  // namespace tensorflow
