# Inputs and Readers

Note: Functions taking `Tensor` arguments can also take anything accepted by
@{tf.convert_to_tensor}.

[TOC]

## Placeholders

TensorFlow provides a placeholder operation that must be fed with data
on execution.  For more info, see the section on @{$reading_data#Feeding$Feeding data}.

*   @{tf.placeholder}
*   @{tf.placeholder_with_default}

For feeding `SparseTensor`s which are composite type,
there is a convenience function:

*   @{tf.sparse_placeholder}

## Readers

TensorFlow provides a set of Reader classes for reading data formats.
For more information on inputs and readers, see @{$reading_data$Reading data}.

*   @{tf.ReaderBase}
*   @{tf.TextLineReader}
*   @{tf.WholeFileReader}
*   @{tf.IdentityReader}
*   @{tf.TFRecordReader}
*   @{tf.FixedLengthRecordReader}

## Converting

TensorFlow provides several operations that you can use to convert various data
formats into tensors.

*   @{tf.decode_csv}
*   @{tf.decode_raw}

- - -

### Example protocol buffer

TensorFlow's @{$reading_data#standard_tensorflow_format$recommended format for training examples}
is serialized `Example` protocol buffers, [described
here](https://www.tensorflow.org/code/tensorflow/core/example/example.proto).
They contain `Features`, [described
here](https://www.tensorflow.org/code/tensorflow/core/example/feature.proto).

*   @{tf.VarLenFeature}
*   @{tf.FixedLenFeature}
*   @{tf.FixedLenSequenceFeature}
*   @{tf.SparseFeature}
*   @{tf.parse_example}
*   @{tf.parse_single_example}
*   @{tf.parse_tensor}
*   @{tf.decode_json_example}

## Queues

TensorFlow provides several implementations of 'Queues', which are
structures within the TensorFlow computation graph to stage pipelines
of tensors together. The following describe the basic Queue interface
and some implementations.  To see an example use, see @{$threading_and_queues$Threading and Queues}.

*   @{tf.QueueBase}
*   @{tf.FIFOQueue}
*   @{tf.PaddingFIFOQueue}
*   @{tf.RandomShuffleQueue}
*   @{tf.PriorityQueue}

## Conditional Accumulators

*   @{tf.ConditionalAccumulatorBase}
*   @{tf.ConditionalAccumulator}
*   @{tf.SparseConditionalAccumulator}

## Dealing with the filesystem

*   @{tf.matching_files}
*   @{tf.read_file}
*   @{tf.write_file}

## Input pipeline

TensorFlow functions for setting up an input-prefetching pipeline.
Please see the @{$reading_data$reading data how-to}
for context.

### Beginning of an input pipeline

The "producer" functions add a queue to the graph and a corresponding
`QueueRunner` for running the subgraph that fills that queue.

*   @{tf.train.match_filenames_once}
*   @{tf.train.limit_epochs}
*   @{tf.train.input_producer}
*   @{tf.train.range_input_producer}
*   @{tf.train.slice_input_producer}
*   @{tf.train.string_input_producer}

### Batching at the end of an input pipeline

These functions add a queue to the graph to assemble a batch of
examples, with possible shuffling.  They also add a `QueueRunner` for
running the subgraph that fills that queue.

Use @{tf.train.batch} or @{tf.train.batch_join} for batching
examples that have already been well shuffled.  Use
@{tf.train.shuffle_batch} or
@{tf.train.shuffle_batch_join} for examples that would
benefit from additional shuffling.

Use @{tf.train.batch} or @{tf.train.shuffle_batch} if you want a
single thread producing examples to batch, or if you have a
single subgraph producing examples but you want to run it in *N* threads
(where you increase *N* until it can keep the queue full).  Use
@{tf.train.batch_join} or @{tf.train.shuffle_batch_join}
if you have *N* different subgraphs producing examples to batch and you
want them run by *N* threads. Use `maybe_*` to enqueue conditionally.

*   @{tf.train.batch}
*   @{tf.train.maybe_batch}
*   @{tf.train.batch_join}
*   @{tf.train.maybe_batch_join}
*   @{tf.train.shuffle_batch}
*   @{tf.train.maybe_shuffle_batch}
*   @{tf.train.shuffle_batch_join}
*   @{tf.train.maybe_shuffle_batch_join}
