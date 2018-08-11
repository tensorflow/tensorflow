# Layers (contrib)
[TOC]

Ops for building neural network layers, regularizers, summaries, etc.

## Higher level ops for building neural network layers

This package provides several ops that take care of creating variables that are
used internally in a consistent way and provide the building blocks for many
common machine learning algorithms.

*   `tf.contrib.layers.avg_pool2d`
*   `tf.contrib.layers.batch_norm`
*   `tf.contrib.layers.convolution2d`
*   `tf.contrib.layers.conv2d_in_plane`
*   `tf.contrib.layers.convolution2d_in_plane`
*   `tf.nn.conv2d_transpose`
*   `tf.contrib.layers.convolution2d_transpose`
*   `tf.nn.dropout`
*   `tf.contrib.layers.flatten`
*   `tf.contrib.layers.fully_connected`
*   `tf.contrib.layers.layer_norm`
*   `tf.contrib.layers.max_pool2d`
*   `tf.contrib.layers.one_hot_encoding`
*   `tf.nn.relu`
*   `tf.nn.relu6`
*   `tf.contrib.layers.repeat`
*   `tf.contrib.layers.safe_embedding_lookup_sparse`
*   `tf.nn.separable_conv2d`
*   `tf.contrib.layers.separable_convolution2d`
*   `tf.nn.softmax`
*   `tf.stack`
*   `tf.contrib.layers.unit_norm`
*   `tf.contrib.layers.embed_sequence`

Aliases for fully_connected which set a default activation function are
available: `relu`, `relu6` and `linear`.

`stack` operation is also available. It builds a stack of layers by applying
a layer repeatedly.

## Regularizers

Regularization can help prevent overfitting. These have the signature
`fn(weights)`. The loss is typically added to
`tf.GraphKeys.REGULARIZATION_LOSSES`.

*   `tf.contrib.layers.apply_regularization`
*   `tf.contrib.layers.l1_regularizer`
*   `tf.contrib.layers.l2_regularizer`
*   `tf.contrib.layers.sum_regularizer`

## Initializers

Initializers are used to initialize variables with sensible values given their
size, data type, and purpose.

*   `tf.contrib.layers.xavier_initializer`
*   `tf.contrib.layers.xavier_initializer_conv2d`
*   `tf.contrib.layers.variance_scaling_initializer`

## Optimization

Optimize weights given a loss.

*   `tf.contrib.layers.optimize_loss`

## Summaries

Helper functions to summarize specific variables or ops.

*   `tf.contrib.layers.summarize_activation`
*   `tf.contrib.layers.summarize_tensor`
*   `tf.contrib.layers.summarize_tensors`
*   `tf.contrib.layers.summarize_collection`

The layers module defines convenience functions `summarize_variables`,
`summarize_weights` and `summarize_biases`, which set the `collection` argument
of `summarize_collection` to `VARIABLES`, `WEIGHTS` and `BIASES`, respectively.

*   `tf.contrib.layers.summarize_activations`

## Feature columns

Feature columns provide a mechanism to map data to a model.

*   `tf.contrib.layers.bucketized_column`
*   `tf.contrib.layers.check_feature_columns`
*   `tf.contrib.layers.create_feature_spec_for_parsing`
*   `tf.contrib.layers.crossed_column`
*   `tf.contrib.layers.embedding_column`
*   `tf.contrib.layers.scattered_embedding_column`
*   `tf.contrib.layers.input_from_feature_columns`
*   `tf.contrib.layers.joint_weighted_sum_from_feature_columns`
*   `tf.contrib.layers.make_place_holder_tensors_for_base_features`
*   `tf.contrib.layers.multi_class_target`
*   `tf.contrib.layers.one_hot_column`
*   `tf.contrib.layers.parse_feature_columns_from_examples`
*   `tf.contrib.layers.parse_feature_columns_from_sequence_examples`
*   `tf.contrib.layers.real_valued_column`
*   `tf.contrib.layers.shared_embedding_columns`
*   `tf.contrib.layers.sparse_column_with_hash_bucket`
*   `tf.contrib.layers.sparse_column_with_integerized_feature`
*   `tf.contrib.layers.sparse_column_with_keys`
*   `tf.contrib.layers.sparse_column_with_vocabulary_file`
*   `tf.contrib.layers.weighted_sparse_column`
*   `tf.contrib.layers.weighted_sum_from_feature_columns`
*   `tf.contrib.layers.infer_real_valued_columns`
*   `tf.contrib.layers.sequence_input_from_feature_columns`
