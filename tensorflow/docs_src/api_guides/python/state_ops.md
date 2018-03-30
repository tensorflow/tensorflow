# Variables

Note: Functions taking `Tensor` arguments can also take anything accepted by
@{tf.convert_to_tensor}.

[TOC]

## Variables

*   @{tf.Variable}

## Variable helper functions

TensorFlow provides a set of functions to help manage the set of variables
collected in the graph.

*   @{tf.global_variables}
*   @{tf.local_variables}
*   @{tf.model_variables}
*   @{tf.trainable_variables}
*   @{tf.moving_average_variables}
*   @{tf.global_variables_initializer}
*   @{tf.local_variables_initializer}
*   @{tf.variables_initializer}
*   @{tf.is_variable_initialized}
*   @{tf.report_uninitialized_variables}
*   @{tf.assert_variables_initialized}
*   @{tf.assign}
*   @{tf.assign_add}
*   @{tf.assign_sub}

## Saving and Restoring Variables

*   @{tf.train.Saver}
*   @{tf.train.latest_checkpoint}
*   @{tf.train.get_checkpoint_state}
*   @{tf.train.update_checkpoint_state}

## Sharing Variables

TensorFlow provides several classes and operations that you can use to
create variables contingent on certain conditions.

*   @{tf.get_variable}
*   @{tf.get_local_variable}
*   @{tf.VariableScope}
*   @{tf.variable_scope}
*   @{tf.variable_op_scope}
*   @{tf.get_variable_scope}
*   @{tf.make_template}
*   @{tf.no_regularizer}
*   @{tf.constant_initializer}
*   @{tf.random_normal_initializer}
*   @{tf.truncated_normal_initializer}
*   @{tf.random_uniform_initializer}
*   @{tf.uniform_unit_scaling_initializer}
*   @{tf.zeros_initializer}
*   @{tf.ones_initializer}
*   @{tf.orthogonal_initializer}

## Variable Partitioners for Sharding

*   @{tf.fixed_size_partitioner}
*   @{tf.variable_axis_size_partitioner}
*   @{tf.min_max_variable_partitioner}

## Sparse Variable Updates

The sparse update ops modify a subset of the entries in a dense `Variable`,
either overwriting the entries or adding / subtracting a delta.  These are
useful for training embedding models and similar lookup-based networks, since
only a small subset of embedding vectors change in any given step.

Since a sparse update of a large tensor may be generated automatically during
gradient computation (as in the gradient of
@{tf.gather}),
an @{tf.IndexedSlices} class is provided that encapsulates a set
of sparse indices and values.  `IndexedSlices` objects are detected and handled
automatically by the optimizers in most cases.

*   @{tf.scatter_update}
*   @{tf.scatter_add}
*   @{tf.scatter_sub}
*   @{tf.scatter_mul}
*   @{tf.scatter_div}
*   @{tf.scatter_min}
*   @{tf.scatter_max}
*   @{tf.scatter_nd_update}
*   @{tf.scatter_nd_add}
*   @{tf.scatter_nd_sub}
*   @{tf.sparse_mask}
*   @{tf.IndexedSlices}

### Read-only Lookup Tables

*   @{tf.initialize_all_tables}
*   @{tf.tables_initializer}


## Exporting and Importing Meta Graphs

*   @{tf.train.export_meta_graph}
*   @{tf.train.import_meta_graph}

# Deprecated functions (removed after 2017-03-02). Please don't use them.

*   @{tf.all_variables}
*   @{tf.initialize_all_variables}
*   @{tf.initialize_local_variables}
*   @{tf.initialize_variables}
