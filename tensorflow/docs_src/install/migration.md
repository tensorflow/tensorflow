# Transition to TensorFlow 1.0


The APIs in TensorFlow 1.0 have changed in ways that are not all backwards
compatible.  That is, TensorFlow programs that worked on TensorFlow 0.n won't
necessarily work on TensorFlow 1.0.  We have made this API changes to ensure an
internally-consistent API, and do not plan to make backwards-breaking changes
throughout the 1.N lifecycle.

This guide walks you through the major changes in the API and how to
automatically upgrade your programs for TensorFlow 1.0.  This guide not
only steps you through the changes but also explains why we've made them.

## How to upgrade

If you would like to automatically  port your code to 1.0, you can try our
`tf_upgrade.py` script. While this script handles many cases, manual changes
are sometimes necessary.
  Get this script from our
[GitHub tree](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/compatibility).

To convert a single 0.n TensorFlow source file to 1.0, enter a
command of the following format:

<pre>
$ <b>python tf_upgrade.py --infile</b> <i>InputFile</i> <b>--outfile</b> <i>OutputFile</i>
</pre>

For example, the following command converts a 0.n TensorFlow
program named `test.py` to a 1.0 TensorFlow program named `test_1.0.py`:

<pre>
$ <b>python tf_upgrade.py --infile test.py --outfile test_1.0.py</b>
</pre>

The `tf_upgrade.py` script also generates a file named `report.txt`, which
details all the changes it performed and makes additional suggestions about
changes you might need to make manually.

To upgrade a whole directory of 0.n TensorFlow programs to 1.0,
enter a command having the following format:

<pre>
$ <b>python tf_upgrade.py --intree</b> <i>InputDir</i> <b>--outtree</b> <i>OutputDir</i>
</pre>

For example, the following command converts all the 0.n TensorFlow programs
in the `/home/user/cool` directory, creating their 1.0 equivalents in
the `/home/user/cool_1.0` directory:

<pre>
$ <b>python tf_upgrade.py --intree /home/user/cool --outtree /home/user/cool_1.0</b>
</pre>

### Limitations

There are a few things to watch out for. Specifically:

 * You must manually fix any instances of `tf.reverse()`.
   The `tf_upgrade.py` script will warn you about `tf.reverse()` in
   stdout and in the `report.txt` file.
 * On reordered arguments, `tf_upgrade.py` tries to minimally reformat
   your code, so it cannot automatically change the actual argument order.
   Instead, `tf_upgrade.py` makes your function invocations order-independent
   by introducing keyword arguments.
 * Constructions like `tf.get_variable_scope().reuse_variables()`
   will likely not work. We recommend deleting those lines and replacing
   them with lines such as the following:

   <pre class="prettyprint">
   with tf.variable_scope(tf.get_variable_scope(), reuse=True):
     ...
   </pre>

 * Analogously to `tf.pack` and  `tf.unpack`, we're renamed
   `TensorArray.pack` and `TensorArray.unpack` to
   `TensorArray.stack` and `TensorArray.unstack`. However, `TensorArray.pack`
   and `TensorArray.unpack` cannot be detected lexically since they are
   indirectly related to the `tf` namespace e.g.
   `foo = tf.TensorArray(); foo.unpack()`

## Upgrading your code manually

Instead of running `tf_upgrade.py`, you may manually upgrade your code.
The remainder of this document provides a comprehensive list of
all backward incompatible changes made in TensorFlow 1.0.


### Variables

Variable functions have been made more consistent and less confusing.

* `tf.VARIABLES`
    * should be renamed to `tf.GLOBAL_VARIABLES`
* `tf.all_variables`
    * should be renamed to `tf.global_variables`
* `tf.initialize_all_variables`
    * should be renamed to `tf.global_variables_initializer`
* `tf.initialize_local_variables`
    * should be renamed to `tf.local_variables_initializer`
* `tf.initialize_variables`
    * should be renamed to `tf.variables_initializer`

### Summary functions

Summary functions have been consolidated under the `tf.summary` namespace.

* `tf.audio_summary`
    * should be renamed to `tf.summary.audio`
* `tf.contrib.deprecated.histogram_summary`
    * should be renamed to `tf.summary.histogram`
* `tf.contrib.deprecated.scalar_summary`
    * should be renamed to `tf.summary.scalar`
* `tf.histogram_summary`
    * should be renamed to `tf.summary.histogram`
* `tf.image_summary`
    * should be renamed to `tf.summary.image`
* `tf.merge_all_summaries`
    * should be renamed to `tf.summary.merge_all`
* `tf.merge_summary`
    * should be renamed to `tf.summary.merge`
* `tf.scalar_summary`
    * should be renamed to `tf.summary.scalar`
* `tf.train.SummaryWriter`
    * should be renamed to `tf.summary.FileWriter`

### Numeric differences


Integer division and `tf.floordiv` now uses flooring semantics. This is to
make the results of `np.divide` and `np.mod` consistent with `tf.divide` and
`tf.mod`, respectively. In addition we have changed the rounding algorithm
used by `tf.round` to match NumPy.


* `tf.div`

    * The semantics of `tf.divide` division have been changed to match Python
semantics completely. That is, `/` in Python 3     and future division mode in
Python 2 will produce floating point numbers always, `//` will produce floored
division.     However, even `tf.div` will produce floored integer division.
To force C-style truncation semantics, you must use `tf.truncatediv`.

    * Consider changing your code to use `tf.divide`, which follows Python semantics for promotion.

* `tf.mod`

    * The semantics of `tf.mod` have been changed to match Python semantics. In
particular, flooring semantics are used for     integers. If you wish to have
C-style truncation mod (remainders), you can use `tf.truncatemod`


The old and new behavior of division can be summarized with this table:

| Expr                | TF 0.11 (py2) | TF 0.11 (py3) | TF 1.0 (py2) | TF 1.0 (py3) |
|---------------------|---------------|---------------|--------------|--------------|
| tf.div(3,4)         | 0             | 0             | 0            | 0            |
| tf.div(-3,4)        | 0             | 0             | -1           | -1           |
| tf.mod(-3,4)        | -3            | -3            | 1            | 1            |
| -3/4                | 0             | -0.75         | -1           | -0.75        |
| -3/4tf.divide(-3,4) | N/A           | N/A           | -0.75        | -1           |

The old and new behavior of rounding can be summarized with this table:

| Input | Python | NumPy | C++ round() | TensorFlow 0.11(floor(x+.5)) | TensorFlow 1.0 |
|-------|--------|-------|-------------|------------------------------|----------------|
| -3.5  | -4     | -4    | -4          | -3                           | -4             |
| -2.5  | -2     | -2    | -3          | -2                           | -2             |
| -1.5  | -2     | -2    | -2          | -1                           | -2             |
| -0.5  | 0      | 0     | -1          | 0                            | 0              |
| 0.5   | 0      | 0     | 1           | 1                            | 0              |
| 1.5   | 2      | 2     | 2           | 2                            | 2              |
| 2.5   | 2      | 2     | 3           | 3                            | 2              |
| 3.5   | 4      | 4     | 4           | 4                            | 4              |



### NumPy matching names


Many functions have been renamed to match NumPy. This was done to make the
transition between NumPy and TensorFlow as easy as possible. There are still
numerous cases where functions do not match, so this is far from a hard and
fast rule, but we have removed several commonly noticed inconsistencies.

* `tf.inv`
    * should be renamed to `tf.reciprocal`
    * This was done to avoid confusion with NumPy's matrix inverse `np.inv`
* `tf.list_diff`
    * should be renamed to `tf.setdiff1d`
* `tf.listdiff`
    * should be renamed to `tf.setdiff1d`
* `tf.mul`
    * should be renamed to `tf.multiply`
* `tf.neg`
    * should be renamed to `tf.negative`
* `tf.select`
    * should be renamed to `tf.where`
    * `tf.where` now takes 3 arguments or 1 argument, just like `np.where`
* `tf.sub`
    * should be renamed to `tf.subtract`

### NumPy matching arguments

Arguments for certain TensorFlow 1.0 methods now match arguments in certain
NumPy methods.  To achieve this, TensorFlow 1.0 has changed keyword arguments
and reordered some arguments. Notably, TensorFlow 1.0 now uses `axis` rather
than `dimension`. TensorFlow 1.0 aims to keep the tensor argument first on
operations that modify Tensors. (see the `tf.concat` change).


* `tf.argmax`
    * keyword argument `dimension` should be renamed to `axis`
* `tf.argmin`
    * keyword argument `dimension` should be renamed to `axis`
* `tf.concat`
    * keyword argument `concat_dim` should be renamed to `axis`
    * arguments have been reordered to `tf.concat(values, axis, name='concat')`.
* `tf.count_nonzero`
    * keyword argument `reduction_indices` should be renamed to `axis`
* `tf.expand_dims`
    * keyword argument `dim` should be renamed to `axis`
* `tf.reduce_all`
    * keyword argument `reduction_indices` should be renamed to `axis`
* `tf.reduce_any`
    * keyword argument `reduction_indices` should be renamed to `axis`
* `tf.reduce_join`
    * keyword argument `reduction_indices` should be renamed to `axis`
* `tf.reduce_logsumexp`
    * keyword argument `reduction_indices` should be renamed to `axis`
* `tf.reduce_max`
    * keyword argument `reduction_indices` should be renamed to `axis`
* `tf.reduce_mean`
    * keyword argument `reduction_indices` should be renamed to `axis`
* `tf.reduce_min`
    * keyword argument `reduction_indices` should be renamed to `axis`
* `tf.reduce_prod`
    * keyword argument `reduction_indices` should be renamed to `axis`
* `tf.reduce_sum`
    * keyword argument `reduction_indices` should be renamed to `axis`
* `tf.reverse`
    * `tf.reverse` used to take a 1D `bool` tensor to control which dimensions were reversed. Now we use a Tensor of axis indices.
    * For example `tf.reverse(a, [True, False, True])` now must be `tf.reverse(a, [0, 2])`
* `tf.reverse_sequence`
    * keyword argument `batch_dim` should be renamed to `batch_axis`
    * keyword argument `seq_dim` should be renamed to `seq_axis`
* `tf.sparse_concat`
    * keyword argument `concat_dim` should be renamed to `axis`
* `tf.sparse_reduce_sum`
    * keyword argument `reduction_axes` should be renamed to `axis`
* `tf.sparse_reduce_sum_sparse`
    * keyword argument `reduction_axes` should be renamed to `axis`
* `tf.sparse_split`
    * keyword argument `split_dim` should be renamed to `axis`
    * arguments have been reordered to `tf.sparse_split(keyword_required=KeywordRequired(), sp_input=None, num_split=None, axis=None, name=None, split_dim=None)`.
* `tf.split`
    * keyword argument `split_dim` should be renamed to `axis`
    * keyword argument `num_split` should be renamed to `num_or_size_splits`
    * arguments have been reordered to `tf.split(value, num_or_size_splits, axis=0, num=None, name='split')`.
* `tf.squeeze`
    * keyword argument `squeeze_dims` should be renamed to `axis`
* `tf.svd`
    * arguments have been reordered to `tf.svd(tensor, full_matrices=False, compute_uv=True, name=None)`.

### Simplified math variants

Batched versions of math operations have been removed. Now the functionality is
contained in the non-batched versions. Similarly,`tf.complex_abs` has had its
functionality moved to `tf.abs`

* `tf.batch_band_part`
    * should be renamed to `tf.band_part`
* `tf.batch_cholesky`
    * should be renamed to `tf.cholesky`
* `tf.batch_cholesky_solve`
    * should be renamed to `tf.cholesky_solve`
* `tf.batch_fft`
    * should be renamed to `tf.fft`
* `tf.batch_fft3d`
    * should be renamed to `tf.fft3d`
* `tf.batch_ifft`
    * should be renamed to `tf.ifft`
* `tf.batch_ifft2d`
    * should be renamed to `tf.ifft2d`
* `tf.batch_ifft3d`
    * should be renamed to `tf.ifft3d`
* `tf.batch_matmul`
    * should be renamed to `tf.matmul`
* `tf.batch_matrix_determinant`
    * should be renamed to `tf.matrix_determinant`
* `tf.batch_matrix_diag`
    * should be renamed to `tf.matrix_diag`
* `tf.batch_matrix_inverse`
    * should be renamed to `tf.matrix_inverse`
* `tf.batch_matrix_solve`
    * should be renamed to `tf.matrix_solve`
* `tf.batch_matrix_solve_ls`
    * should be renamed to `tf.matrix_solve_ls`
* `tf.batch_matrix_transpose`
    * should be renamed to `tf.matrix_transpose`
* `tf.batch_matrix_triangular_solve`
    * should be renamed to `tf.matrix_triangular_solve`
* `tf.batch_self_adjoint_eig`
    * should be renamed to `tf.self_adjoint_eig`
* `tf.batch_self_adjoint_eigvals`
    * should be renamed to `tf.self_adjoint_eigvals`
* `tf.batch_set_diag`
    * should be renamed to `tf.set_diag`
* `tf.batch_svd`
    * should be renamed to `tf.svd`
* `tf.complex_abs`
    * should be renamed to `tf.abs`

### Misc Changes

Several other changes have been made, including the following:

* `tf.image.per_image_whitening`
    * should be renamed to `tf.image.per_image_standardization`
* `tf.nn.sigmoid_cross_entropy_with_logits`
    * arguments have been reordered to `tf.nn.sigmoid_cross_entropy_with_logits(_sentinel=None, labels=None, logits=None, name=None)`.
* `tf.nn.softmax_cross_entropy_with_logits`
    * arguments have been reordered to `tf.nn.softmax_cross_entropy_with_logits(_sentinel=None, labels=None, logits=None, dim=-1, name=None)`.
* `tf.nn.sparse_softmax_cross_entropy_with_logits`
    * arguments have been reordered to `tf.nn.sparse_softmax_cross_entropy_with_logits(_sentinel=None, labels=None, logits=None, name=None)`.
* `tf.ones_initializer`
    * should be changed to a function call i.e. `tf.ones_initializer()`
* `tf.pack`
    * should be renamed to `tf.stack`
* `tf.round`
    * The semantics of `tf.round` now match Banker's rounding.
* `tf.unpack`
    * should be renamed to `tf.unstack`
* `tf.zeros_initializer`
    * should be changed to a function call i.e. `tf.zeros_initializer()`

