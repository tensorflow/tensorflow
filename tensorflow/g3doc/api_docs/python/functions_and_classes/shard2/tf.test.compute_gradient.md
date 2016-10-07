### `tf.test.compute_gradient(x, x_shape, y, y_shape, x_init_value=None, delta=0.001, init_targets=None)` {#compute_gradient}

Computes and returns the theoretical and numerical Jacobian.

If `x` or `y` is complex, the Jacobian will still be real but the
corresponding Jacobian dimension(s) will be twice as large.  This is required
even if both input and output is complex since TensorFlow graphs are not
necessarily holomorphic, and may have gradients not expressible as complex
numbers.  For example, if `x` is complex with shape `[m]` and `y` is complex
with shape `[n]`, each Jacobian `J` will have shape `[m * 2, n * 2]` with

    J[:m, :n] = d(Re y)/d(Re x)
    J[:m, n:] = d(Im y)/d(Re x)
    J[m:, :n] = d(Re y)/d(Im x)
    J[m:, n:] = d(Im y)/d(Im x)

##### Args:


*  <b>`x`</b>: a tensor or list of tensors
*  <b>`x_shape`</b>: the dimensions of x as a tuple or an array of ints. If x is a list,
  then this is the list of shapes.

*  <b>`y`</b>: a tensor
*  <b>`y_shape`</b>: the dimensions of y as a tuple or an array of ints.
*  <b>`x_init_value`</b>: (optional) a numpy array of the same shape as "x"
    representing the initial value of x. If x is a list, this should be a list
    of numpy arrays.  If this is none, the function will pick a random tensor
    as the initial value.
*  <b>`delta`</b>: (optional) the amount of perturbation.
*  <b>`init_targets`</b>: list of targets to run to initialize model params.
    TODO(mrry): remove this argument.

##### Returns:

  Two 2-d numpy arrays representing the theoretical and numerical
  Jacobian for dy/dx. Each has "x_size" rows and "y_size" columns
  where "x_size" is the number of elements in x and "y_size" is the
  number of elements in y. If x is a list, returns a list of two numpy arrays.

