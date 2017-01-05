### `tf.alias_inplace_add(value, loc, update)` {#alias_inplace_add}

Updates input `value` at `loc` with `update`. Aliases value.

   If `loc` is None, `value` and `update` must be the same size.
   ```
   value += update
   ```

   If `loc` is a scalar, `value` has rank 1 higher than `update`
   ```
   value[i, :] += update
   ```

   If `loc` is a vector, `value` has the same rank as `update`
   ```
   value[loc, :] += update
   ```

   Warning: If you use this function you will almost certainly want to add
   a control dependency as done in the implementation of parallel_stack to
   avoid race conditions.

##### Args:


*  <b>`value`</b>: A `Tensor` object that will be updated in-place.
*  <b>`loc`</b>: None, scalar or 1-D `Tensor`.
*  <b>`update`</b>: A `Tensor` of rank one less than `value` if `loc` is a scalar,
          otherwise of rank equal to `value` that contains the new values
          for `value`.

##### Returns:


*  <b>`output`</b>: `value` that has been updated accordingly.

