### `tf.Assert(condition, data, summarize=None, name=None)` {#Assert}

Asserts that the given condition is true.

If `condition` evaluates to false, print the list of tensors in `data`.
`summarize` determines how many entries of the tensors to print.

##### Args:


*  <b>`condition`</b>: The condition to evaluate.
*  <b>`data`</b>: The tensors to print out when condition is false.
*  <b>`summarize`</b>: Print this many entries of each tensor.
*  <b>`name`</b>: A name for this operation (optional).

