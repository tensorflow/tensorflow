### `tf.Print(input_, data, message=None, first_n=None, summarize=None, name=None)` {#Print}

Prints a list of tensors.

This is an identity op with the side effect of printing `data` when
evaluating.

##### Args:


*  <b>`input_`</b>: A tensor passed through this op.
*  <b>`data`</b>: A list of tensors to print out when op is evaluated.
*  <b>`message`</b>: A string, prefix of the error message.
*  <b>`first_n`</b>: Only log `first_n` number of times. Negative numbers log always;
           this is the default.
*  <b>`summarize`</b>: Only print this many entries of each tensor. If None, then a
             maximum of 3 elements are printed per input tensor.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  Same tensor as `input_`.

