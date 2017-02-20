### `tf.contrib.framework.filter_variables(var_list, include_patterns=None, exclude_patterns=None, reg_search=True)` {#filter_variables}

Filter a list of variables using regular expressions.

First includes variables according to the list of include_patterns.
Afterwards, eliminates variables according to the list of exclude_patterns.

For example, one can obtain a list of variables with the weights of all
convolutional layers (depending on the network definition) by:

```python
variables = tf.contrib.framework.get_model_variables()
conv_weight_variables = tf.contrib.framework.filter_variables(
    variables,
    include_patterns=['Conv'],
    exclude_patterns=['biases', 'Logits'])
```

##### Args:


*  <b>`var_list`</b>: list of variables.
*  <b>`include_patterns`</b>: list of regular expressions to include. Defaults to None,
      which means all variables are selected according to the include rules.
      A variable is included if it matches any of the include_patterns.
*  <b>`exclude_patterns`</b>: list of regular expressions to exclude. Defaults to None,
      which means all variables are selected according to the exclude rules.
      A variable is excluded if it matches any of the exclude_patterns.
*  <b>`reg_search`</b>: boolean. If True (default), performs re.search to find matches
      (i.e. pattern can match any substring of the variable name). If False,
      performs re.match (i.e. regexp should match from the beginning of the
      variable name).

##### Returns:

  filtered list of variables.

