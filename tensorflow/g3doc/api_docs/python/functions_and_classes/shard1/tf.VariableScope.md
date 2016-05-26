Variable scope object to carry defaults to provide to get_variable.

Many of the arguments we need for get_variable in a variable store are most
easily handled with a context. This object is used for the defaults.

Attributes:
  name: name of the current scope, used as prefix in get_variable.
  initializer: default initializer passed to get_variable.
  regularizer: default regularizer passed to get_variable.
  reuse: Boolean or None, setting the reuse in get_variable.
  caching_device: string, callable, or None: the caching device passed to
    get_variable.
  partitioner: callable or `None`: the partitioner passed to `get_variable`.
  name_scope: The name passed to `tf.name_scope`.
- - -

#### `tf.VariableScope.__init__(reuse, name='', initializer=None, regularizer=None, caching_device=None, partitioner=None, name_scope='')` {#VariableScope.__init__}

Creates a new VariableScope with the given properties.


- - -

#### `tf.VariableScope.caching_device` {#VariableScope.caching_device}




- - -

#### `tf.VariableScope.get_variable(var_store, name, shape=None, dtype=tf.float32, initializer=None, regularizer=None, trainable=True, collections=None, caching_device=None, partitioner=None, validate_shape=True)` {#VariableScope.get_variable}

Gets an existing variable with this name or create a new one.


- - -

#### `tf.VariableScope.initializer` {#VariableScope.initializer}




- - -

#### `tf.VariableScope.name` {#VariableScope.name}




- - -

#### `tf.VariableScope.partitioner` {#VariableScope.partitioner}




- - -

#### `tf.VariableScope.regularizer` {#VariableScope.regularizer}




- - -

#### `tf.VariableScope.reuse` {#VariableScope.reuse}




- - -

#### `tf.VariableScope.reuse_variables()` {#VariableScope.reuse_variables}

Reuse variables in this scope.


- - -

#### `tf.VariableScope.set_caching_device(caching_device)` {#VariableScope.set_caching_device}

Set caching_device for this scope.


- - -

#### `tf.VariableScope.set_initializer(initializer)` {#VariableScope.set_initializer}

Set initializer for this scope.


- - -

#### `tf.VariableScope.set_partitioner(partitioner)` {#VariableScope.set_partitioner}

Set partitioner for this scope.


- - -

#### `tf.VariableScope.set_regularizer(regularizer)` {#VariableScope.set_regularizer}

Set regularizer for this scope.


