Adding variables
----------------

Do not add variables using ``tf.Variable([shape], initializer)``, they will fail
to obey certain operations, such as ``assign_add``. Make sure that all variables
are added using a variable scope that is marked as a resource. This can be done
globally:

::

  vscope = tf.get_variable_scope()
  vscope.set_use_resource(True)
  ...
  var = tf.get_variable(name, shape=[...], dtype=tf.float32, initializer=tf.constant_initializer(0.5))
  ...

or locally in a specific scope:

::

  with tf.variable_scope("vs", use_resource=True):
    var = tf.get_variable(name, shape=[...], dtype=tf.float32, initializer=tf.constant_initializer(0.5))

Note on the global_step counter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

More advanced execution control frameworks in Tensorflow use a scalar counter
called ``global_step`` to count the number of iterations of training which have
occurred. This counter is serialized along with the model. It allows the model
to base parameters on the step count, even if the model is run multiple times.

There is an ``add`` operation which adds to the ``global_step`` scalar on each
training pass.  If the ``global_step`` variable is placed on the IPU device,
then this increment operation will occur on the IPU too.  This will cause the
Poplar training engine to be swapped out for the increment engine on each
training step, causing very poor performance.

To avoid this, in the CPU context, use the expression
``tf.train.get_or_create_global_step()`` before you create any special training
sessions.  This will ensure that the global_step variable is on the CPU.

::

  with tf.device("cpu"):
    tf.train.get_or_create_global_step()

