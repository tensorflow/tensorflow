##Profile Model Architecture

* [Profile Model Parameters](#profile-model-parameters)
* [Profile Model Float Operations](#profile-model-float-operations)

###Profile Model Parameters

<b>Notes:</b>
`VariableV2` operation type might contain variables created by TensorFlow
implicitly. User normally don't want to count them as "model capacity".
We can use customized operation type to select a subset of variables.
For example `_trainable_variables` is created automatically by tfprof Python
API. User can also define customized operation type.

```
# parameters are created by operation type 'VariableV2' (For older model,
# it's 'Variable'). scope view is usually suitable in this case.
tfprof> scope -account_type_regexes VariableV2 -max_depth 4 -select params
_TFProfRoot (--/930.58k params)
  global_step (1/1 params)
  init/init_conv/DW (3x3x3x16, 432/864 params)
  pool_logit/DW (64x10, 640/1.28k params)
    pool_logit/DW/Momentum (64x10, 640/640 params)
  pool_logit/biases (10, 10/20 params)
    pool_logit/biases/Momentum (10, 10/10 params)
  unit_last/final_bn/beta (64, 64/128 params)
  unit_last/final_bn/gamma (64, 64/128 params)
  unit_last/final_bn/moving_mean (64, 64/64 params)
  unit_last/final_bn/moving_variance (64, 64/64 params)

# The Python API profiles tf.trainable_variables() instead of VariableV2.
#
# By default, it's printed to stdout. User can update options['output']
# to write to file. The result is always returned as a proto buffer.
param_stats = tf.profiler.profile(
    tf.get_default_graph(),
    options=tf.profiler.ProfileOptionBuilder
        .trainable_variables_parameter())
sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)
```

###Profile Model Float Operations

####Caveats

For an operation to have float operation statistics:

* It must have `RegisterStatistics('flops')` defined in TensorFlow. tfprof
use the definition to calculate float operations. Contributes are welcome.

* It must have known "shape" information for RegisterStatistics('flops')
to calculate the statistics. It is suggested to pass in `-run_meta_path` if
shape is only known during runtime. tfprof can fill in the missing shape with
the runtime shape information from RunMetadata.
Hence, it is suggested to use `-account_displayed_op_only`
option so that you know the statistics are only for the operations printed out.

* If no RunMetadata provided, tfprof count float_ops of each graph node once,
even if it is defined in tf.while_loop. This is because tfprof doesn't know
how many times are run statically. If RunMetadata provided, tfprof calculate
float_ops as float_ops * run_count.



```python
# To profile float opertions in commandline, you need to pass --graph_path
# and --op_log_path.
tfprof> scope -min_float_ops 1 -select float_ops -account_displayed_op_only
node name | # float_ops
_TFProfRoot (--/17.63b flops)
  gradients/pool_logit/xw_plus_b/MatMul_grad/MatMul (163.84k/163.84k flops)
  gradients/pool_logit/xw_plus_b/MatMul_grad/MatMul_1 (163.84k/163.84k flops)
  init/init_conv/Conv2D (113.25m/113.25m flops)
  pool_logit/xw_plus_b (1.28k/165.12k flops)
    pool_logit/xw_plus_b/MatMul (163.84k/163.84k flops)
  unit_1_0/sub1/conv1/Conv2D (603.98m/603.98m flops)
  unit_1_0/sub2/conv2/Conv2D (603.98m/603.98m flops)
  unit_1_1/sub1/conv1/Conv2D (603.98m/603.98m flops)
  unit_1_1/sub2/conv2/Conv2D (603.98m/603.98m flops)

# Some might prefer op view that aggregate by operation type.
tfprof> op -min_float_ops 1 -select float_ops -account_displayed_op_only -order_by float_ops
node name | # float_ops
Conv2D                   17.63b float_ops (100.00%, 100.00%)
MatMul                   491.52k float_ops (0.00%, 0.00%)
BiasAdd                  1.28k float_ops (0.00%, 0.00%)

# You can also do that in Python API.
tf.profiler.profile(
    tf.get_default_graph(),
    options=tf.profiler.ProfileOptionBuilder.float_operation())
```
