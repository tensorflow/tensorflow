# tfprof: TensorFlow Profiler and Beyond

### Features

* Profile model architectures
  * parameters, tensor shapes, float operations, device placement, etc.
* Profile model performance
  * execution time, memory consumption
  * Profile multiple steps.
* Auto profile and advise.
  * accelerator utilization check
  * expensive operation check
  * operation configuration check
  * distributed runtime check (Not OSS)

### Interfaces

* Python API
* Command Line
* Visualization
* C++ API (Not public, contact us if needed.)

### Views and Options

tfprof provides 4 different views to organize the statistics.

    *  code view: operations are grouped by Python codes that generate them.
    *  op view: operations are grouped by operation type (E.g. MatMul, Conv2D).
    *  scope view: operations are organized based on name scope hierarchies.
    *  graph view: operations are organized based on input/output.

tfprof provides options to help user select, filter and order statistics.
See [Options](g3doc/options.md) for detail instructions.

```
-max_depth                  10
-min_bytes                  0
-min_micros                 0
-min_params                 0
-min_float_ops              0
-min_occurrence             0
-step                       -1
-order_by                   name
-account_type_regexes       .*
-start_name_regexes         .*
-trim_name_regexes
-show_name_regexes          .*
-hide_name_regexes
-account_displayed_op_only  false
-select                     params
-output                     stdout:
```

### Tutorials

*  [Python API](g3doc/python_api.md)
*  [Command Line Interface](g3doc/command_line.md)
*  [Profile Time](g3doc/profile_time.md)
*  [Profile Memory](g3doc/profile_memory.md)
*  [Profile Model Architecture](g3doc/profile_model_architecture.md)
*  [Auto Detect and Advise](g3doc/advise.md)
*  [Options](g3doc/options.md)

## Demo

### Attribute TensorFlow graph running time to your Python codes.
```shell
tfprof> code -max_depth 1000 -show_name_regexes .*model_analyzer.*py.* -select micros -account_type_regexes .* -order_by micros
_TFProfRoot (0us/22.44ms)
  model_analyzer_test.py:149:run_filename_as_m...:none (0us/22.44ms)
    model_analyzer_test.py:33:_run_code_in_main:none (0us/22.44ms)
      model_analyzer_test.py:208:<module>:test.main() (0us/22.44ms)
        model_analyzer_test.py:132:testComplexCodeView:x = lib.BuildFull... (0us/22.44ms)
          model_analyzer_testlib.py:63:BuildFullModel:return sgd_op.min... (0us/21.83ms)
          model_analyzer_testlib.py:58:BuildFullModel:cell, array_ops.c... (0us/333us)
          model_analyzer_testlib.py:54:BuildFullModel:seq.append(array_... (0us/254us)
            model_analyzer_testlib.py:42:BuildSmallModel:x = nn_ops.conv2d... (0us/134us)
            model_analyzer_testlib.py:46:BuildSmallModel:initializer=init_... (0us/40us)
            ...
          model_analyzer_testlib.py:61:BuildFullModel:loss = nn_ops.l2_... (0us/28us)
          model_analyzer_testlib.py:60:BuildFullModel:target = array_op... (0us/0us)
        model_analyzer_test.py:134:testComplexCodeView:sess.run(variable... (0us/0us)
```

### Show your model variables and the number of parameters.
```
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
```

### Show the most expensive operation types.
```
tfprof> op -select micros,bytes,occurrence -order_by micros
SoftmaxCrossEntropyWithLogits      36.58MB (100.00%, 0.05%),      1.37sec (100.00%, 23.56%),         30
MatMul                        2720.57MB (99.95%, 3.66%),      988.90ms (76.44%, 17.05%),       3450
ConcatV2                       741.37MB (96.29%, 1.00%),       421.44ms (59.38%, 7.27%),       6098
Mul                           3957.24MB (95.29%, 5.33%),       418.90ms (52.12%, 7.22%),       9427
Add                            740.05MB (89.96%, 1.00%),       335.26ms (44.89%, 5.78%),       2180
Sub                             32.46MB (88.97%, 0.04%),       216.44ms (39.11%, 3.73%),       4372
AddN                           733.21MB (88.92%, 0.99%),       208.46ms (35.38%, 3.59%),       5481
Slice                          708.07MB (87.94%, 0.95%),       205.27ms (31.78%, 3.54%),       7277
Fill                           954.27MB (86.98%, 1.28%),       154.50ms (28.24%, 2.66%),       9686
Select                         312.33MB (85.70%, 0.42%),       123.04ms (25.58%, 2.12%),       5746
Sigmoid                        152.57MB (85.28%, 0.21%),        96.66ms (23.46%, 1.67%),       2970
```

### Visualize time and memory.
<left>
[CodeTimeline](g3doc/graph_timeline.png)
</left>

### Teams

* Xin Pan (xpan@google.com, github: panyx0718)
* Jon Shlens
* Yao Zhang
