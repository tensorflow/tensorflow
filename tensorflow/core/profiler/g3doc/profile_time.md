## Profile Time

* [Times in TensorFlow and tfprof](#times-in-tensorflow-and-tfprof)
* [Profile by Python Code](#profile-by-python-code)
* [Profile by Operation Type](#profile-by-operation-type)
* [Profile by Graph](#profile-by-graph)
* [Profile by Name Scope](#profile-by-name-scope)


### Times in TensorFlow and tfprof
When we run a model, Tensorflow schedules and runs the nodes (operations)
in the graph. An operation can be placed on an accelerator or on CPU.


#### On Accelerator
When an operation is placed on accelerator, it will first be scheduled
by TensorFlow on CPU. Normally, it's the code in OpKernel::Compute.
OpKernel::Compute can decide to dispatch some of the computations on the
accelerator. While some computation (e.g. pre-processing) is still done
in CPU. OpKernel::Compute can dispatch computation on accelerator
and return, or it can also wait for the accelerator to finish.

tfprof reports 3 execution times:

  * <b>accelerator_micros</b>, which is the part of computation time spent on accelerator.
  * <b>cpu_micros</b>, which is the part of computation time spent on cpu, including
    any wait times that might happen if OpKernel::Compute decides to wait.
  * <b>exec_micros</b>, which is the sum of accelerator_micros and cpu_micros.

Since accelerator, such as GPU, usually runs operation asynchronously, you
might notice an operation finishes on cpu before it starts running on
accelerator.

#### On CPU
When an operation is placed on CPU, it will completely run on CPU. Hence,
<b>exec_micros</b> is equal to <b>cpu_micros</b> and <b>accelerator_micros</b>
should be 0.


### Profile by Python Code
```python
# In code view, the time of each line of Python code is the aggregated
# times of all operations created by that line.
# In command line, it requires --graph_path --op_log_path and --run_meta_path.
# --op_log_path provides the code traces information.
# --run_meta_path provides the time information.

tfprof> code -show_name_regexes seq2seq_attention.* -max_depth 10 -select micros -order_by micros
node name | execution time
_TFProfRoot (--/3.74sec)
  seq2seq_attention.py'>:168:run_filename_from...:none (0us/3.74sec)
    seq2seq_attention.py'>:33:_run_code_in_main:none (0us/3.74sec)
      seq2seq_attention.py:316:<module>:app.run() (0us/3.74sec)
        seq2seq_attention.py:270:main:_Train(model, bat... (0us/3.74sec)
          seq2seq_attention.py:128:_Train:model.build_graph() (0us/3.74sec)
            seq2seq_attention_model.py:360:build_graph:self._add_seq2seq() (0us/2.79sec)
              seq2seq_attention_model.py:293:_add_seq2seq:decoder_outputs, ... (0us/2.46sec)
              seq2seq_attention_model.py:192:_add_seq2seq:sequence_length=a... (0us/265.31ms)
              seq2seq_attention_model.py:253:_add_seq2seq:initial_state_att... (0us/50.35ms)
              seq2seq_attention_model.py:173:_add_seq2seq:for x in encoder_... (0us/8.72ms)
              seq2seq_attention_model.py:218:_add_seq2seq:w_t = tf.transpos... (0us/2.39ms)
              ...
            seq2seq_attention_model.py:363:build_graph:self._add_train_o... (0us/949.10ms)
              seq2seq_attention_model.py:307:_add_train_op:tf.gradients(self... (0us/641.44ms)
              seq2seq_attention_model.py:322:_add_train_op:zip(grads, tvars)... (0us/307.56ms)
              ...
            seq2seq_attention_model.py:364:build_graph:self._summaries =... (0us/13us)
            seq2seq_attention_model.py:361:build_graph:self.global_step ... (0us/12us)
            ...
          seq2seq_attention.py:129:_Train:saver = tf.train.... (0us/0us)
          seq2seq_attention.py:140:_Train:global_step=model... (0us/0us)

# Sometimes you want to explore a specific function. You can do that
# with -start_name_regexes.
tfprof> code -start_name_regexes .*_add_seq2seq.* -show_name_regexes seq2seq_attention.* -max_depth 10 -select micros -order_by micros
node name | execution time
_TFProfRoot (--/3.74sec)
  seq2seq_attention_model.py:360:build_graph:self._add_seq2seq() (0us/2.79sec)
    seq2seq_attention_model.py:293:_add_seq2seq:decoder_outputs, ... (0us/2.46sec)
      seq2seq_attention_model.py:289:sampled_loss_func:num_classes=vsize) (0us/2.46sec)
      seq2seq_attention_model.py:282:sampled_loss_func:labels = tf.resha... (0us/164us)

# You can also dive deeper into tensorflow's libraries.
tfprof> code  -max_depth 5 -select micros -order_by micros -start_name_regexes .*_add_seq2seq.* -min_micros 100000
_TFProfRoot (--/3.74sec)
  seq2seq_attention_model.py:360:build_graph:self._add_seq2seq() (0us/2.79sec)
    seq2seq_attention_model.py:293:_add_seq2seq:decoder_outputs, ... (0us/2.46sec)
      seq2seq_lib.py:181:sampled_sequence_...:average_across_ti... (0us/2.46sec)
        seq2seq_lib.py:147:sequence_loss_by_...:crossent = loss_f... (0us/2.46sec)
    seq2seq_attention_model.py:192:_add_seq2seq:sequence_length=a... (0us/265.31ms)
      seq2seq_lib.py:104:bidirectional_rnn:sequence_length, ... (0us/127.27ms)
        core_rnn.py:195:static_rnn:state_size=cell.s... (0us/127.20ms)
      seq2seq_lib.py:110:bidirectional_rnn:initial_state_bw,... (0us/125.96ms)
        core_rnn.py:195:static_rnn:state_size=cell.s... (0us/125.86ms)


# It can also be done in Python API
opts = model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS.copy()
opts['account_type_regexes'] = ['.*']
opts['show_name_regexes'] = ['.*model_analyzer_testlib.py.*']
opts['account_displayed_op_only'] = False
opts['select'] = ['micros']

tfprof_node = model_analyzer.print_model_analysis(
    sess.graph, run_meta, cmd='code', options=opts)
```

You can generate some visualization in code view:
Set ```-output timeline:outfile=<filename>``` to generate timeline instead of stdout.
<left>
![CodeTimeline](code_timeline.png)
</left>


### Profile by Operation Type
```python
# In op view, you can view the aggregated time of each operation type.
tfprof> op -select micros,occurrence -order_by micros
node name | execution time | op occurrence
SoftmaxCrossEntropyWithLogits     1.37sec (100.00%, 36.44%),         30
MatMul                        618.97ms (63.56%, 16.51%),       3450
Add                            273.76ms (47.06%, 7.30%),       2180
Sub                            215.41ms (39.76%, 5.74%),       4372
ConcatV2                       203.88ms (34.01%, 5.44%),       6098
Mul                            134.32ms (28.58%, 3.58%),       9427
ApplyAdam                       92.66ms (25.00%, 2.47%),         27
Switch                          72.43ms (22.53%, 1.93%),      30654
LogUniformCandidateSampler       69.01ms (20.59%, 1.84%),         30
Unique                          53.50ms (18.75%, 1.43%),          2
AddN                            50.10ms (17.33%, 1.34%),       5481

# You might be surprised to see that SoftmaxCrossEntropyWithLogits is
# that expensive. As shown below, it is placed on cpu.
tfprof> op -select micros,device -order_by micros
node name | execution time | assigned devices
SoftmaxCrossEntropyWithLogits     1.37sec (100.00%, 36.44%), /job:worker/replica:0/task:0/cpu:0
MatMul                        618.97ms (63.56%, 16.51%), |/job:worker/replica:0/task:0/cpu:0|/job:worker/replica:0/task:0/device:GPU:0|/job:worker/replica:0/task:0/device:GPU:1|/job:worker/replica:0/task:0/device:GPU:2|/job:worker/replica:0/task:0/device:GPU:3
```


### Profile by Graph

Usually, use graph view to generate a timeline to visualize the result.

In the chrome://tracing UI, click "Flow Event" in "View Options" of upper
right corner to see the flow of tensors.

<left>
TODO(xpan): Show the image correctly in github.
![Timeline](graph_timeline.png)
</left>

tfprof options allow users to generate timeline in some advanced ways.

```python
# Only generate timeline for gpu3 and cpu on workers.
graph -max_depth 10000000 -step 0 -account_type_regexes .*gpu:3.*,.*worker.*cpu:0.* -output timeline:outfile=<filename.json>
generating trace file.

******************************************************
Timeline file is written to <filename.json>.
Open a Chrome browser, enter URL chrome://tracing and load the timeline file.
******************************************************
```

### Profile by Name Scope

Usually scope view allows you to pin point the problematic places if you
have properly named your operations with tf.name_scope or tf.variable_scope.

```python
tfprof> scope -max_depth 30 -select micros -min_micros 100000 -order_by micros
node name | execution time
_TFProfRoot (--/8.12sec)
  tower_3/gradients/tower_3/Conv2d_1a_3x3/convolution_grad/Conv2DBackpropFilter (126.34ms/126.34ms)
  tower_1/gradients/tower_1/Conv2d_1a_3x3/convolution_grad/Conv2DBackpropFilter (125.44ms/125.44ms)
  tower_2/gradients/tower_2/Conv2d_1a_3x3/convolution_grad/Conv2DBackpropFilter (124.85ms/124.85ms)
  tower_0/gradients/tower_0/Conv2d_1a_3x3/convolution_grad/Conv2DBackpropFilter (124.45ms/124.45ms)
```
