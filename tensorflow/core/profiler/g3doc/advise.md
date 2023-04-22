## Auto Detect and Advise

tfprof analyzes profiles and generates advice for common issues.

### Run Advise.

```python
# First create a profiler. See profiler tutorials for more details.
profiler = tf.profiler.Profiler(sess.graph)
run_meta = config_pb2.RunMetadata()
_ = sess.run(r1,
             options=config_pb2.RunOptions(
                 trace_level=config_pb2.RunOptions.FULL_TRACE),
             run_metadata=run_meta)
profiler.add_step(1, run_meta)

# Then Start advise.
profiler.advise()

# For one-shot API
tf.profiler.advise(
    sess.graph, run_meta=run_metadata)
```

```shell
# Run advisor on CLI
# See CLI tutorial on generating the files.
tfprof --graph_path=graph.pbtxt \
       --run_meta_path=run_metadata \
       --op_log_path=tfprof_log

tfprof> advise
AcceleratorUtilizationChecker:
device: /job:worker/replica:0/task:0/device:GPU:0 low utilization: 0.03
device: /job:worker/replica:0/task:0/device:GPU:1 low utilization: 0.08
device: /job:worker/replica:0/task:0/device:GPU:2 low utilization: 0.04
device: /job:worker/replica:0/task:0/device:GPU:3 low utilization: 0.21

OperationChecker:
Found operation using NHWC data_format on GPU. Maybe NCHW is faster.

ExpensiveOperationChecker:
top 1 operation type: SoftmaxCrossEntropyWithLogits, cpu: 1.37sec, accelerator: 0us, total: 1.37sec (26.68%)
top 2 operation type: MatMul, cpu: 427.39ms, accelerator: 280.76ms, total: 708.14ms (13.83%)
top 3 operation type: ConcatV2, cpu: 357.83ms, accelerator: 31.80ms, total: 389.63ms (7.61%)
seq2seq_attention_model.py:360:build_graph:self._add_seq2seq(), cpu: 3.16sec, accelerator: 214.84ms, total: 3.37sec
  seq2seq_attention_model.py:293:_add_seq2seq:decoder_outputs, ..., cpu: 2.46sec, accelerator: 3.25ms, total: 2.47sec
    seq2seq_lib.py:181:sampled_sequence_...:average_across_ti..., cpu: 2.46sec, accelerator: 3.24ms, total: 2.47sec
      seq2seq_lib.py:147:sequence_loss_by_...:crossent = loss_f..., cpu: 2.46sec, accelerator: 3.06ms, total: 2.46sec
        seq2seq_attention_model.py:289:sampled_loss_func:num_classes=vsize), cpu: 2.46sec, accelerator: 3.06ms, total: 2.46sec
        seq2seq_attention_model.py:282:sampled_loss_func:labels = tf.resha..., cpu: 164us, accelerator: 0us, total: 164us
      seq2seq_lib.py:148:sequence_loss_by_...:log_perp_list.app..., cpu: 1.33ms, accelerator: 120us, total: 1.45ms
      seq2seq_lib.py:151:sequence_loss_by_...:total_size = tf.a..., cpu: 154us, accelerator: 23us, total: 177us
    seq2seq_lib.py:184:sampled_sequence_...:return cost / tf...., cpu: 97us, accelerator: 8us, total: 105us
      math_ops.py:690:cast:return gen_math_o..., cpu: 62us, accelerator: 3us, total: 65us
      math_ops.py:839:binary_op_wrapper:return func(x, y,..., cpu: 35us, accelerator: 5us, total: 40us
  seq2seq_attention_model.py:192:_add_seq2seq:sequence_length=a..., cpu: 651.56ms, accelerator: 158.92ms, total: 810.48ms
    seq2seq_lib.py:104:bidirectional_rnn:sequence_length, ..., cpu: 306.58ms, accelerator: 73.54ms, total: 380.12ms
      core_rnn.py:195:static_rnn:state_size=cell.s..., cpu: 306.52ms, accelerator: 73.54ms, total: 380.05ms
        rnn.py:218:_rnn_step:_maybe_copy_some_..., cpu: 303.76ms, accelerator: 73.54ms, total: 377.30ms
        rnn.py:216:_rnn_step:time >= max_seque..., cpu: 2.75ms, accelerator: 0us, total: 2.75ms
      core_rnn.py:179:static_rnn:max_sequence_leng..., cpu: 67us, accelerator: 0us, total: 67us
    seq2seq_lib.py:110:bidirectional_rnn:initial_state_bw,..., cpu: 296.21ms, accelerator: 73.54ms, total: 369.75ms
      core_rnn.py:195:static_rnn:state_size=cell.s..., cpu: 296.11ms, accelerator: 73.54ms, total: 369.65ms
        rnn.py:218:_rnn_step:_maybe_copy_some_..., cpu: 292.04ms, accelerator: 73.54ms, total: 365.58ms
        rnn.py:216:_rnn_step:time >= max_seque..., cpu: 4.07ms, accelerator: 0us, total: 4.07ms
      core_rnn.py:178:static_rnn:min_sequence_leng..., cpu: 85us, accelerator: 0us, total: 85us
      core_rnn.py:179:static_rnn:max_sequence_leng..., cpu: 16us, accelerator: 0us, total: 16us
    seq2seq_lib.py:113:bidirectional_rnn:outputs = [tf.con..., cpu: 46.88ms, accelerator: 3.87ms, total: 50.75ms
 ...(omitted)
top 1 graph node: seq2seq/loss/sampled_sequence_loss/sequence_loss_by_example/SoftmaxCrossEntropyWithLogits_11, cpu: 89.92ms, accelerator: 0us, total: 89.92ms
top 2 graph node: train_step/update_seq2seq/output_projection/w/ApplyAdam, cpu: 84.52ms, accelerator: 0us, total: 84.52ms
top 3 graph node: seq2seq/loss/sampled_sequence_loss/sequence_loss_by_example/SoftmaxCrossEntropyWithLogits_19, cpu: 73.02ms, accelerator: 0us, total: 73.02ms
```

### Checker

There is no magic behind advise mode. tfprof builds the profiles first, then
it runs through a list of `Checkers`, each one responsible for checking one
area with the profile and report issues. A `Checker` is like a plugin.

For example:

#### JobChecker (Not Available OSS)

*   Checks RecvTensor RPC latency and bandwidth.
*   Checks CPU/Memory utilization of the job.

#### AcceleratorUtilization Checker
* Checks what percentage of time the accelerator spends on computation.

#### OperationChecker

*   Checks whether the operation runs with optimal options.
*   Checks if there is a better implementation to replace the current operation.

#### ExpensiveOperationChecker

*   Checks the most expensive operation type.
*   Checks the most expensive graph nodes.
*   Checks the most expensive graph-building Python codes.

#### Contribute Your Checker

Follow examples of accelerator_utilization_checker.h



