## Profile Memory

It is generally a good idea to visualize the memory usage in timeline.
It allows you to see the memory consumption of each GPU over time.

```python
#To get memory information, you need --graph_path and --run_meta_path
tfprof> graph -max_depth 10000000 -step 0 -account_type_regexes .* -output timeline:outfile=<filename>
generating trace file.

******************************************************
Timeline file is written to <filename>
Open a Chrome browser, enter URL chrome://tracing and load the timeline file.
******************************************************
```

<left>
![Timeline](graph_timeline.png)
</left>


```python
# You can also visualize the memory information through other methods.

# With op view, it shows you the aggregated output tensor bytes of each
# operation type.
tfprof> op -select bytes -order_by bytes
node name | requested bytes
Identity                   32515.37MB (100.00%, 27.02%)
FusedBatchNormGrad           10802.14MB (72.98%, 8.98%)
FusedBatchNorm               10517.52MB (64.01%, 8.74%)
Conv2D                       10509.25MB (55.27%, 8.73%)
Conv2DBackpropInput           9701.39MB (46.54%, 8.06%)
ReluGrad                      9206.45MB (38.48%, 7.65%)
Relu                          8462.80MB (30.83%, 7.03%)
DepthwiseConv2dNativeBackpropInput     7899.35MB (23.80%, 6.56%)
DepthwiseConv2dNative         7425.17MB (17.23%, 6.17%)
MaxPoolGrad                   3015.44MB (11.06%, 2.51%)
AddN                           2741.49MB (8.56%, 2.28%)

# With scope view, you can see the operations that outputs largest tensors.
tfprof> scope -order_by bytes -select bytes -min_bytes 100000000
node name | requested bytes
_TFProfRoot (--/120356.38MB)
  tower_3/SepConv2d_2b_3x3/separable_conv2d (346.85MB/854.00MB)
    tower_3/SepConv2d_2b_3x3/separable_conv2d/depthwise (507.15MB/507.15MB)
  tower_0/SepConv2d_2b_3x3/separable_conv2d (346.85MB/693.71MB)
    tower_0/SepConv2d_2b_3x3/separable_conv2d/depthwise (346.85MB/346.85MB)
  tower_2/SepConv2d_2b_3x3/separable_conv2d (346.85MB/693.71MB)
    tower_2/SepConv2d_2b_3x3/separable_conv2d/depthwise (346.85MB/346.85MB)
  tower_1/SepConv2d_2b_3x3/separable_conv2d (346.85MB/693.71MB)
    tower_1/SepConv2d_2b_3x3/separable_conv2d/depthwise (346.85MB/346.85MB)
  tower_3/SepConv2d_2a_3x3/separable_conv2d (346.85MB/520.28MB)
    tower_3/SepConv2d_2a_3x3/separable_conv2d/depthwise (173.43MB/173.43MB)
  tower_2/SepConv2d_2a_3x3/separable_conv2d (346.85MB/520.28MB)
    tower_2/SepConv2d_2a_3x3/separable_conv2d/depthwise (173.43MB/173.43MB)
  tower_0/SepConv2d_2a_3x3/separable_conv2d (346.85MB/520.28MB)
    tower_0/SepConv2d_2a_3x3/separable_conv2d/depthwise (173.43MB/173.43MB)
  ...

# code view.
tfprof> code  -max_depth 10 -select bytes -order_by bytes -start_name_regexes .*seq2seq.* -min_bytes 1
node name | requested bytes
_TFProfRoot (--/74148.60MB)
  seq2seq_attention.py'>:168:run_filename_from...:none (0B/74148.60MB)
    seq2seq_attention.py'>:33:_run_code_in_main:none (0B/74148.60MB)
      seq2seq_attention.py:316:<module>:app.run() (0B/74148.60MB)
        app.py:432:run:_run_main(main or... (0B/74148.60MB)
          app.py:352:_run_main:sys.exit(main(arg... (0B/74148.60MB)
            seq2seq_attention.py:270:main:_Train(model, bat... (0B/74148.60MB)
              seq2seq_attention.py:128:_Train:model.build_graph() (0B/74148.60MB)
                seq2seq_attention_model.py:363:build_graph:self._add_train_o... (0B/48931.86MB)
                  seq2seq_attention_model.py:307:_add_train_op:tf.gradients(self... (0B/46761.06MB)
                  seq2seq_attention_model.py:322:_add_train_op:zip(grads, tvars)... (0B/2170.80MB)
                  seq2seq_attention_model.py:312:_add_train_op:tf.train.exponent... (0B/2.56KB)
                  seq2seq_attention_model.py:308:_add_train_op:tf.summary.scalar... (0B/64B)
                  seq2seq_attention_model.py:320:_add_train_op:tf.summary.scalar... (0B/64B)
                seq2seq_attention_model.py:360:build_graph:self._add_seq2seq() (0B/25216.74MB)
                  seq2seq_attention_model.py:192:_add_seq2seq:sequence_length=a... (0B/21542.55MB)
```