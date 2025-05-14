# LiteRT Delegate Culprit Finder Tool

This tool helps debug numeric/NaN issues caused by specific TfLite operations
delegated to a Hardware accelerator. When delegate results differ from CPU
(e.g., producing NaN or significant numeric deviations), pinpointing the
problematic op(s) is hard manually. This tool automates finding the culprit
op(s), simplifying the workflow for delegate developers, regardless of their
experience level.

Currently supports the GPU delegate only.

## Basic usage

```
# Run from root folder of repository.
bazel build -c opt --config=android_arm64\
 tflite/tools/culprit_finder:culprit_finder_main \
 --copt=-DTFLITE_DEBUG_DELEGATE

# Push the culprit_finder tool to the device.
adb push bazel-bin/tflite/tools/culprit_finder/culprit_finder_main /data/local/tmp

# Push the model file to the device.
adb push $MODEL_PATH /data/local/tmp
MODEL_PATH_ON_DEVICE="/data/local/tmp/$(basename "$MODEL_PATH")"

# Run the cuprit finder tool.
adb shell /data/local/tmp/culprit_finder_main \
--model_file=$MODEL_PATH_ON_DEVICE \
--use_gpu=true --gpu_precision_loss_allowed=false \
--find_nan=true --find_numeric_error=true --search_strategy=binary
```

## How it works

While comparing intermediate tensor values between the CPU reference and the
delegated model would be ideal for debugging, it's often impractical for two key
reasons:

1.  Delegates typically don't expose their internal intermediate tensors.
2.  Mapping between the original CPU ops and the delegate's kernels isn't
    readily available and is often not one-to-one, making it hard to know which
    tensors to compare.

Because of these challenges, current debugging workflows usually rely on
comparing the final model output after delegating different subsets of nodes.
This can be done by using the following debug flags that are present in the
`DefaultExecutionProvider`:
`tflite/tools/delegates:default_execution_provider`

-   `first_delegate_node_index`: The first TfLite node to delegate.
-   `last_delegate_node_index`: The last TfLite node to delegate.

### General flags supported

| **Flag**               | **Type** | **Default | **Description**              |
:                        :          : Value**   :                              :
| :--------------------- | :------- | :-------- | :--------------------------- |
| `--model_file`         | `string` |           | Path to the TFLite model     |
:                        :          :           : file to test. Required.      :
| `--search_strategy`    | `string` | linear    | "Search strategy to use      |
:                        :          :           : (binary/linear). <br>The     :
:                        :          :           : default value is dependent   :
:                        :          :           : on the `--find_nan` and      :
:                        :          :           : `--find_numeric_error`       :
:                        :          :           : flags. <ul><li>`linear`\: if :
:                        :          :           : only                         :
:                        :          :           : `--find_numeric_error=true`. :
:                        :          :           : <li>`binary`\: if only       :
:                        :          :           : `--find_nan=true`.</ul>      :
| `--find_nan`           | `bool`   | TRUE      | If specified, search for NAN |
:                        :          :           : culprits.                    :
| `--find_numeric_error` | `bool`   | TRUE      | If specified, search for     |
:                        :          :           : numeric error culprits.      :
| `--min_numeric_error`  | `float`  | 0.0001    | Minimum absolute difference  |
:                        :          :           : to consider an inference as  :
:                        :          :           : an error.                    :

### Search Strategy = Binary Search

This strategy uses binary search to find a range of culprit nodes.

#### Flags supported

| **Flag**                        | **Type** | **Default | **Description** |
:                                 :          : Value**   :                 :
| :------------------------------ | :------- | :-------- | :-------------- |
| `--binary_search_reverse_sweep` | `bool`   | FALSE     | If true, do a   |
:                                 :          :           : binary search   :
:                                 :          :           : in reverse.     :
:                                 :          :           : Default is      :
:                                 :          :           : false. Useful   :
:                                 :          :           : to check if     :
:                                 :          :           : multiple        :
:                                 :          :           : culprits exist. :

#### Sample Output

```
INFO: Model file: [/data/local/tmp/ST5_512_string_quant.tflite]
INFO: Search strategy: [binary]
INFO: Binary search find end first: [0]
INFO: Linear search batch size: [1]
INFO: Linear search node filter: []
INFO: Find NAN: [1]
INFO: Find numeric error: [1]
INFO: Min numeric error: [0.0001]
INFO: Allow GPU precision loss: [0]
INFO: Use gpuv3: [0]
INFO: GPU backend: [cl]
INFO: Use opencl: [0]
INFO: Use webgpu: [0]
INFO: print out all supported flags: [0]
INFO: #threads used for CPU inference: [-1]
INFO: Max number of delegated partitions: [0]
INFO: Min nodes per partition: [0]
INFO: Index of the first node that could be delegated: [0]
INFO: Index of the last node that could be delegated: [2147483647]
INFO: Number of GPU delegate invoke loop iterations: [-1]
INFO: Directory for delegate serialization: []
INFO: Model-specific token/key for delegate serialization.: []
INFO: Use gpu: [1]
INFO: Allow lower precision in gpu: [0]
INFO: Enable running quant models in gpu: [1]
INFO: Prefer maximizing the throughput in gpu: [0]
INFO: GPU backend: [cl]
INFO: Loaded model: /data/local/tmp/ST5_512_string_quant.tflite
INFO: GPU delegate created.
INFO: Reference interpreter prepared
INFO: Model runtime info generated
INFO: Reference inference run completed!
INFO: Loaded OpenCL library with dlopen.
INFO: Looking for start node in node range: [0 - 643] by computing error stats for range [321 - 643]
INFO: Looking for start node in node range: [322 - 643] by computing error stats for range [482 - 643]
INFO: Looking for start node in node range: [483 - 643] by computing error stats for range [563 - 643]
INFO: Looking for start node in node range: [483 - 562] by computing error stats for range [522 - 643]
INFO: Looking for start node in node range: [523 - 562] by computing error stats for range [542 - 643]
INFO: Looking for start node in node range: [523 - 541] by computing error stats for range [532 - 643]
INFO: Looking for start node in node range: [533 - 541] by computing error stats for range [537 - 643]
INFO: Looking for start node in node range: [538 - 541] by computing error stats for range [539 - 643]
INFO: Looking for start node in node range: [538 - 538] by computing error stats for range [538 - 643]
INFO: ### Found max start_node: 537
INFO: Looking for end node in node range: [537 - 643] by computing error stats for range [537 - 590]
INFO: Looking for end node in node range: [537 - 589] by computing error stats for range [537 - 563]
INFO: Looking for end node in node range: [537 - 562] by computing error stats for range [537 - 549]
INFO: Looking for end node in node range: [537 - 548] by computing error stats for range [537 - 542]
INFO: Looking for end node in node range: [537 - 541] by computing error stats for range [537 - 539]
INFO: Looking for end node in node range: [540 - 541] by computing error stats for range [537 - 540]
INFO: Looking for end node in node range: [541 - 541] by computing error stats for range [537 - 541]
INFO: ### Culprit node range: [537 - 542]
INFO: Beginning NodeRangeAnalysis: 537 - 542
INFO: Done with Node range: [537 - 537] with node: 537
INFO: Done with Node range: [537 - 538] with node: 538
INFO: Done with Node range: [537 - 539] with node: 539
INFO: Done with Node range: [537 - 540] with node: 540
INFO: Done with Node range: [537 - 541] with node: 541
INFO: Overall stat:
INFO:   Delegated node range: [[RESHAPE]:537, [SOFTMAX]:542]
INFO:   Min elementwise error: 0
INFO:   Max elementwise error: 0.00192938
INFO:   Total average error: 0.000458489
INFO:   NAN output indices:
INFO: Done with Node range: [537 - 542] with node: 542
INFO: ### Peak memory usage in MB: 2251.04
```

### Search Strategy = Linear Search

#### Flags Supported

| **Flag**                      | **Type** | **Default | **Description** |
:                               :          : Value**   :                 :
| :---------------------------- | :------- | :-------- | :-------------- |
| `--linear_search_batch_size`  | int      | 1         | If provided,    |
:                               :          :           : the culprit     :
:                               :          :           : finder will run :
:                               :          :           : the linear      :
:                               :          :           : search for      :
:                               :          :           : batches of this :
:                               :          :           : size.           :
| `--linear_search_node_filter` | string   |           | A comma         |
:                               :          :           : separated list  :
:                               :          :           : of TfLite Op    :
:                               :          :           : types that the  :
:                               :          :           : culprit finder  :
:                               :          :           : will run on.    :

#### Sample Output

```
adb shell /data/local/tmp/culprit_finder_main --model_file=$MODEL_PATH_ON_DEVICE --use_gpu=true --gpu_precision_loss_allowed=false --find_nan=true --find_numeric_error=true --search_strategy=linear --linear_search_node_filter=TANH 2> /dev/null

INFO: Model file: [/data/local/tmp/ST5_512_string_quant.tflite]
INFO: Search strategy: [linear]
INFO: Binary search find end first: [0]
INFO: Linear search batch size: [1]
INFO: Linear search node filter: [TANH]
INFO: Find NAN: [1]
INFO: Find numeric error: [1]
INFO: Min numeric error: [0.0001]
INFO: Allow GPU precision loss: [0]
INFO: Use gpuv3: [0]
INFO: GPU backend: [cl]
INFO: Use opencl: [0]
INFO: Use webgpu: [0]
INFO: print out all supported flags: [0]
INFO: #threads used for CPU inference: [-1]
INFO: Max number of delegated partitions: [0]
INFO: Min nodes per partition: [0]
INFO: Index of the first node that could be delegated: [0]
INFO: Index of the last node that could be delegated: [2147483647]
INFO: Number of GPU delegate invoke loop iterations: [-1]
INFO: Directory for delegate serialization: []
INFO: Model-specific token/key for delegate serialization.: []
INFO: Use gpu: [1]
INFO: Allow lower precision in gpu: [0]
INFO: Enable running quant models in gpu: [1]
INFO: Prefer maximizing the throughput in gpu: [0]
INFO: GPU backend: [cl]
INFO: Loaded model: /data/local/tmp/ST5_512_string_quant.tflite
INFO: GPU delegate created.
INFO: Reference interpreter prepared
INFO: Model runtime info generated
INFO: Reference inference run completed!
INFO: Nodes size: 643
INFO: Subgraphs size: 1
INFO: Execution plan size: 643
INFO: ### Node ranges size: 12
INFO: Loaded OpenCL library with dlopen.
INFO: Done with Node range: [65 - 65]
INFO: Done with Node range: [115 - 115]
INFO: Done with Node range: [165 - 165]
INFO: Done with Node range: [215 - 215]
INFO: Done with Node range: [265 - 265]
INFO: Done with Node range: [315 - 315]
INFO: Done with Node range: [365 - 365]
INFO: Done with Node range: [415 - 415]
INFO: Done with Node range: [465 - 465]
INFO: Done with Node range: [515 - 515]
INFO: Done with Node range: [565 - 565]
INFO: Done with Node range: [615 - 615]
INFO: CULPRIT FINDER REPORT
INFO: -------------------------------------------------------------
INFO: Total number of nodes with errors: 9
INFO: Top 5 node ranges sorted by error (node_range, op_name(s), input/output shapes, total_error):
INFO: 115 - 115, TANH - TANH, (FLOAT32[1,512,2048,],) -> (FLOAT32[1,512,2048,],), 0.00141406
INFO: 65 - 65, TANH - TANH, (FLOAT32[1,512,2048,],) -> (FLOAT32[1,512,2048,],), 0.001414
INFO: 165 - 165, TANH - TANH, (FLOAT32[1,512,2048,],) -> (FLOAT32[1,512,2048,],), 0.00120078
INFO: 215 - 215, TANH - TANH, (FLOAT32[1,512,2048,],) -> (FLOAT32[1,512,2048,],), 0.00119554
INFO: 265 - 265, TANH - TANH, (FLOAT32[1,512,2048,],) -> (FLOAT32[1,512,2048,],), 0.00112813
INFO: -------------------------------------------------------------
INFO: Top 5 node(s) with most errors (op_name, count):
INFO: TANH, 9
INFO: -------------------------------------------------------------
INFO: ### Peak memory usage in MB: 2221.22
```
