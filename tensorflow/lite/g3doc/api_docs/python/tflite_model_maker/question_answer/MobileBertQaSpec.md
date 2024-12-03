page_type: reference
description: Creates MobileBert model spec for the question answer task. See also: <a href="../../tflite_model_maker/question_answer/BertQaSpec"><code>tflite_model_maker.question_answer.BertQaSpec</code></a>.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_model_maker.question_answer.MobileBertQaSpec" />
<meta itemprop="path" content="Stable" />
</div>

# tflite_model_maker.question_answer.MobileBertQaSpec

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Creates MobileBert model spec for the question answer task. See also: <a href="../../tflite_model_maker/question_answer/BertQaSpec"><code>tflite_model_maker.question_answer.BertQaSpec</code></a>.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_model_maker.question_answer.MobileBertQaSpec(
    *,
    uri=&#x27;https://tfhub.dev/google/mobilebert/uncased_L-24_H-128_B-512_A-4_F-4_OPT/1&#x27;,
    model_dir=None,
    seq_len=384,
    query_len=64,
    doc_stride=128,
    dropout_rate=0.1,
    initializer_range=0.02,
    learning_rate=4e-05,
    distribution_strategy=&#x27;off&#x27;,
    num_gpus=-1,
    tpu=&#x27;&#x27;,
    trainable=True,
    predict_batch_size=8,
    do_lower_case=True,
    is_tf2=False,
    tflite_input_name=None,
    tflite_output_name=None,
    init_from_squad_model=False,
    default_batch_size=32,
    name=&#x27;MobileBert&#x27;
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`uri`<a id="uri"></a>
</td>
<td>
TF-Hub path/url to Bert module.
</td>
</tr><tr>
<td>
`model_dir`<a id="model_dir"></a>
</td>
<td>
The location of the model checkpoint files.
</td>
</tr><tr>
<td>
`seq_len`<a id="seq_len"></a>
</td>
<td>
Length of the sequence to feed into the model.
</td>
</tr><tr>
<td>
`query_len`<a id="query_len"></a>
</td>
<td>
Length of the query to feed into the model.
</td>
</tr><tr>
<td>
`doc_stride`<a id="doc_stride"></a>
</td>
<td>
The stride when we do a sliding window approach to take chunks
of the documents.
</td>
</tr><tr>
<td>
`dropout_rate`<a id="dropout_rate"></a>
</td>
<td>
The rate for dropout.
</td>
</tr><tr>
<td>
`initializer_range`<a id="initializer_range"></a>
</td>
<td>
The stdev of the truncated_normal_initializer for
initializing all weight matrices.
</td>
</tr><tr>
<td>
`learning_rate`<a id="learning_rate"></a>
</td>
<td>
The initial learning rate for Adam.
</td>
</tr><tr>
<td>
`distribution_strategy`<a id="distribution_strategy"></a>
</td>
<td>
 A string specifying which distribution strategy to
use. Accepted values are 'off', 'one_device', 'mirrored',
'parameter_server', 'multi_worker_mirrored', and 'tpu' -- case
insensitive. 'off' means not to use Distribution Strategy; 'tpu' means
to use TPUStrategy using `tpu_address`.
</td>
</tr><tr>
<td>
`num_gpus`<a id="num_gpus"></a>
</td>
<td>
How many GPUs to use at each worker with the
DistributionStrategies API. The default is -1, which means utilize all
available GPUs.
</td>
</tr><tr>
<td>
`tpu`<a id="tpu"></a>
</td>
<td>
TPU address to connect to.
</td>
</tr><tr>
<td>
`trainable`<a id="trainable"></a>
</td>
<td>
boolean, whether pretrain layer is trainable.
</td>
</tr><tr>
<td>
`predict_batch_size`<a id="predict_batch_size"></a>
</td>
<td>
Batch size for prediction.
</td>
</tr><tr>
<td>
`do_lower_case`<a id="do_lower_case"></a>
</td>
<td>
boolean, whether to lower case the input text. Should be
True for uncased models and False for cased models.
</td>
</tr><tr>
<td>
`is_tf2`<a id="is_tf2"></a>
</td>
<td>
boolean, whether the hub module is in TensorFlow 2.x format.
</td>
</tr><tr>
<td>
`tflite_input_name`<a id="tflite_input_name"></a>
</td>
<td>
Dict, input names for the TFLite model.
</td>
</tr><tr>
<td>
`tflite_output_name`<a id="tflite_output_name"></a>
</td>
<td>
Dict, output names for the TFLite model.
</td>
</tr><tr>
<td>
`init_from_squad_model`<a id="init_from_squad_model"></a>
</td>
<td>
boolean, whether to initialize from the model that
is already retrained on Squad 1.1.
</td>
</tr><tr>
<td>
`default_batch_size`<a id="default_batch_size"></a>
</td>
<td>
Default batch size for training.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
Name of the object.
</td>
</tr>
</table>
