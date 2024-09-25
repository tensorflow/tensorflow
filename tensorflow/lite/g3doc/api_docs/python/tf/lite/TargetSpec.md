page_type: reference
description: Specification of target device used to optimize the model.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.lite.TargetSpec" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
</div>

# tf.lite.TargetSpec

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tensorflow/blob/v2.11.0/tensorflow/lite/python/lite.py#L182-L227">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Specification of target device used to optimize the model.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.lite.TargetSpec(
    supported_ops=None,
    supported_types=None,
    experimental_select_user_tf_ops=None,
    experimental_supported_backends=None
)
</code></pre>




<h3>Used in the notebooks</h3>
<table class="vertical-rules">
  <thead>
    <tr>
      <th>Used in the guide</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
  <ul>
    <li><a href="https://www.tensorflow.org/lite/guide/authoring">TFLite Authoring Tool</a></li>
  </ul>
</td>
    </tr>
  </tbody>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`supported_ops`<a id="supported_ops"></a>
</td>
<td>
Experimental flag, subject to change. Set of <a href="../../tf/lite/OpsSet"><code>tf.lite.OpsSet</code></a>
options, where each option represents a set of operators supported by the
target device. (default {tf.lite.OpsSet.TFLITE_BUILTINS}))
</td>
</tr><tr>
<td>
`supported_types`<a id="supported_types"></a>
</td>
<td>
Set of <a href="https://www.tensorflow.org/api_docs/python/tf/dtypes/DType"><code>tf.dtypes.DType</code></a> data types supported on the target
device. If initialized, optimization might be driven by the smallest type
in this set. (default set())
</td>
</tr><tr>
<td>
`experimental_select_user_tf_ops`<a id="experimental_select_user_tf_ops"></a>
</td>
<td>
Experimental flag, subject to change. Set
of user's TensorFlow operators' names that are required in the TensorFlow
Lite runtime. These ops will be exported as select TensorFlow ops in the
model (in conjunction with the tf.lite.OpsSet.SELECT_TF_OPS flag). This is
an advanced feature that should only be used if the client is using TF ops
that may not be linked in by default with the TF ops that are provided
when using the SELECT_TF_OPS path. The client is responsible for linking
these ops into the target runtime.
</td>
</tr><tr>
<td>
`experimental_supported_backends`<a id="experimental_supported_backends"></a>
</td>
<td>
Experimental flag, subject to change.
Set containing names of supported backends. Currently only "GPU" is
supported, more options will be available later.
</td>
</tr>
</table>
