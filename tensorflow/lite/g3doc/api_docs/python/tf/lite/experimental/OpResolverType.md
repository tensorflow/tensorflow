page_type: reference
description: Different types of op resolvers for Tensorflow Lite.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.lite.experimental.OpResolverType" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="AUTO"/>
<meta itemprop="property" content="BUILTIN"/>
<meta itemprop="property" content="BUILTIN_REF"/>
<meta itemprop="property" content="BUILTIN_WITHOUT_DEFAULT_DELEGATES"/>
</div>

# tf.lite.experimental.OpResolverType

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tensorflow/blob/v2.11.0/tensorflow/lite/python/interpreter.py#L309-L337">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Different types of op resolvers for Tensorflow Lite.

<!-- Placeholder for "Used in" -->

* `AUTO`: Indicates the op resolver that is chosen by default in TfLite
   Python, which is the "BUILTIN" as described below.
* `BUILTIN`: Indicates the op resolver for built-in ops with optimized kernel
  implementation.
* `BUILTIN_REF`: Indicates the op resolver for built-in ops with reference
  kernel implementation. It's generally used for testing and debugging.
* `BUILTIN_WITHOUT_DEFAULT_DELEGATES`: Indicates the op resolver for
  built-in ops with optimized kernel implementation, but it will disable
  the application of default TfLite delegates (like the XNNPACK delegate) to
  the model graph. Generally this should not be used unless there are issues
  with the default configuration.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>

<tr>
<td>
AUTO<a id="AUTO"></a>
</td>
<td>
`<OpResolverType.AUTO: 0>`
</td>
</tr><tr>
<td>
BUILTIN<a id="BUILTIN"></a>
</td>
<td>
`<OpResolverType.BUILTIN: 1>`
</td>
</tr><tr>
<td>
BUILTIN_REF<a id="BUILTIN_REF"></a>
</td>
<td>
`<OpResolverType.BUILTIN_REF: 2>`
</td>
</tr><tr>
<td>
BUILTIN_WITHOUT_DEFAULT_DELEGATES<a id="BUILTIN_WITHOUT_DEFAULT_DELEGATES"></a>
</td>
<td>
`<OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES: 3>`
</td>
</tr>
</table>
