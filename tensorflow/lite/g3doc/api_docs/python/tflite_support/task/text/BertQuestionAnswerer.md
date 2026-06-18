page_type: reference
description: Class that performs Bert question answering on text.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_support.task.text.BertQuestionAnswerer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="answer"/>
<meta itemprop="property" content="create_from_file"/>
<meta itemprop="property" content="create_from_options"/>
</div>

# tflite_support.task.text.BertQuestionAnswerer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/text/bert_question_answerer.py#L32-L103">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Class that performs Bert question answering on text.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_support.task.text.BertQuestionAnswerer(
    options: <a href="../../../tflite_support/task/text/BertQuestionAnswererOptions"><code>tflite_support.task.text.BertQuestionAnswererOptions</code></a>,
    cpp_bert_question_answerer: _CppBertQuestionAnswerer
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->




<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`options`<a id="options"></a>
</td>
<td>

</td>
</tr>
</table>



## Methods

<h3 id="answer"><code>answer</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/text/bert_question_answerer.py#L79-L99">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>answer(
    context: str, question: str
) -> <a href="../../../tflite_support/task/processor/QuestionAnswererResult"><code>tflite_support.task.processor.QuestionAnswererResult</code></a>
</code></pre>

Answers question based on the context.

Could be empty if no answer was
  found from the given context.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`context`
</td>
<td>
Context the question bases on.
</td>
</tr><tr>
<td>
`question`
</td>
<td>
Question to ask.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Question answerer result.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If any of the input arguments is invalid.
</td>
</tr><tr>
<td>
`RuntimeError`
</td>
<td>
If failed to calculate the embedding vector.
</td>
</tr>
</table>



<h3 id="create_from_file"><code>create_from_file</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/text/bert_question_answerer.py#L42-L58">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>create_from_file(
    file_path: str
) -> 'BertQuestionAnswerer'
</code></pre>

Creates the `BertQuestionAnswerer` object from a TensorFlow Lite model.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`file_path`
</td>
<td>
Path to the model.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
`BertQuestionAnswerer` object that's created from the model file.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If failed to create `BertQuestionAnswerer` object from the
provided file such as invalid file.
</td>
</tr><tr>
<td>
`RuntimeError`
</td>
<td>
If other types of error occurred.
</td>
</tr>
</table>



<h3 id="create_from_options"><code>create_from_options</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/text/bert_question_answerer.py#L60-L77">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>create_from_options(
    options: <a href="../../../tflite_support/task/text/BertQuestionAnswererOptions"><code>tflite_support.task.text.BertQuestionAnswererOptions</code></a>
) -> 'BertQuestionAnswerer'
</code></pre>

Creates the `BertQuestionAnswerer` object from the options.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`options`
</td>
<td>
Options for the Bert question answerer task.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
`BertQuestionAnswerer` object that's created from `options`.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If failed to create `BertQuestionAnswerer` object from
`BertQuestionAnswererOptions` such as missing the model.
</td>
</tr><tr>
<td>
`RuntimeError`
</td>
<td>
If other types of error occurred.
</td>
</tr>
</table>
