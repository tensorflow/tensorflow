page_type: reference
description: APIs to train a model that can answer questions based on a predefined text.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_model_maker.question_answer" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tflite_model_maker.question_answer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/examples/blob/tflmm/v0.4.2/tensorflow_examples/lite/model_maker/public/question_answer/__init__.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



APIs to train a model that can answer questions based on a predefined text.



#### Task guide:


<a href="https://www.tensorflow.org/lite/tutorials/model_maker_question_answer">https://www.tensorflow.org/lite/tutorials/model_maker_question_answer</a>

## Classes

[`class BertQaSpec`](../tflite_model_maker/question_answer/BertQaSpec): A specification of BERT model for question answering.

[`class DataLoader`](../tflite_model_maker/question_answer/DataLoader): DataLoader for question answering.

[`class QuestionAnswer`](../tflite_model_maker/question_answer/QuestionAnswer): QuestionAnswer class for inference and exporting to tflite.

## Functions

[`MobileBertQaSpec(...)`](../tflite_model_maker/question_answer/MobileBertQaSpec): Creates MobileBert model spec for the question answer task. See also: <a href="../tflite_model_maker/question_answer/BertQaSpec"><code>tflite_model_maker.question_answer.BertQaSpec</code></a>.

[`MobileBertQaSquadSpec(...)`](../tflite_model_maker/question_answer/MobileBertQaSquadSpec): Creates MobileBert model spec that's already retrained on SQuAD1.1 for the question answer task. See also: <a href="../tflite_model_maker/question_answer/BertQaSpec"><code>tflite_model_maker.question_answer.BertQaSpec</code></a>.

[`create(...)`](../tflite_model_maker/question_answer/create): Loads data and train the model for question answer.
