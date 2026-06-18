page_type: reference
description: APIs to train a text classification model.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_model_maker.text_classifier" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tflite_model_maker.text_classifier

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/examples/blob/tflmm/v0.4.2/tensorflow_examples/lite/model_maker/public/text_classifier/__init__.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



APIs to train a text classification model.



#### Task guide:


<a href="https://www.tensorflow.org/lite/tutorials/model_maker_text_classification">https://www.tensorflow.org/lite/tutorials/model_maker_text_classification</a>

## Classes

[`class AverageWordVecSpec`](../tflite_model_maker/text_classifier/AverageWordVecSpec): A specification of averaging word vector model.

[`class BertClassifierSpec`](../tflite_model_maker/text_classifier/BertClassifierSpec): A specification of BERT model for text classification.

[`class DataLoader`](../tflite_model_maker/text_classifier/DataLoader): DataLoader for text classifier.

[`class TextClassifier`](../tflite_model_maker/text_classifier/TextClassifier): TextClassifier class for inference and exporting to tflite.

## Functions

[`MobileBertClassifierSpec(...)`](../tflite_model_maker/text_classifier/MobileBertClassifierSpec): Creates MobileBert model spec for the text classification task. See also: <a href="../tflite_model_maker/text_classifier/BertClassifierSpec"><code>tflite_model_maker.text_classifier.BertClassifierSpec</code></a>.

[`create(...)`](../tflite_model_maker/text_classifier/create): Loads data and train the model for test classification.
