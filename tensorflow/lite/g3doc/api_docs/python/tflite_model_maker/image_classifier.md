page_type: reference
description: APIs to train an image classification model.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_model_maker.image_classifier" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tflite_model_maker.image_classifier

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/examples/blob/tflmm/v0.4.2/tensorflow_examples/lite/model_maker/public/image_classifier/__init__.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



APIs to train an image classification model.



#### Task guide:


<a href="https://www.tensorflow.org/lite/tutorials/model_maker_image_classification">https://www.tensorflow.org/lite/tutorials/model_maker_image_classification</a>

## Classes

[`class DataLoader`](../tflite_model_maker/image_classifier/DataLoader): DataLoader for image classifier.

[`class ImageClassifier`](../tflite_model_maker/image_classifier/ImageClassifier): ImageClassifier class for inference and exporting to tflite.

[`class ModelSpec`](../tflite_model_maker/image_classifier/ModelSpec): A specification of image model.

## Functions

[`EfficientNetLite0Spec(...)`](../tflite_model_maker/image_classifier/EfficientNetLite0Spec): Creates EfficientNet-Lite0 model spec. See also: <a href="../tflite_model_maker/image_classifier/ModelSpec"><code>tflite_model_maker.image_classifier.ModelSpec</code></a>.

[`EfficientNetLite1Spec(...)`](../tflite_model_maker/image_classifier/EfficientNetLite1Spec): Creates EfficientNet-Lite1 model spec. See also: <a href="../tflite_model_maker/image_classifier/ModelSpec"><code>tflite_model_maker.image_classifier.ModelSpec</code></a>.

[`EfficientNetLite2Spec(...)`](../tflite_model_maker/image_classifier/EfficientNetLite2Spec): Creates EfficientNet-Lite2 model spec. See also: <a href="../tflite_model_maker/image_classifier/ModelSpec"><code>tflite_model_maker.image_classifier.ModelSpec</code></a>.

[`EfficientNetLite3Spec(...)`](../tflite_model_maker/image_classifier/EfficientNetLite3Spec): Creates EfficientNet-Lite3 model spec. See also: <a href="../tflite_model_maker/image_classifier/ModelSpec"><code>tflite_model_maker.image_classifier.ModelSpec</code></a>.

[`EfficientNetLite4Spec(...)`](../tflite_model_maker/image_classifier/EfficientNetLite4Spec): Creates EfficientNet-Lite4 model spec. See also: <a href="../tflite_model_maker/image_classifier/ModelSpec"><code>tflite_model_maker.image_classifier.ModelSpec</code></a>.

[`MobileNetV2Spec(...)`](../tflite_model_maker/image_classifier/MobileNetV2Spec): Creates MobileNet v2 model spec. See also: <a href="../tflite_model_maker/image_classifier/ModelSpec"><code>tflite_model_maker.image_classifier.ModelSpec</code></a>.

[`Resnet50Spec(...)`](../tflite_model_maker/image_classifier/Resnet50Spec): Creates ResNet 50 model spec. See also: <a href="../tflite_model_maker/image_classifier/ModelSpec"><code>tflite_model_maker.image_classifier.ModelSpec</code></a>.

[`create(...)`](../tflite_model_maker/image_classifier/create): Loads data and retrains the model based on data for image classification.
