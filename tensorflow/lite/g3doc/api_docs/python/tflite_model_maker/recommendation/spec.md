page_type: reference
description: APIs for recommendation specifications.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_model_maker.recommendation.spec" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="EncoderType"/>
<meta itemprop="property" content="FeatureType"/>
</div>

# Module: tflite_model_maker.recommendation.spec

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/examples/blob/tflmm/v0.4.2/tensorflow_examples/lite/model_maker/public/recommendation/spec/__init__.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



APIs for recommendation specifications.



#### Example:


```python
input_spec = recommendation.spec.InputSpec(
    activity_feature_groups=[
        # Group #1: defines how features are grouped in the first Group.
        dict(
            features=[
                # First feature.
                dict(
                    feature_name='context_movie_id',  # Feature name
                    feature_type='INT',  # Feature type
                    vocab_size=3953,     # ID size (number of IDs)
                    embedding_dim=8,     # Projected feature embedding dim
                    feature_length=10,   # History length of 10.
                ),
                # Maybe more features...
            ],
            encoder_type='CNN',  # CNN encoder (e.g. CNN, LSTM, BOW)
        ),
        # Maybe more groups...
    ],
    label_feature=dict(
        feature_name='label_movie_id',  # Label feature name
        feature_type='INT',  # Label type
        vocab_size=3953,   # Label size (number of classes)
        embedding_dim=8,   # label embedding demension
        feature_length=1,  # Exactly 1 label
    ),
)

model_hparams = recommendation.spec.ModelHParams(
    hidden_layer_dims=[32, 32],  # Hidden layers dimension.
    eval_top_k=[1, 5],           # Eval top 1 and top 5.
    conv_num_filter_ratios=[2, 4],  # For CNN encoder, conv filter mutipler.
    conv_kernel_size=16,            # For CNN encoder, base kernel size.
    lstm_num_units=16,              # For LSTM/RNN, num units.
    num_predictions=10,          # Number of output predictions. Select top 10.
)

spec = recommendation.ModelSpec(
    input_spec=input_spec, model_hparams=model_hparams)
# Or:
spec = model_spec.get(
    'recommendation', input_spec=input_spec, model_hparams=model_hparams)
```

## Classes

[`class Feature`](../../tflite_model_maker/recommendation/spec/Feature): A ProtocolMessage

[`class FeatureGroup`](../../tflite_model_maker/recommendation/spec/FeatureGroup): A ProtocolMessage

[`class InputSpec`](../../tflite_model_maker/recommendation/spec/InputSpec): A ProtocolMessage

[`class ModelHParams`](../../tflite_model_maker/recommendation/spec/ModelHParams): Class to hold parameters for model architecture configuration.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Other Members</h2></th></tr>

<tr>
<td>
EncoderType<a id="EncoderType"></a>
</td>
<td>
Instance of `google.protobuf.internal.enum_type_wrapper.EnumTypeWrapper`

EncoderType Enum (valid: BOW, CNN, LSTM).
</td>
</tr><tr>
<td>
FeatureType<a id="FeatureType"></a>
</td>
<td>
Instance of `google.protobuf.internal.enum_type_wrapper.EnumTypeWrapper`

FeatureType Enum (valid: STRING, INT, FLOAT).
</td>
</tr>
</table>
