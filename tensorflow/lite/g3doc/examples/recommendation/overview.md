# Recommendation

Personalized recommendations are widely used for a variety of use cases on
mobile devices, such as media content retrieval, shopping product suggestion,
and next app recommendation. If you are interested in providing personalized
recommendations in your application while respecting user privacy, we recommend
exploring the following example and toolkit.

## Get started

<img src="images/screenshot.gif" class="attempt-right" style="max-width: 300px">

We provide a TensorFlow Lite sample application that demonstrates how to
recommend relevant items to users on Android.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/recommendation/android">Android
example</a>

If you are using a platform other than Android, or you are already familiar with
the TensorFlow Lite APIs, you can download our starter recommendation model.

<a class="button button-primary" href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/recommendation/20200720/recommendation.tar.gz">Download
starter model</a>

We also provide training script in Github to train your own model.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/recommendation/ml">Training
code</a>

## Understand the model architecture

We leverage a dual-encoder model architecture, with context-encoder to encode
sequential user history and label-encoder to encode predicted recommendation
candidate. Similarity between context and label encodings is used to represent
the likelihood that the predicted candidate meets the user's needs.

Three different sequential user history encoding techniques are provided with
this code base:

*   Bag-of-words encoder (BOW): averaging user activities' embeddings without
    considering context order.
*   Convolutional neural network encoder (CNN): applying multiple layers of
    convolutional neural networks to generate context encoding.
*   Recurrent neural network encoder (RNN): applying recurrent neural network to
    encode context sequence.

*Note: The model is trained based on
[MovieLens](https://grouplens.org/datasets/movielens/1m/) dataset for research
purpose.

## Examples

Input IDs:

*   Matrix (ID: 260)
*   Saving Private Ryan (ID: 2028)
*   (and more)

Output IDs:

*   Star Wars: Episode VI - Return of the Jedi (ID: 1210)
*   (and more)

## Performance benchmarks

Performance benchmark numbers are generated with the tool
[described here](https://www.tensorflow.org/lite/performance/benchmarks).

<table>
  <thead>
    <tr>
      <th>Model Name</th>
      <th>Model Size </th>
      <th>Device </th>
      <th>CPU</th>
    </tr>
  </thead>
  <tr>
    <td rowspan = 3>
      <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/recommendation/20200720/model.tar.gz">recommendation</a>
    </td>
    <td rowspan = 3>
      0.52 Mb
    </td>
    <td>Pixel 3</td>
    <td>0.09ms*</td>
  </tr>
   <tr>
     <td>Pixel 4 </td>
    <td>0.05ms*</td>
  </tr>
</table>

\* 4 threads used.

## Use your training data

In addition to the trained model, we provide an open-sourced
[toolkit in GitHub](https://github.com/tensorflow/examples/tree/master/lite/examples/recommendation/ml)
to train models with your own data. You can follow this tutorial to learn how to
use the toolkit and deploy trained models in your own mobile applications.

Please follow this
[tutorial](https://github.com/tensorflow/examples/tree/master/lite/examples/recommendation/ml/ondevice_recommendation.ipynb)
to apply the same technique used here to train a recommendation model using your
own datasets.

## Tips for model customization with your data

The pretrained model integrated in this demo application is trained with
[MovieLens](https://grouplens.org/datasets/movielens/1m/) dataset, you may want
to modify model configuration based on your own data, such as vocab size,
embedding dims and input context length. Here are a few tips:

*   Input context length: The best input context length varies with datasets. We
    suggest selecting input context length based on how much label events are
    correlated with long-term interests vs short-term context.

*   Encoder type selection: we suggest selecting encoder type based on input
    context length. Bag-of-words encoder works well for short input context
    length (e.g. <10), CNN and RNN encoders bring in more summarization ability
    for long input context length.
