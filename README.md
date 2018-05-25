<div align="center">
  <img src="https://www.tensorflow.org/images/tf_logo_transp.png"><br><br>
</div>

-----------------
This forked tensorflow repository is used for [Parallax](https://github.com/snuspl/parallax). It changes the original tensorflow-v1.6 to keep all the gradients and its target tensors made by automatic differentiation in [TensorFlow collection](https://www.tensorflow.org/api_guides/python/framework#Graph_collections). We also add `average_option` in `SparseConditionalAccumulator` to support the diverse average patterns of language models(`average_option` : Integer indicating how to average out the accumulated values(1: divide by the accumulation count, 2: divide by each element counter, 3: no division).
The original tensorflow is [here](https://github.com/tensorflow/tensorflow/tree/r1.6).
