# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Model evaluation tools for TFGAN.

These methods come from https://arxiv.org/abs/1606.03498 and
https://arxiv.org/abs/1706.08500.

NOTE: This implementation uses the same weights as in
https://github.com/openai/improved-gan/blob/master/inception_score/model.py,
but is more numerically stable and is an unbiased estimator of the true
Inception score even when splitting the inputs into batches.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import sys
import tarfile

from six.moves import urllib

from tensorflow.contrib.layers.python.layers import layers
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import gfile
from tensorflow.python.platform import resource_loader

__all__ = [
    'get_graph_def_from_disk',
    'get_graph_def_from_resource',
    'get_graph_def_from_url_tarball',
    'preprocess_image',
    'run_image_classifier',
    'run_inception',
    'inception_score',
    'classifier_score',
    'classifier_score_from_logits',
    'frechet_inception_distance',
    'frechet_classifier_distance',
    'frechet_classifier_distance_from_activations',
    'mean_only_frechet_classifier_distance_from_activations',
    'diagonal_only_frechet_classifier_distance_from_activations',
    'INCEPTION_DEFAULT_IMAGE_SIZE',
]

INCEPTION_URL = 'http://download.tensorflow.org/models/frozen_inception_v1_2015_12_05.tar.gz'
INCEPTION_FROZEN_GRAPH = 'inceptionv1_for_inception_score.pb'
INCEPTION_INPUT = 'Mul:0'
INCEPTION_OUTPUT = 'logits:0'
INCEPTION_FINAL_POOL = 'pool_3:0'
INCEPTION_DEFAULT_IMAGE_SIZE = 299


def _validate_images(images, image_size):
  images = ops.convert_to_tensor(images)
  images.shape.with_rank(4)
  images.shape.assert_is_compatible_with([None, image_size, image_size, None])
  return images


def _symmetric_matrix_square_root(mat, eps=1e-10):
  """Compute square root of a symmetric matrix.

  Note that this is different from an elementwise square root. We want to
  compute M' where M' = sqrt(mat) such that M' * M' = mat.

  Also note that this method **only** works for symmetric matrices.

  Args:
    mat: Matrix to take the square root of.
    eps: Small epsilon such that any element less than eps will not be square
      rooted to guard against numerical instability.

  Returns:
    Matrix square root of mat.
  """
  # Unlike numpy, tensorflow's return order is (s, u, v)
  s, u, v = linalg_ops.svd(mat)
  # sqrt is unstable around 0, just use 0 in such case
  si = array_ops.where(math_ops.less(s, eps), s, math_ops.sqrt(s))
  # Note that the v returned by Tensorflow is v = V
  # (when referencing the equation A = U S V^T)
  # This is unlike Numpy which returns v = V^T
  return math_ops.matmul(
      math_ops.matmul(u, array_ops.diag(si)), v, transpose_b=True)


def preprocess_image(images,
                     height=INCEPTION_DEFAULT_IMAGE_SIZE,
                     width=INCEPTION_DEFAULT_IMAGE_SIZE,
                     scope=None):
  """Prepare a batch of images for evaluation.

  This is the preprocessing portion of the graph from
  http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz.

  Note that it expects Tensors in [0, 255]. This function maps pixel values to
  [-1, 1] and resizes to match the InceptionV1 network.

  Args:
    images: 3-D or 4-D Tensor of images. Values are in [0, 255].
    height: Integer. Height of resized output image.
    width: Integer. Width of resized output image.
    scope: Optional scope for name_scope.

  Returns:
    3-D or 4-D float Tensor of prepared image(s). Values are in [-1, 1].
  """
  is_single = images.shape.ndims == 3
  with ops.name_scope(scope, 'preprocess', [images, height, width]):
    if not images.dtype.is_floating:
      images = math_ops.to_float(images)
    if is_single:
      images = array_ops.expand_dims(images, axis=0)
    resized = image_ops.resize_bilinear(images, [height, width])
    resized = (resized - 128.0) / 128.0
    if is_single:
      resized = array_ops.squeeze(resized, axis=0)
    return resized


def _kl_divergence(p, p_logits, q):
  """Computes the Kullback-Liebler divergence between p and q.

  This function uses p's logits in some places to improve numerical stability.

  Specifically:

  KL(p || q) = sum[ p * log(p / q) ]
    = sum[ p * ( log(p)                - log(q) ) ]
    = sum[ p * ( log_softmax(p_logits) - log(q) ) ]

  Args:
    p: A 2-D floating-point Tensor p_ij, where `i` corresponds to the minibatch
      example and `j` corresponds to the probability of being in class `j`.
    p_logits: A 2-D floating-point Tensor corresponding to logits for `p`.
    q: A 1-D floating-point Tensor, where q_j corresponds to the probability
      of class `j`.

  Returns:
    KL divergence between two distributions. Output dimension is 1D, one entry
    per distribution in `p`.

  Raises:
    ValueError: If any of the inputs aren't floating-point.
    ValueError: If p or p_logits aren't 2D.
    ValueError: If q isn't 1D.
  """
  for tensor in [p, p_logits, q]:
    if not tensor.dtype.is_floating:
      raise ValueError('Input %s must be floating type.', tensor.name)
  p.shape.assert_has_rank(2)
  p_logits.shape.assert_has_rank(2)
  q.shape.assert_has_rank(1)
  return math_ops.reduce_sum(
      p * (nn_ops.log_softmax(p_logits) - math_ops.log(q)), axis=1)


def get_graph_def_from_disk(filename):
  """Get a GraphDef proto from a disk location."""
  with gfile.FastGFile(filename, 'rb') as f:
    return graph_pb2.GraphDef.FromString(f.read())


def get_graph_def_from_resource(filename):
  """Get a GraphDef proto from within a .par file."""
  return graph_pb2.GraphDef.FromString(resource_loader.load_resource(filename))


def get_graph_def_from_url_tarball(url, filename, tar_filename=None):
  """Get a GraphDef proto from a tarball on the web.

  Args:
    url: Web address of tarball
    filename: Filename of graph definition within tarball
    tar_filename: Temporary download filename (None = always download)

  Returns:
    A GraphDef loaded from a file in the downloaded tarball.
  """
  if not (tar_filename and os.path.exists(tar_filename)):

    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' %
                       (url,
                        float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()

    tar_filename, _ = urllib.request.urlretrieve(url, tar_filename, _progress)
  with tarfile.open(tar_filename, 'r:gz') as tar:
    proto_str = tar.extractfile(filename).read()
  return graph_pb2.GraphDef.FromString(proto_str)


def _default_graph_def_fn():
  return get_graph_def_from_url_tarball(INCEPTION_URL, INCEPTION_FROZEN_GRAPH,
                                        os.path.basename(INCEPTION_URL))


def run_inception(images,
                  graph_def=None,
                  default_graph_def_fn=_default_graph_def_fn,
                  image_size=INCEPTION_DEFAULT_IMAGE_SIZE,
                  input_tensor=INCEPTION_INPUT,
                  output_tensor=INCEPTION_OUTPUT):
  """Run images through a pretrained Inception classifier.

  Args:
    images: Input tensors. Must be [batch, height, width, channels]. Input shape
      and values must be in [-1, 1], which can be achieved using
      `preprocess_image`.
    graph_def: A GraphDef proto of a pretrained Inception graph. If `None`,
      call `default_graph_def_fn` to get GraphDef.
    default_graph_def_fn: A function that returns a GraphDef. Used if
      `graph_def` is `None. By default, returns a pretrained InceptionV3 graph.
    image_size: Required image width and height. See unit tests for the default
      values.
    input_tensor: Name of input Tensor.
    output_tensor: Name or list of output Tensors. This function will compute
      activations at the specified layer. Examples include INCEPTION_V3_OUTPUT
      and INCEPTION_V3_FINAL_POOL which would result in this function computing
      the final logits or the penultimate pooling layer.

  Returns:
    Tensor or Tensors corresponding to computed `output_tensor`.

  Raises:
    ValueError: If images are not the correct size.
    ValueError: If neither `graph_def` nor `default_graph_def_fn` are provided.
  """
  images = _validate_images(images, image_size)

  if graph_def is None:
    if default_graph_def_fn is None:
      raise ValueError('If `graph_def` is `None`, must provide '
                       '`default_graph_def_fn`.')
    graph_def = default_graph_def_fn()

  activations = run_image_classifier(images, graph_def, input_tensor,
                                     output_tensor)
  if isinstance(activations, list):
    for i, activation in enumerate(activations):
      if array_ops.rank(activation) != 2:
        activations[i] = layers.flatten(activation)
  else:
    if array_ops.rank(activations) != 2:
      activations = layers.flatten(activations)

  return activations


def run_image_classifier(tensor,
                         graph_def,
                         input_tensor,
                         output_tensor,
                         scope='RunClassifier'):
  """Runs a network from a frozen graph.

  Args:
    tensor: An Input tensor.
    graph_def: A GraphDef proto.
    input_tensor: Name of input tensor in graph def.
    output_tensor: A tensor name or list of tensor names in graph def.
    scope: Name scope for classifier.

  Returns:
    Classifier output if `output_tensor` is a string, or a list of outputs if
    `output_tensor` is a list.

  Raises:
    ValueError: If `input_tensor` or `output_tensor` aren't in the graph_def.
  """
  input_map = {input_tensor: tensor}
  is_singleton = isinstance(output_tensor, str)
  if is_singleton:
    output_tensor = [output_tensor]
  classifier_outputs = importer.import_graph_def(
      graph_def, input_map, output_tensor, name=scope)
  if is_singleton:
    classifier_outputs = classifier_outputs[0]

  return classifier_outputs


def classifier_score(images, classifier_fn, num_batches=1):
  """Classifier score for evaluating a conditional generative model.

  This is based on the Inception Score, but for an arbitrary classifier.

  This technique is described in detail in https://arxiv.org/abs/1606.03498. In
  summary, this function calculates

  exp( E[ KL(p(y|x) || p(y)) ] )

  which captures how different the network's classification prediction is from
  the prior distribution over classes.

  NOTE: This function consumes images, computes their logits, and then
  computes the classifier score. If you would like to precompute many logits for
  large batches, use classifier_score_from_logits(), which this method also
  uses.

  Args:
    images: Images to calculate the classifier score for.
    classifier_fn: A function that takes images and produces logits based on a
      classifier.
    num_batches: Number of batches to split `generated_images` in to in order to
      efficiently run them through the classifier network.

  Returns:
    The classifier score. A floating-point scalar of the same type as the output
    of `classifier_fn`.
  """
  generated_images_list = array_ops.split(
      images, num_or_size_splits=num_batches)

  # Compute the classifier splits using the memory-efficient `map_fn`.
  logits = functional_ops.map_fn(
      fn=classifier_fn,
      elems=array_ops.stack(generated_images_list),
      parallel_iterations=1,
      back_prop=False,
      swap_memory=True,
      name='RunClassifier')
  logits = array_ops.concat(array_ops.unstack(logits), 0)

  return classifier_score_from_logits(logits)


def classifier_score_from_logits(logits):
  """Classifier score for evaluating a generative model from logits.

  This method computes the classifier score for a set of logits. This can be
  used independently of the classifier_score() method, especially in the case
  of using large batches during evaluation where we would like precompute all
  of the logits before computing the classifier score.

  This technique is described in detail in https://arxiv.org/abs/1606.03498. In
  summary, this function calculates:

  exp( E[ KL(p(y|x) || p(y)) ] )

  which captures how different the network's classification prediction is from
  the prior distribution over classes.

  Args:
    logits: Precomputed 2D tensor of logits that will be used to
      compute the classifier score.

  Returns:
    The classifier score. A floating-point scalar of the same type as the output
    of `logits`.
  """
  logits.shape.assert_has_rank(2)

  # Use maximum precision for best results.
  logits_dtype = logits.dtype
  if logits_dtype != dtypes.float64:
    logits = math_ops.to_double(logits)

  p = nn_ops.softmax(logits)
  q = math_ops.reduce_mean(p, axis=0)
  kl = _kl_divergence(p, logits, q)
  kl.shape.assert_has_rank(1)
  log_score = math_ops.reduce_mean(kl)
  final_score = math_ops.exp(log_score)

  if logits_dtype != dtypes.float64:
    final_score = math_ops.cast(final_score, logits_dtype)

  return final_score


inception_score = functools.partial(
    classifier_score,
    classifier_fn=functools.partial(
        run_inception, output_tensor=INCEPTION_OUTPUT))


def trace_sqrt_product(sigma, sigma_v):
  """Find the trace of the positive sqrt of product of covariance matrices.

  '_symmetric_matrix_square_root' only works for symmetric matrices, so we
  cannot just take _symmetric_matrix_square_root(sigma * sigma_v).
  ('sigma' and 'sigma_v' are symmetric, but their product is not necessarily).

  Let sigma = A A so A = sqrt(sigma), and sigma_v = B B.
  We want to find trace(sqrt(sigma sigma_v)) = trace(sqrt(A A B B))
  Note the following properties:
  (i) forall M1, M2: eigenvalues(M1 M2) = eigenvalues(M2 M1)
     => eigenvalues(A A B B) = eigenvalues (A B B A)
  (ii) if M1 = sqrt(M2), then eigenvalues(M1) = sqrt(eigenvalues(M2))
     => eigenvalues(sqrt(sigma sigma_v)) = sqrt(eigenvalues(A B B A))
  (iii) forall M: trace(M) = sum(eigenvalues(M))
     => trace(sqrt(sigma sigma_v)) = sum(eigenvalues(sqrt(sigma sigma_v)))
                                   = sum(sqrt(eigenvalues(A B B A)))
                                   = sum(eigenvalues(sqrt(A B B A)))
                                   = trace(sqrt(A B B A))
                                   = trace(sqrt(A sigma_v A))
  A = sqrt(sigma). Both sigma and A sigma_v A are symmetric, so we **can**
  use the _symmetric_matrix_square_root function to find the roots of these
  matrices.

  Args:
    sigma: a square, symmetric, real, positive semi-definite covariance matrix
    sigma_v: same as sigma

  Returns:
    The trace of the positive square root of sigma*sigma_v
  """

  # Note sqrt_sigma is called "A" in the proof above
  sqrt_sigma = _symmetric_matrix_square_root(sigma)

  # This is sqrt(A sigma_v A) above
  sqrt_a_sigmav_a = math_ops.matmul(sqrt_sigma,
                                    math_ops.matmul(sigma_v, sqrt_sigma))

  return math_ops.trace(_symmetric_matrix_square_root(sqrt_a_sigmav_a))


def frechet_classifier_distance(real_images,
                                generated_images,
                                classifier_fn,
                                num_batches=1):
  """Classifier distance for evaluating a generative model.

  This is based on the Frechet Inception distance, but for an arbitrary
  classifier.

  This technique is described in detail in https://arxiv.org/abs/1706.08500.
  Given two Gaussian distribution with means m and m_w and covariance matrices
  C and C_w, this function calculates

              |m - m_w|^2 + Tr(C + C_w - 2(C * C_w)^(1/2))

  which captures how different the distributions of real images and generated
  images (or more accurately, their visual features) are. Note that unlike the
  Inception score, this is a true distance and utilizes information about real
  world images.

  Note that when computed using sample means and sample covariance matrices,
  Frechet distance is biased. It is more biased for small sample sizes. (e.g.
  even if the two distributions are the same, for a small sample size, the
  expected Frechet distance is large). It is important to use the same
  sample size to compute Frechet classifier distance when comparing two
  generative models.

  NOTE: This function consumes images, computes their activations, and then
  computes the classifier score. If you would like to precompute many
  activations for real and generated images for large batches, please use
  frechet_clasifier_distance_from_activations(), which this method also uses.

  Args:
    real_images: Real images to use to compute Frechet Inception distance.
    generated_images: Generated images to use to compute Frechet Inception
      distance.
    classifier_fn: A function that takes images and produces activations
      based on a classifier.
    num_batches: Number of batches to split images in to in order to
      efficiently run them through the classifier network.

  Returns:
    The Frechet Inception distance. A floating-point scalar of the same type
    as the output of `classifier_fn`.
  """
  real_images_list = array_ops.split(
      real_images, num_or_size_splits=num_batches)
  generated_images_list = array_ops.split(
      generated_images, num_or_size_splits=num_batches)

  real_imgs = array_ops.stack(real_images_list)
  generated_imgs = array_ops.stack(generated_images_list)

  # Compute the activations using the memory-efficient `map_fn`.
  def compute_activations(elems):
    return functional_ops.map_fn(fn=classifier_fn,
                                 elems=elems,
                                 parallel_iterations=1,
                                 back_prop=False,
                                 swap_memory=True,
                                 name='RunClassifier')

  real_a = compute_activations(real_imgs)
  gen_a = compute_activations(generated_imgs)

  # Ensure the activations have the right shapes.
  real_a = array_ops.concat(array_ops.unstack(real_a), 0)
  gen_a = array_ops.concat(array_ops.unstack(gen_a), 0)

  return frechet_classifier_distance_from_activations(real_a, gen_a)


def mean_only_frechet_classifier_distance_from_activations(
    real_activations, generated_activations):
  """Classifier distance for evaluating a generative model from activations.

  Given two Gaussian distribution with means m and m_w and covariance matrices
  C and C_w, this function calcuates

                                |m - m_w|^2

  which captures how different the distributions of real images and generated
  images (or more accurately, their visual features) are. Note that unlike the
  Inception score, this is a true distance and utilizes information about real
  world images.

  Note that when computed using sample means and sample covariance matrices,
  Frechet distance is biased. It is more biased for small sample sizes. (e.g.
  even if the two distributions are the same, for a small sample size, the
  expected Frechet distance is large). It is important to use the same
  sample size to compute frechet classifier distance when comparing two
  generative models.

  In this variant, we only compute the difference between the means of the
  fitted Gaussians. The computation leads to O(n) vs. O(n^2) memory usage, yet
  still retains much of the same information as FID.

  Args:
    real_activations: 2D array of activations of real images of size
      [num_images, num_dims] to use to compute Frechet Inception distance.
    generated_activations: 2D array of activations of generated images of size
      [num_images, num_dims] to use to compute Frechet Inception distance.

  Returns:
    The mean-only Frechet Inception distance. A floating-point scalar of the
    same type as the output of the activations.
  """
  real_activations.shape.assert_has_rank(2)
  generated_activations.shape.assert_has_rank(2)

  activations_dtype = real_activations.dtype
  if activations_dtype != dtypes.float64:
    real_activations = math_ops.to_double(real_activations)
    generated_activations = math_ops.to_double(generated_activations)

  # Compute means of activations.
  m = math_ops.reduce_mean(real_activations, 0)
  m_w = math_ops.reduce_mean(generated_activations, 0)

  # Next the distance between means.
  mean = math_ops.reduce_sum(
      math_ops.squared_difference(m, m_w))  # Equivalent to L2 but more stable.
  mofid = mean
  if activations_dtype != dtypes.float64:
    mofid = math_ops.cast(mofid, activations_dtype)

  return mofid


def diagonal_only_frechet_classifier_distance_from_activations(
    real_activations, generated_activations):
  """Classifier distance for evaluating a generative model.

  This is based on the Frechet Inception distance, but for an arbitrary
  classifier.

  This technique is described in detail in https://arxiv.org/abs/1706.08500.
  Given two Gaussian distribution with means m and m_w and covariance matrices
  C and C_w, this function calcuates

          |m - m_w|^2 + (sigma + sigma_w - 2(sigma x sigma_w)^(1/2))

  which captures how different the distributions of real images and generated
  images (or more accurately, their visual features) are. Note that unlike the
  Inception score, this is a true distance and utilizes information about real
  world images. In this variant, we compute diagonal-only covariance matrices.
  As a result, instead of computing an expensive matrix square root, we can do
  something much simpler, and has O(n) vs O(n^2) space complexity.

  Note that when computed using sample means and sample covariance matrices,
  Frechet distance is biased. It is more biased for small sample sizes. (e.g.
  even if the two distributions are the same, for a small sample size, the
  expected Frechet distance is large). It is important to use the same
  sample size to compute frechet classifier distance when comparing two
  generative models.

  Args:
    real_activations: Real images to use to compute Frechet Inception distance.
    generated_activations: Generated images to use to compute Frechet Inception
      distance.

  Returns:
    The diagonal-only Frechet Inception distance. A floating-point scalar of
    the same type as the output of the activations.

  Raises:
    ValueError: If the shape of the variance and mean vectors are not equal.
  """
  real_activations.shape.assert_has_rank(2)
  generated_activations.shape.assert_has_rank(2)

  activations_dtype = real_activations.dtype
  if activations_dtype != dtypes.float64:
    real_activations = math_ops.to_double(real_activations)
    generated_activations = math_ops.to_double(generated_activations)

  # Compute mean and covariance matrices of activations.
  m, var = nn_impl.moments(real_activations, axes=[0])
  m_w, var_w = nn_impl.moments(generated_activations, axes=[0])

  actual_shape = var.get_shape()
  expected_shape = m.get_shape()

  if actual_shape != expected_shape:
    raise ValueError('shape: {} must match expected shape: {}'.format(
        actual_shape, expected_shape))

  # Compute the two components of FID.

  # First the covariance component.
  # Here, note that trace(A + B) = trace(A) + trace(B)
  trace = math_ops.reduce_sum(
      (var + var_w) - 2.0 * math_ops.sqrt(math_ops.multiply(var, var_w)))

  # Next the distance between means.
  mean = math_ops.reduce_sum(
      math_ops.squared_difference(m, m_w))  # Equivalent to L2 but more stable.
  dofid = trace + mean
  if activations_dtype != dtypes.float64:
    dofid = math_ops.cast(dofid, activations_dtype)

  return dofid


def frechet_classifier_distance_from_activations(real_activations,
                                                 generated_activations):
  """Classifier distance for evaluating a generative model.

  This methods computes the Frechet classifier distance from activations of
  real images and generated images. This can be used independently of the
  frechet_classifier_distance() method, especially in the case of using large
  batches during evaluation where we would like precompute all of the
  activations before computing the classifier distance.

  This technique is described in detail in https://arxiv.org/abs/1706.08500.
  Given two Gaussian distribution with means m and m_w and covariance matrices
  C and C_w, this function calculates

                |m - m_w|^2 + Tr(C + C_w - 2(C * C_w)^(1/2))

  which captures how different the distributions of real images and generated
  images (or more accurately, their visual features) are. Note that unlike the
  Inception score, this is a true distance and utilizes information about real
  world images.

  Note that when computed using sample means and sample covariance matrices,
  Frechet distance is biased. It is more biased for small sample sizes. (e.g.
  even if the two distributions are the same, for a small sample size, the
  expected Frechet distance is large). It is important to use the same
  sample size to compute frechet classifier distance when comparing two
  generative models.

  Args:
    real_activations: 2D Tensor containing activations of real data. Shape is
      [batch_size, activation_size].
    generated_activations: 2D Tensor containing activations of generated data.
      Shape is [batch_size, activation_size].

  Returns:
   The Frechet Inception distance. A floating-point scalar of the same type
   as the output of the activations.

  """
  real_activations.shape.assert_has_rank(2)
  generated_activations.shape.assert_has_rank(2)

  activations_dtype = real_activations.dtype
  if activations_dtype != dtypes.float64:
    real_activations = math_ops.to_double(real_activations)
    generated_activations = math_ops.to_double(generated_activations)

  # Compute mean and covariance matrices of activations.
  m = math_ops.reduce_mean(real_activations, 0)
  m_w = math_ops.reduce_mean(generated_activations, 0)
  num_examples_real = math_ops.to_double(array_ops.shape(real_activations)[0])
  num_examples_generated = math_ops.to_double(
      array_ops.shape(generated_activations)[0])

  # sigma = (1 / (n - 1)) * (X - mu) (X - mu)^T
  real_centered = real_activations - m
  sigma = math_ops.matmul(
      real_centered, real_centered, transpose_a=True) / (
          num_examples_real - 1)

  gen_centered = generated_activations - m_w
  sigma_w = math_ops.matmul(
      gen_centered, gen_centered, transpose_a=True) / (
          num_examples_generated - 1)

  # Find the Tr(sqrt(sigma sigma_w)) component of FID
  sqrt_trace_component = trace_sqrt_product(sigma, sigma_w)

  # Compute the two components of FID.

  # First the covariance component.
  # Here, note that trace(A + B) = trace(A) + trace(B)
  trace = math_ops.trace(sigma + sigma_w) - 2.0 * sqrt_trace_component

  # Next the distance between means.
  mean = math_ops.reduce_sum(
      math_ops.squared_difference(m, m_w))  # Equivalent to L2 but more stable.
  fid = trace + mean
  if activations_dtype != dtypes.float64:
    fid = math_ops.cast(fid, activations_dtype)

  return fid

frechet_inception_distance = functools.partial(
    frechet_classifier_distance,
    classifier_fn=functools.partial(
        run_inception, output_tensor=INCEPTION_FINAL_POOL))
