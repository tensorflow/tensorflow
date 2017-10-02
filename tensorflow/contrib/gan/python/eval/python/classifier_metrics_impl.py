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
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
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
    'frechet_inception_distance',
    'frechet_classifier_distance',
]


INCEPTION_URL = 'http://download.tensorflow.org/models/frozen_inception_v3_2017_09_13.tar.gz'
INCEPTION_FROZEN_GRAPH = 'frozen_inception_v3.pb'
INCEPTION_V3_INPUT = 'input'
INCEPTION_V3_OUTPUT = 'InceptionV3/Logits/SpatialSqueeze:0'
INCEPTION_V3_FINAL_POOL = 'InceptionV3/Logits/AvgPool_1a_8x8/AvgPool:0'
_INCEPTION_V3_NUM_CLASSES = 1001
_INCEPTION_V3_FINAL_POOL_SIZE = 2048
INCEPTION_V3_DEFAULT_IMG_SIZE = 299


def _validate_images(images, image_size):
  images = ops.convert_to_tensor(images)
  images.shape.with_rank(4)
  images.shape.assert_is_compatible_with(
      [None, image_size, image_size, None])
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


# Convenience preprocessing function, with fixed defaults.
# NOTE: Floating-point inputs are expected to be in [0, 1].
# Copied from /tensorflow_models/slim/preprocessing/inception_preprocessing.py.
def preprocess_image(
    image, height=INCEPTION_V3_DEFAULT_IMG_SIZE,
    width=INCEPTION_V3_DEFAULT_IMG_SIZE, central_fraction=0.875, scope=None):
  """Prepare one image for evaluation.

  If height and width are specified it would output an image with that size by
  applying resize_bilinear.

  If central_fraction is specified it would crop the central fraction of the
  input image.

  Args:
    image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
      [0, 1], otherwise it would converted to tf.float32 assuming that the range
      is [0, MAX], where MAX is largest positive representable number for
      int(8/16/32) data type (see `tf.image.convert_image_dtype` for details).
    height: integer
    width: integer
    central_fraction: Optional Float, fraction of the image to crop.
    scope: Optional scope for name_scope.
  Returns:
    3-D float Tensor of prepared image.
  """
  with ops.name_scope(scope, 'eval_image', [image, height, width]):
    if image.dtype != dtypes.float32:
      image = image_ops.convert_image_dtype(image, dtype=dtypes.float32)
    # Crop the central region of the image with an area containing 87.5% of
    # the original image.
    image = image_ops.central_crop(image, central_fraction=central_fraction)

    # Resize the image to the specified height and width.
    image = array_ops.expand_dims(image, 0)
    image = image_ops.resize_bilinear(image, [height, width],
                                      align_corners=False)
    image = array_ops.squeeze(image, [0])
    image = (image - 0.5) * 2.0
    return image


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


def get_graph_def_from_url_tarball(url, filename):
  """Get a GraphDef proto from a tarball on the web."""
  def _progress(count, block_size, total_size):
    sys.stdout.write('\r>> Downloading %s %.1f%%' % (
        url, float(count * block_size) / float(total_size) * 100.0))
    sys.stdout.flush()
  tar_filename, _ = urllib.request.urlretrieve(url, reporthook=_progress)
  with tarfile.open(tar_filename, 'r:gz') as tar:
    proto_str = tar.extractfile(filename).read()
  return graph_pb2.GraphDef.FromString(proto_str)


def _default_graph_def_fn():
  return get_graph_def_from_url_tarball(INCEPTION_URL, INCEPTION_FROZEN_GRAPH)


def run_inception(images,
                  graph_def=None,
                  default_graph_def_fn=_default_graph_def_fn,
                  image_size=INCEPTION_V3_DEFAULT_IMG_SIZE,
                  input_tensor=INCEPTION_V3_INPUT,
                  output_tensor=INCEPTION_V3_OUTPUT):
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
    output_tensor: Name of output Tensor. This function will compute activations
      at the specified layer. Examples include INCEPTION_V3_OUTPUT and
      INCEPTION_V3_FINAL_POOL which would result in this function computing
      the final logits or the penultimate pooling layer.

  Returns:
    Logits.

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
  if array_ops.rank(activations) != 2:
    activations = layers.flatten(activations)
  return activations


def run_image_classifier(tensor, graph_def, input_tensor,
                         output_tensor, scope='RunClassifier'):
  """Runs a network from a frozen graph.

  Args:
    tensor: An Input tensor.
    graph_def: A GraphDef proto.
    input_tensor: Name of input tensor in graph def.
    output_tensor: Name of output tensor in graph def.
    scope: Name scope for classifier.

  Returns:
    Classifier output. Shape depends on the classifier used, but is often
    [batch, classes].

  Raises:
    ValueError: If `image_size` is not `None`, and `tensor` are not the correct
      size.
  """
  input_map = {input_tensor: tensor}
  return_elements = [output_tensor]
  classifier_output = importer.import_graph_def(
      graph_def, input_map, return_elements, name=scope)[0]

  return classifier_output


def classifier_score(images, classifier_fn, num_batches=1):
  """Classifier score for evaluating a conditional generative model.

  This is based on the Inception Score, but for an arbitrary classifier.

  This technique is described in detail in https://arxiv.org/abs/1606.03498. In
  summary, this function calculates

  exp( E[ KL(p(y|x) || p(y)) ] )

  which captures how different the network's classification prediction is from
  the prior distribution over classes.

  Args:
    images: Images to calculate the classifier score for.
    classifier_fn: A function that takes images and produces logits based on a
      classifier.
    num_batches: Number of batches to split `generated_images` in to in order to
      efficiently run them through the classifier network.

  Returns:
    The classifier score. A floating-point scalar.
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
  logits.shape.assert_has_rank(2)
  p = nn_ops.softmax(logits)
  q = math_ops.reduce_mean(p, axis=0)
  kl = _kl_divergence(p, logits, q)
  kl.shape.assert_has_rank(1)
  log_score = math_ops.reduce_mean(kl)

  return math_ops.exp(log_score)


inception_score = functools.partial(
    classifier_score,
    classifier_fn=functools.partial(
        run_inception, output_tensor=INCEPTION_V3_OUTPUT))


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
  sqrt_a_sigmav_a = math_ops.matmul(
      sqrt_sigma, math_ops.matmul(sigma_v, sqrt_sigma))

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
  C and C_w, this function calcuates

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
    real_images: Real images to use to compute Frechet Inception distance.
    generated_images: Generated images to use to compute Frechet Inception
      distance.
    classifier_fn: A function that takes images and produces activations
      based on a classifier.
    num_batches: Number of batches to split images in to in order to
      efficiently run them through the classifier network.

  Returns:
    The Frechet Inception distance. A floating-point scalar.
  """

  real_images_list = array_ops.split(
      real_images, num_or_size_splits=num_batches)
  generated_images_list = array_ops.split(
      generated_images, num_or_size_splits=num_batches)

  imgs = array_ops.stack(real_images_list + generated_images_list)

  # Compute the activations using the memory-efficient `map_fn`.
  activations = functional_ops.map_fn(
      fn=classifier_fn,
      elems=imgs,
      parallel_iterations=1,
      back_prop=False,
      swap_memory=True,
      name='RunClassifier')

  # Split the activations by the real and generated images.
  real_a, gen_a = array_ops.split(activations, [num_batches, num_batches], 0)

  # Ensure the activations have the right shapes.
  real_a = array_ops.concat(array_ops.unstack(real_a), 0)
  gen_a = array_ops.concat(array_ops.unstack(gen_a), 0)
  real_a.shape.assert_has_rank(2)
  gen_a.shape.assert_has_rank(2)

  # Compute mean and covariance matrices of activations.
  m = math_ops.reduce_mean(real_a, 0)
  m_v = math_ops.reduce_mean(gen_a, 0)
  num_examples = math_ops.to_float(array_ops.shape(real_a)[0])

  # sigma = (1 / (n - 1)) * (X - mu) (X - mu)^T
  sigma = math_ops.matmul(
      real_a - m, real_a - m, transpose_a=True) / (num_examples - 1)

  sigma_v = math_ops.matmul(
      gen_a - m_v, gen_a - m_v, transpose_a=True) / (num_examples - 1)

  # Find the Tr(sqrt(sigma sigma_v)) component of FID
  sqrt_trace_component = trace_sqrt_product(sigma, sigma_v)

  # Compute the two components of FID.

  # First the covariance component.
  # Here, note that trace(A + B) = trace(A) + trace(B)
  trace = math_ops.trace(sigma + sigma_v) - 2.0 * sqrt_trace_component

  # Next the distance between means.
  mean = math_ops.square(linalg_ops.norm(m - m_v))  # This uses the L2 norm.
  fid = trace + mean

  return fid


frechet_inception_distance = functools.partial(
    frechet_classifier_distance,
    classifier_fn=functools.partial(
        run_inception, output_tensor=INCEPTION_V3_FINAL_POOL))
