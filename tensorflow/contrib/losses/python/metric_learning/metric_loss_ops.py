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
"""Implements various metric learning losses."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.summary import summary
try:
  # pylint: disable=g-import-not-at-top
  from sklearn import metrics
  HAS_SKLEARN = True
except ImportError:
  HAS_SKLEARN = False


def pairwise_distance(feature, squared=False):
  """Computes the pairwise distance matrix with numerical stability.

  output[i, j] = || feature[i, :] - feature[j, :] ||_2

  Args:
    feature: 2-D Tensor of size [number of data, feature dimension].
    squared: Boolean, whether or not to square the pairwise distances.

  Returns:
    pairwise_distances: 2-D Tensor of size [number of data, number of data].
  """
  pairwise_distances_squared = math_ops.add(
      math_ops.reduce_sum(
          math_ops.square(feature),
          axis=[1],
          keep_dims=True),
      math_ops.reduce_sum(
          math_ops.square(
              array_ops.transpose(feature)),
          axis=[0],
          keep_dims=True)) - 2.0 * math_ops.matmul(
              feature, array_ops.transpose(feature))

  # Deal with numerical inaccuracies. Set small negatives to zero.
  pairwise_distances_squared = math_ops.maximum(pairwise_distances_squared, 0.0)
  # Get the mask where the zero distances are at.
  error_mask = math_ops.less_equal(pairwise_distances_squared, 0.0)

  # Optionally take the sqrt.
  if squared:
    pairwise_distances = pairwise_distances_squared
  else:
    pairwise_distances = math_ops.sqrt(
        pairwise_distances_squared + math_ops.to_float(error_mask) * 1e-16)

  # Undo conditionally adding 1e-16.
  pairwise_distances = math_ops.multiply(
      pairwise_distances, math_ops.to_float(math_ops.logical_not(error_mask)))

  num_data = array_ops.shape(feature)[0]
  # Explicitly set diagonals to zero.
  mask_offdiagonals = array_ops.ones_like(pairwise_distances) - array_ops.diag(
      array_ops.ones([num_data]))
  pairwise_distances = math_ops.multiply(pairwise_distances, mask_offdiagonals)
  return pairwise_distances


def contrastive_loss(labels, embeddings_anchor, embeddings_positive,
                     margin=1.0):
  """Computes the contrastive loss.

  This loss encourages the embedding to be close to each other for
    the samples of the same label and the embedding to be far apart at least
    by the margin constant for the samples of different labels.
  See: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

  Args:
    labels: 1-D tf.int32 `Tensor` with shape [batch_size] of
      binary labels indicating positive vs negative pair.
    embeddings_anchor: 2-D float `Tensor` of embedding vectors for the anchor
      images. Embeddings should be l2 normalized.
    embeddings_positive: 2-D float `Tensor` of embedding vectors for the
      positive images. Embeddings should be l2 normalized.
    margin: margin term in the loss definition.

  Returns:
    contrastive_loss: tf.float32 scalar.
  """
  # Get per pair distances
  distances = math_ops.sqrt(
      math_ops.reduce_sum(
          math_ops.square(embeddings_anchor - embeddings_positive), 1))

  # Add contrastive loss for the siamese network.
  #   label here is {0,1} for neg, pos.
  return math_ops.reduce_mean(
      math_ops.to_float(labels) * math_ops.square(distances) +
      (1. - math_ops.to_float(labels)) *
      math_ops.square(math_ops.maximum(margin - distances, 0.)),
      name='contrastive_loss')


def masked_maximum(data, mask, dim=1):
  """Computes the axis wise maximum over chosen elements.

  Args:
    data: 2-D float `Tensor` of size [n, m].
    mask: 2-D Boolean `Tensor` of size [n, m].
    dim: The dimension over which to compute the maximum.

  Returns:
    masked_maximums: N-D `Tensor`.
      The maximized dimension is of size 1 after the operation.
  """
  axis_minimums = math_ops.reduce_min(data, dim, keep_dims=True)
  masked_maximums = math_ops.reduce_max(
      math_ops.multiply(
          data - axis_minimums, mask), dim, keep_dims=True) + axis_minimums
  return masked_maximums


def masked_minimum(data, mask, dim=1):
  """Computes the axis wise minimum over chosen elements.

  Args:
    data: 2-D float `Tensor` of size [n, m].
    mask: 2-D Boolean `Tensor` of size [n, m].
    dim: The dimension over which to compute the minimum.

  Returns:
    masked_minimums: N-D `Tensor`.
      The minimized dimension is of size 1 after the operation.
  """
  axis_maximums = math_ops.reduce_max(data, dim, keep_dims=True)
  masked_minimums = math_ops.reduce_min(
      math_ops.multiply(
          data - axis_maximums, mask), dim, keep_dims=True) + axis_maximums
  return masked_minimums


def triplet_semihard_loss(labels, embeddings, margin=1.0):
  """Computes the triplet loss with semi-hard negative mining.

  The loss encourages the positive distances (between a pair of embeddings with
  the same labels) to be smaller than the minimum negative distance among
  which are at least greater than the positive distance plus the margin constant
  (called semi-hard negative) in the mini-batch. If no such negative exists,
  uses the largest negative distance instead.
  See: https://arxiv.org/abs/1503.03832.

  Args:
    labels: 1-D tf.int32 `Tensor` with shape [batch_size] of
      multiclass integer labels.
    embeddings: 2-D float `Tensor` of embedding vectors. Embeddings should
      be l2 normalized.
    margin: Float, margin term in the loss definition.

  Returns:
    triplet_loss: tf.float32 scalar.
  """
  # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
  lshape = array_ops.shape(labels)
  assert lshape.shape == 1
  labels = array_ops.reshape(labels, [lshape[0], 1])

  # Build pairwise squared distance matrix.
  pdist_matrix = pairwise_distance(embeddings, squared=True)
  # Build pairwise binary adjacency matrix.
  adjacency = math_ops.equal(labels, array_ops.transpose(labels))
  # Invert so we can select negatives only.
  adjacency_not = math_ops.logical_not(adjacency)

  batch_size = array_ops.size(labels)

  # Compute the mask.
  pdist_matrix_tile = array_ops.tile(pdist_matrix, [batch_size, 1])
  mask = math_ops.logical_and(
      array_ops.tile(adjacency_not, [batch_size, 1]),
      math_ops.greater(
          pdist_matrix_tile, array_ops.reshape(
              array_ops.transpose(pdist_matrix), [-1, 1])))
  mask_final = array_ops.reshape(
      math_ops.greater(
          math_ops.reduce_sum(
              math_ops.cast(
                  mask, dtype=dtypes.float32), 1, keep_dims=True),
          0.0), [batch_size, batch_size])
  mask_final = array_ops.transpose(mask_final)

  adjacency_not = math_ops.cast(adjacency_not, dtype=dtypes.float32)
  mask = math_ops.cast(mask, dtype=dtypes.float32)

  # negatives_outside: smallest D_an where D_an > D_ap.
  negatives_outside = array_ops.reshape(
      masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size])
  negatives_outside = array_ops.transpose(negatives_outside)

  # negatives_inside: largest D_an.
  negatives_inside = array_ops.tile(
      masked_maximum(pdist_matrix, adjacency_not), [1, batch_size])
  semi_hard_negatives = array_ops.where(
      mask_final, negatives_outside, negatives_inside)

  loss_mat = math_ops.add(margin, pdist_matrix - semi_hard_negatives)

  mask_positives = math_ops.cast(
      adjacency, dtype=dtypes.float32) - array_ops.diag(
          array_ops.ones([batch_size]))

  # In lifted-struct, the authors multiply 0.5 for upper triangular
  #   in semihard, they take all positive pairs except the diagonal.
  num_positives = math_ops.reduce_sum(mask_positives)

  triplet_loss = math_ops.truediv(
      math_ops.reduce_sum(
          math_ops.maximum(
              math_ops.multiply(loss_mat, mask_positives), 0.0)),
      num_positives,
      name='triplet_semihard_loss')

  return triplet_loss


# pylint: disable=line-too-long
def npairs_loss(labels, embeddings_anchor, embeddings_positive,
                reg_lambda=0.002, print_losses=False):
  """Computes the npairs loss.

  Npairs loss expects paired data where a pair is composed of samples from the
  same labels and each pairs in the minibatch have different labels. The loss
  has two components. The first component is the L2 regularizer on the
  embedding vectors. The second component is the sum of cross entropy loss
  which takes each row of the pair-wise similarity matrix as logits and
  the remapped one-hot labels as labels.

  See: http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf

  Args:
    labels: 1-D tf.int32 `Tensor` of shape [batch_size/2].
    embeddings_anchor: 2-D Tensor of shape [batch_size/2, embedding_dim] for the
      embedding vectors for the anchor images. Embeddings should not be
      l2 normalized.
    embeddings_positive: 2-D Tensor of shape [batch_size/2, embedding_dim] for the
      embedding vectors for the positive images. Embeddings should not be
      l2 normalized.
    reg_lambda: Float. L2 regularization term on the embedding vectors.
    print_losses: Boolean. Option to print the xent and l2loss.

  Returns:
    npairs_loss: tf.float32 scalar.
  """
  # pylint: enable=line-too-long
  # Add the regularizer on the embedding.
  reg_anchor = math_ops.reduce_mean(
      math_ops.reduce_sum(math_ops.square(embeddings_anchor), 1))
  reg_positive = math_ops.reduce_mean(
      math_ops.reduce_sum(math_ops.square(embeddings_positive), 1))
  l2loss = math_ops.multiply(
      0.25 * reg_lambda, reg_anchor + reg_positive, name='l2loss')

  # Get per pair similarities.
  similarity_matrix = math_ops.matmul(
      embeddings_anchor, embeddings_positive, transpose_a=False,
      transpose_b=True)

  # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
  lshape = array_ops.shape(labels)
  assert lshape.shape == 1
  labels = array_ops.reshape(labels, [lshape[0], 1])

  labels_remapped = math_ops.to_float(
      math_ops.equal(labels, array_ops.transpose(labels)))
  labels_remapped /= math_ops.reduce_sum(labels_remapped, 1, keep_dims=True)

  # Add the softmax loss.
  xent_loss = nn.softmax_cross_entropy_with_logits(
      logits=similarity_matrix, labels=labels_remapped)
  xent_loss = math_ops.reduce_mean(xent_loss, name='xentropy')

  if print_losses:
    xent_loss = logging_ops.Print(
        xent_loss, ['cross entropy:', xent_loss, 'l2loss:', l2loss])

  return l2loss + xent_loss


def _build_multilabel_adjacency(sparse_labels):
  """Builds multilabel adjacency matrix.

  As of March 14th, 2017, there's no op for the dot product between
  two sparse tensors in TF. However, there is `sparse_minimum` op which is
  equivalent to an AND op between two sparse boolean tensors.
  This computes the dot product between two sparse boolean inputs.

  Args:
    sparse_labels: List of 1-D boolean sparse tensors.

  Returns:
    adjacency_matrix: 2-D dense `Tensor`.
  """
  num_pairs = len(sparse_labels)
  adjacency_matrix = array_ops.zeros([num_pairs, num_pairs])
  for i in range(num_pairs):
    for j in range(num_pairs):
      sparse_dot_product = math_ops.to_float(
          sparse_ops.sparse_reduce_sum(sparse_ops.sparse_minimum(
              sparse_labels[i], sparse_labels[j])))
      sparse_dot_product = array_ops.expand_dims(sparse_dot_product, 0)
      sparse_dot_product = array_ops.expand_dims(sparse_dot_product, 1)
      one_hot_matrix = array_ops.pad(sparse_dot_product,
                                     [[i, num_pairs-i-1],
                                      [j, num_pairs-j-1]], 'CONSTANT')
      adjacency_matrix += one_hot_matrix

  return adjacency_matrix


def npairs_loss_multilabel(sparse_labels, embeddings_anchor,
                           embeddings_positive, reg_lambda=0.002,
                           print_losses=False):
  r"""Computes the npairs loss with multilabel data.

  Npairs loss expects paired data where a pair is composed of samples from the
  same labels and each pairs in the minibatch have different labels. The loss
  has two components. The first component is the L2 regularizer on the
  embedding vectors. The second component is the sum of cross entropy loss
  which takes each row of the pair-wise similarity matrix as logits and
  the remapped one-hot labels as labels. Here, the similarity is defined by the
  dot product between two embedding vectors. S_{i,j} = f(x_i)^T f(x_j)

  To deal with multilabel inputs, we use the count of label intersection
  i.e. L_{i,j} = | set_of_labels_for(i) \cap set_of_labels_for(j) |
  Then we normalize each rows of the count based label matrix so that each row
  sums to one.

  Args:
    sparse_labels: List of 1-D Boolean `SparseTensor` of dense_shape
                   [batch_size/2, num_classes] labels for the anchor-pos pairs.
    embeddings_anchor: 2-D `Tensor` of shape [batch_size/2, embedding_dim] for
      the embedding vectors for the anchor images. Embeddings should not be
      l2 normalized.
    embeddings_positive: 2-D `Tensor` of shape [batch_size/2, embedding_dim] for
      the embedding vectors for the positive images. Embeddings should not be
      l2 normalized.
    reg_lambda: Float. L2 regularization term on the embedding vectors.
    print_losses: Boolean. Option to print the xent and l2loss.

  Returns:
    npairs_loss: tf.float32 scalar.
  Raises:
    TypeError: When the specified sparse_labels is not a `SparseTensor`.
  """
  if False in [isinstance(
      l, sparse_tensor.SparseTensor) for l in sparse_labels]:
    raise TypeError(
        'sparse_labels must be a list of SparseTensors, but got %s' % str(
            sparse_labels))

  with ops.name_scope('NpairsLossMultiLabel'):
    # Add the regularizer on the embedding.
    reg_anchor = math_ops.reduce_mean(
        math_ops.reduce_sum(math_ops.square(embeddings_anchor), 1))
    reg_positive = math_ops.reduce_mean(
        math_ops.reduce_sum(math_ops.square(embeddings_positive), 1))
    l2loss = math_ops.multiply(0.25 * reg_lambda,
                               reg_anchor + reg_positive, name='l2loss')

    # Get per pair similarities.
    similarity_matrix = math_ops.matmul(
        embeddings_anchor, embeddings_positive, transpose_a=False,
        transpose_b=True)

    # TODO(coreylynch): need to check the sparse values
    # TODO(coreylynch): are composed only of 0's and 1's.

    multilabel_adjacency_matrix = _build_multilabel_adjacency(sparse_labels)
    labels_remapped = math_ops.to_float(multilabel_adjacency_matrix)
    labels_remapped /= math_ops.reduce_sum(labels_remapped, 1, keep_dims=True)

    # Add the softmax loss.
    xent_loss = nn.softmax_cross_entropy_with_logits(
        logits=similarity_matrix, labels=labels_remapped)
    xent_loss = math_ops.reduce_mean(xent_loss, name='xentropy')

    if print_losses:
      xent_loss = logging_ops.Print(
          xent_loss, ['cross entropy:', xent_loss, 'l2loss:', l2loss])

    return l2loss + xent_loss


def lifted_struct_loss(labels, embeddings, margin=1.0):
  """Computes the lifted structured loss.

  The loss encourages the positive distances (between a pair of embeddings
  with the same labels) to be smaller than any negative distances (between a
  pair of embeddings with different labels) in the mini-batch in a way
  that is differentiable with respect to the embedding vectors.
  See: https://arxiv.org/abs/1511.06452.

  Args:
    labels: 1-D tf.int32 `Tensor` with shape [batch_size] of
      multiclass integer labels.
    embeddings: 2-D float `Tensor` of embedding vectors. Embeddings should not
      be l2 normalized.
    margin: Float, margin term in the loss definition.

  Returns:
    lifted_loss: tf.float32 scalar.
  """
  # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
  lshape = array_ops.shape(labels)
  assert lshape.shape == 1
  labels = array_ops.reshape(labels, [lshape[0], 1])

  # Build pairwise squared distance matrix.
  pairwise_distances = pairwise_distance(embeddings)

  # Build pairwise binary adjacency matrix.
  adjacency = math_ops.equal(labels, array_ops.transpose(labels))
  # Invert so we can select negatives only.
  adjacency_not = math_ops.logical_not(adjacency)

  batch_size = array_ops.size(labels)

  diff = margin - pairwise_distances
  mask = math_ops.cast(adjacency_not, dtype=dtypes.float32)
  # Safe maximum: Temporarily shift negative distances
  #   above zero before taking max.
  #     this is to take the max only among negatives.
  row_minimums = math_ops.reduce_min(diff, 1, keep_dims=True)
  row_negative_maximums = math_ops.reduce_max(
      math_ops.multiply(
          diff - row_minimums, mask), 1, keep_dims=True) + row_minimums

  # Compute the loss.
  # Keep track of matrix of maximums where M_ij = max(m_i, m_j)
  #   where m_i is the max of alpha - negative D_i's.
  # This matches the Caffe loss layer implementation at:
  #   https://github.com/rksltnl/Caffe-Deep-Metric-Learning-CVPR16/blob/0efd7544a9846f58df923c8b992198ba5c355454/src/caffe/layers/lifted_struct_similarity_softmax_layer.cpp  # pylint: disable=line-too-long

  max_elements = math_ops.maximum(
      row_negative_maximums, array_ops.transpose(row_negative_maximums))
  diff_tiled = array_ops.tile(diff, [batch_size, 1])
  mask_tiled = array_ops.tile(mask, [batch_size, 1])
  max_elements_vect = array_ops.reshape(
      array_ops.transpose(max_elements), [-1, 1])

  loss_exp_left = array_ops.reshape(
      math_ops.reduce_sum(math_ops.multiply(
          math_ops.exp(
              diff_tiled - max_elements_vect),
          mask_tiled), 1, keep_dims=True), [batch_size, batch_size])

  loss_mat = max_elements + math_ops.log(
      loss_exp_left + array_ops.transpose(loss_exp_left))
  # Add the positive distance.
  loss_mat += pairwise_distances

  mask_positives = math_ops.cast(
      adjacency, dtype=dtypes.float32) - array_ops.diag(
          array_ops.ones([batch_size]))

  # *0.5 for upper triangular, and another *0.5 for 1/2 factor for loss^2.
  num_positives = math_ops.reduce_sum(mask_positives) / 2.0

  lifted_loss = math_ops.truediv(
      0.25 * math_ops.reduce_sum(
          math_ops.square(
              math_ops.maximum(
                  math_ops.multiply(loss_mat, mask_positives), 0.0))),
      num_positives,
      name='liftedstruct_loss')
  return lifted_loss


def update_1d_tensor(y, index, value):
  """Updates 1d tensor y so that y[index] = value.

  Args:
    y: 1-D Tensor.
    index: index of y to modify.
    value: new value to write at y[index].

  Returns:
    y_mod: 1-D Tensor. Tensor y after the update.
  """
  value = array_ops.squeeze(value)
  # modify the 1D tensor x at index with value.
  # ex) chosen_ids = update_1D_tensor(chosen_ids, cluster_idx, best_medoid)
  y_before = array_ops.slice(y, [0], [index])
  y_after = array_ops.slice(y, [index + 1], [-1])
  y_mod = array_ops.concat([y_before, [value], y_after], 0)
  return y_mod


def get_cluster_assignment(pairwise_distances, centroid_ids):
  """Assign data points to the neareset centroids.

  Tensorflow has numerical instability and doesn't always choose
    the data point with theoretically zero distance as it's nearest neighbor.
    Thus, for each centroid in centroid_ids, explicitly assign
    the centroid itself as the nearest centroid.
    This is done through the mask tensor and the constraint_vect tensor.

  Args:
    pairwise_distances: 2-D Tensor of pairwise distances.
    centroid_ids: 1-D Tensor of centroid indices.

  Returns:
    y_fixed: 1-D tensor of cluster assignment.
  """
  predictions = math_ops.argmin(
      array_ops.gather(pairwise_distances, centroid_ids), dimension=0)
  batch_size = array_ops.shape(pairwise_distances)[0]

  # Deal with numerical instability
  mask = math_ops.reduce_any(array_ops.one_hot(
      centroid_ids, batch_size, True, False, axis=-1, dtype=dtypes.bool),
                             axis=0)
  constraint_one_hot = math_ops.multiply(
      array_ops.one_hot(centroid_ids,
                        batch_size,
                        array_ops.constant(1, dtype=dtypes.int64),
                        array_ops.constant(0, dtype=dtypes.int64),
                        axis=0,
                        dtype=dtypes.int64),
      math_ops.to_int64(math_ops.range(array_ops.shape(centroid_ids)[0])))
  constraint_vect = math_ops.reduce_sum(
      array_ops.transpose(constraint_one_hot), axis=0)

  y_fixed = array_ops.where(mask, constraint_vect, predictions)
  return y_fixed


def compute_facility_energy(pairwise_distances, centroid_ids):
  """Compute the average travel distance to the assigned centroid.

  Args:
    pairwise_distances: 2-D Tensor of pairwise distances.
    centroid_ids: 1-D Tensor of indices.

  Returns:
    facility_energy: dtypes.float32 scalar.
  """
  return -1.0 * math_ops.reduce_sum(
      math_ops.reduce_min(
          array_ops.gather(pairwise_distances, centroid_ids), axis=0))


def compute_clustering_score(labels, predictions, margin_type):
  """Computes the clustering score via sklearn.metrics functions.

  There are various ways to compute the clustering score. Intuitively,
  we want to measure the agreement of two clustering assignments (labels vs
  predictions) ignoring the permutations and output a score from zero to one.
  (where the values close to one indicate significant agreement).
  This code supports following scoring functions:
    nmi: normalized mutual information
    ami: adjusted mutual information
    ari: adjusted random index
    vmeasure: v-measure
    const: indicator checking whether the two clusterings are the same.
  See http://scikit-learn.org/stable/modules/classes.html#clustering-metrics
    for the detailed descriptions.
  Args:
    labels: 1-D Tensor. ground truth cluster assignment.
    predictions: 1-D Tensor. predicted cluster assignment.
    margin_type: Type of structured margin to use. Default is nmi.
  Returns:
    clustering_score: dtypes.float32 scalar.
      The possible valid values are from zero to one.
      Zero means the worst clustering and one means the perfect clustering.
  Raises:
    ValueError: margin_type is not recognized.
  """
  margin_type_to_func = {
      'nmi': _compute_nmi_score,
      'ami': _compute_ami_score,
      'ari': _compute_ari_score,
      'vmeasure': _compute_vmeasure_score,
      'const': _compute_zeroone_score
  }

  if margin_type not in margin_type_to_func:
    raise ValueError('Unrecognized margin_type: %s' % margin_type)
  clustering_score_fn = margin_type_to_func[margin_type]
  return array_ops.squeeze(clustering_score_fn(labels, predictions))


def _compute_nmi_score(labels, predictions):
  return math_ops.to_float(
      script_ops.py_func(
          metrics.normalized_mutual_info_score, [labels, predictions],
          [dtypes.float64],
          name='nmi'))


def _compute_ami_score(labels, predictions):
  ami_score = math_ops.to_float(
      script_ops.py_func(
          metrics.adjusted_mutual_info_score, [labels, predictions],
          [dtypes.float64],
          name='ami'))
  return math_ops.maximum(0.0, ami_score)


def _compute_ari_score(labels, predictions):
  ari_score = math_ops.to_float(
      script_ops.py_func(
          metrics.adjusted_rand_score, [labels, predictions], [dtypes.float64],
          name='ari'))
  # ari score can go below 0
  # http://scikit-learn.org/stable/modules/clustering.html#adjusted-rand-score
  return math_ops.maximum(0.0, ari_score)


def _compute_vmeasure_score(labels, predictions):
  vmeasure_score = math_ops.to_float(
      script_ops.py_func(
          metrics.v_measure_score, [labels, predictions], [dtypes.float64],
          name='vmeasure'))
  return math_ops.maximum(0.0, vmeasure_score)


def _compute_zeroone_score(labels, predictions):
  zeroone_score = math_ops.to_float(
      math_ops.equal(
          math_ops.reduce_sum(
              math_ops.to_int32(math_ops.equal(labels, predictions))),
          array_ops.shape(labels)[0]))
  return zeroone_score


def _find_loss_augmented_facility_idx(pairwise_distances, labels, chosen_ids,
                                      candidate_ids, margin_multiplier,
                                      margin_type):
  """Find the next centroid that maximizes the loss augmented inference.

  This function is a subroutine called from compute_augmented_facility_locations

  Args:
    pairwise_distances: 2-D Tensor of pairwise distances.
    labels: 1-D Tensor of ground truth cluster assignment.
    chosen_ids: 1-D Tensor of current centroid indices.
    candidate_ids: 1-D Tensor of candidate indices.
    margin_multiplier: multiplication constant.
    margin_type: Type of structured margin to use. Default is nmi.

  Returns:
    integer index.
  """
  num_candidates = array_ops.shape(candidate_ids)[0]

  pairwise_distances_chosen = array_ops.gather(pairwise_distances, chosen_ids)
  pairwise_distances_candidate = array_ops.gather(
      pairwise_distances, candidate_ids)
  pairwise_distances_chosen_tile = array_ops.tile(
      pairwise_distances_chosen, [1, num_candidates])

  candidate_scores = -1.0 * math_ops.reduce_sum(
      array_ops.reshape(
          math_ops.reduce_min(
              array_ops.concat([
                  pairwise_distances_chosen_tile,
                  array_ops.reshape(pairwise_distances_candidate, [1, -1])
              ], 0),
              axis=0,
              keep_dims=True), [num_candidates, -1]),
      axis=1)

  nmi_scores = array_ops.zeros([num_candidates])
  iteration = array_ops.constant(0)

  def func_cond(iteration, nmi_scores):
    del nmi_scores  # Unused in func_cond()
    return iteration < num_candidates

  def func_body(iteration, nmi_scores):
    predictions = get_cluster_assignment(
        pairwise_distances,
        array_ops.concat([chosen_ids, [candidate_ids[iteration]]], 0))
    nmi_score_i = compute_clustering_score(labels, predictions, margin_type)
    pad_before = array_ops.zeros([iteration])
    pad_after = array_ops.zeros([num_candidates - 1 - iteration])
    # return 1 - NMI score as the structured loss.
    #   because NMI is higher the better [0,1].
    return iteration + 1, nmi_scores + array_ops.concat(
        [pad_before, [1.0 - nmi_score_i], pad_after], 0)

  _, nmi_scores = control_flow_ops.while_loop(
      func_cond, func_body, [iteration, nmi_scores])

  candidate_scores = math_ops.add(
      candidate_scores, margin_multiplier * nmi_scores)

  argmax_index = math_ops.to_int32(
      math_ops.argmax(candidate_scores, dimension=0))

  return candidate_ids[argmax_index]


def compute_augmented_facility_locations(pairwise_distances, labels, all_ids,
                                         margin_multiplier, margin_type):
  """Computes the centroid locations.

  Args:
    pairwise_distances: 2-D Tensor of pairwise distances.
    labels: 1-D Tensor of ground truth cluster assignment.
    all_ids: 1-D Tensor of all data indices.
    margin_multiplier: multiplication constant.
    margin_type: Type of structured margin to use. Default is nmi.

  Returns:
    chosen_ids: 1-D Tensor of chosen centroid indices.
  """

  def func_cond_augmented(iteration, chosen_ids):
    del chosen_ids  # Unused argument in func_cond_augmented.
    return iteration < num_classes

  def func_body_augmented(iteration, chosen_ids):
    # find a new facility location to add
    #  based on the clustering score and the NMI score
    candidate_ids = array_ops.setdiff1d(all_ids, chosen_ids)[0]
    new_chosen_idx = _find_loss_augmented_facility_idx(pairwise_distances,
                                                       labels, chosen_ids,
                                                       candidate_ids,
                                                       margin_multiplier,
                                                       margin_type)
    chosen_ids = array_ops.concat([chosen_ids, [new_chosen_idx]], 0)
    return iteration + 1, chosen_ids

  num_classes = array_ops.size(array_ops.unique(labels)[0])
  chosen_ids = array_ops.constant(0, dtype=dtypes.int32, shape=[0])

  # num_classes get determined at run time based on the sampled batch.
  iteration = array_ops.constant(0)

  _, chosen_ids = control_flow_ops.while_loop(
      func_cond_augmented,
      func_body_augmented, [iteration, chosen_ids],
      shape_invariants=[iteration.get_shape(), tensor_shape.TensorShape(
          [None])])
  return chosen_ids


def update_medoid_per_cluster(pairwise_distances, pairwise_distances_subset,
                              labels, chosen_ids, cluster_member_ids,
                              cluster_idx, margin_multiplier, margin_type):
  """Updates the cluster medoid per cluster.

  Args:
    pairwise_distances: 2-D Tensor of pairwise distances.
    pairwise_distances_subset: 2-D Tensor of pairwise distances for one cluster.
    labels: 1-D Tensor of ground truth cluster assignment.
    chosen_ids: 1-D Tensor of cluster centroid indices.
    cluster_member_ids: 1-D Tensor of cluster member indices for one cluster.
    cluster_idx: Index of this one cluster.
    margin_multiplier: multiplication constant.
    margin_type: Type of structured margin to use. Default is nmi.

  Returns:
    chosen_ids: Updated 1-D Tensor of cluster centroid indices.
  """

  def func_cond(iteration, scores_margin):
    del scores_margin  # Unused variable scores_margin.
    return iteration < num_candidates

  def func_body(iteration, scores_margin):
    # swap the current medoid with the candidate cluster member
    candidate_medoid = math_ops.to_int32(cluster_member_ids[iteration])
    tmp_chosen_ids = update_1d_tensor(chosen_ids, cluster_idx, candidate_medoid)
    predictions = get_cluster_assignment(pairwise_distances, tmp_chosen_ids)
    metric_score = compute_clustering_score(labels, predictions, margin_type)
    pad_before = array_ops.zeros([iteration])
    pad_after = array_ops.zeros([num_candidates - 1 - iteration])
    return iteration + 1, scores_margin + array_ops.concat(
        [pad_before, [1.0 - metric_score], pad_after], 0)

  # pairwise_distances_subset is of size [p, 1, 1, p],
  #   the intermediate dummy dimensions at
  #   [1, 2] makes this code work in the edge case where p=1.
  #   this happens if the cluster size is one.
  scores_fac = -1.0 * math_ops.reduce_sum(
      array_ops.squeeze(pairwise_distances_subset, [1, 2]), axis=0)

  iteration = array_ops.constant(0)
  num_candidates = array_ops.size(cluster_member_ids)
  scores_margin = array_ops.zeros([num_candidates])

  _, scores_margin = control_flow_ops.while_loop(func_cond, func_body,
                                                 [iteration, scores_margin])
  candidate_scores = math_ops.add(scores_fac, margin_multiplier * scores_margin)

  argmax_index = math_ops.to_int32(
      math_ops.argmax(candidate_scores, dimension=0))

  best_medoid = math_ops.to_int32(cluster_member_ids[argmax_index])
  chosen_ids = update_1d_tensor(chosen_ids, cluster_idx, best_medoid)
  return chosen_ids


def update_all_medoids(pairwise_distances, predictions, labels, chosen_ids,
                       margin_multiplier, margin_type):
  """Updates all cluster medoids a cluster at a time.

  Args:
    pairwise_distances: 2-D Tensor of pairwise distances.
    predictions: 1-D Tensor of predicted cluster assignment.
    labels: 1-D Tensor of ground truth cluster assignment.
    chosen_ids: 1-D Tensor of cluster centroid indices.
    margin_multiplier: multiplication constant.
    margin_type: Type of structured margin to use. Default is nmi.

  Returns:
    chosen_ids: Updated 1-D Tensor of cluster centroid indices.
  """

  def func_cond_augmented_pam(iteration, chosen_ids):
    del chosen_ids  # Unused argument.
    return iteration < num_classes

  def func_body_augmented_pam(iteration, chosen_ids):
    """Call the update_medoid_per_cluster subroutine."""
    mask = math_ops.equal(
        math_ops.to_int64(predictions), math_ops.to_int64(iteration))
    this_cluster_ids = array_ops.where(mask)

    pairwise_distances_subset = array_ops.transpose(
        array_ops.gather(
            array_ops.transpose(
                array_ops.gather(pairwise_distances, this_cluster_ids)),
            this_cluster_ids))

    chosen_ids = update_medoid_per_cluster(pairwise_distances,
                                           pairwise_distances_subset, labels,
                                           chosen_ids, this_cluster_ids,
                                           iteration, margin_multiplier,
                                           margin_type)
    return iteration + 1, chosen_ids

  unique_class_ids = array_ops.unique(labels)[0]
  num_classes = array_ops.size(unique_class_ids)
  iteration = array_ops.constant(0)

  _, chosen_ids = control_flow_ops.while_loop(
      func_cond_augmented_pam, func_body_augmented_pam, [iteration, chosen_ids])
  return chosen_ids


def compute_augmented_facility_locations_pam(pairwise_distances,
                                             labels,
                                             margin_multiplier,
                                             margin_type,
                                             chosen_ids,
                                             pam_max_iter=5):
  """Refine the cluster centroids with PAM local search.

  For fixed iterations, alternate between updating the cluster assignment
    and updating cluster medoids.

  Args:
    pairwise_distances: 2-D Tensor of pairwise distances.
    labels: 1-D Tensor of ground truth cluster assignment.
    margin_multiplier: multiplication constant.
    margin_type: Type of structured margin to use. Default is nmi.
    chosen_ids: 1-D Tensor of initial estimate of cluster centroids.
    pam_max_iter: Number of refinement iterations.

  Returns:
    chosen_ids: Updated 1-D Tensor of cluster centroid indices.
  """
  for _ in range(pam_max_iter):
    # update the cluster assignment given the chosen_ids (S_pred)
    predictions = get_cluster_assignment(pairwise_distances, chosen_ids)

    # update the medoids per each cluster
    chosen_ids = update_all_medoids(pairwise_distances, predictions, labels,
                                    chosen_ids, margin_multiplier, margin_type)

  return chosen_ids


def compute_gt_cluster_score(pairwise_distances, labels):
  """Compute ground truth facility location score.

  Loop over each unique classes and compute average travel distances.

  Args:
    pairwise_distances: 2-D Tensor of pairwise distances.
    labels: 1-D Tensor of ground truth cluster assignment.

  Returns:
    gt_cluster_score: dtypes.float32 score.
  """
  unique_class_ids = array_ops.unique(labels)[0]
  num_classes = array_ops.size(unique_class_ids)
  iteration = array_ops.constant(0)
  gt_cluster_score = array_ops.constant(0.0, dtype=dtypes.float32)

  def func_cond(iteration, gt_cluster_score):
    del gt_cluster_score  # Unused argument.
    return iteration < num_classes

  def func_body(iteration, gt_cluster_score):
    """Per each cluster, compute the average travel distance."""
    mask = math_ops.equal(labels, unique_class_ids[iteration])
    this_cluster_ids = array_ops.where(mask)
    pairwise_distances_subset = array_ops.transpose(
        array_ops.gather(
            array_ops.transpose(
                array_ops.gather(pairwise_distances, this_cluster_ids)),
            this_cluster_ids))
    this_cluster_score = -1.0 * math_ops.reduce_min(
        math_ops.reduce_sum(
            pairwise_distances_subset, axis=0))
    return iteration + 1, gt_cluster_score + this_cluster_score

  _, gt_cluster_score = control_flow_ops.while_loop(
      func_cond, func_body, [iteration, gt_cluster_score])
  return gt_cluster_score


def cluster_loss(labels,
                 embeddings,
                 margin_multiplier,
                 enable_pam_finetuning=True,
                 margin_type='nmi',
                 print_losses=False):
  """Computes the clustering loss.

  The following structured margins are supported:
    nmi: normalized mutual information
    ami: adjusted mutual information
    ari: adjusted random index
    vmeasure: v-measure
    const: indicator checking whether the two clusterings are the same.

  Args:
    labels: 2-D Tensor of labels of shape [batch size, 1]
    embeddings: 2-D Tensor of embeddings of shape
      [batch size, embedding dimension]. Embeddings should be l2 normalized.
    margin_multiplier: float32 scalar. multiplier on the structured margin term
      See section 3.2 of paper for discussion.
    enable_pam_finetuning: Boolean, Whether to run local pam refinement.
      See section 3.4 of paper for discussion.
    margin_type: Type of structured margin to use. See section 3.2 of
      paper for discussion. Can be 'nmi', 'ami', 'ari', 'vmeasure', 'const'.
    print_losses: Boolean. Option to print the loss.

  Paper: https://arxiv.org/abs/1612.01213.

  Returns:
    clustering_loss: A float32 scalar `Tensor`.
  Raises:
    ImportError: If sklearn dependency is not installed.
  """
  if not HAS_SKLEARN:
    raise ImportError('Cluster loss depends on sklearn.')
  pairwise_distances = pairwise_distance(embeddings)
  labels = array_ops.squeeze(labels)
  all_ids = math_ops.range(array_ops.shape(embeddings)[0])

  # Compute the loss augmented inference and get the cluster centroids.
  chosen_ids = compute_augmented_facility_locations(pairwise_distances, labels,
                                                    all_ids, margin_multiplier,
                                                    margin_type)
  # Given the predicted centroids, compute the clustering score.
  score_pred = compute_facility_energy(pairwise_distances, chosen_ids)

  # Branch whether to use PAM finetuning.
  if enable_pam_finetuning:
    # Initialize with augmented facility solution.
    chosen_ids = compute_augmented_facility_locations_pam(pairwise_distances,
                                                          labels,
                                                          margin_multiplier,
                                                          margin_type,
                                                          chosen_ids)
    score_pred = compute_facility_energy(pairwise_distances, chosen_ids)

  # Given the predicted centroids, compute the cluster assignments.
  predictions = get_cluster_assignment(pairwise_distances, chosen_ids)

  # Compute the clustering (i.e. NMI) score between the two assignments.
  clustering_score_pred = compute_clustering_score(labels, predictions,
                                                   margin_type)

  # Compute the clustering score from labels.
  score_gt = compute_gt_cluster_score(pairwise_distances, labels)

  # Compute the hinge loss.
  clustering_loss = math_ops.maximum(
      score_pred + margin_multiplier * (1.0 - clustering_score_pred) - score_gt,
      0.0,
      name='clustering_loss')
  clustering_loss.set_shape([])

  if print_losses:
    clustering_loss = logging_ops.Print(
        clustering_loss,
        ['clustering_loss: ', clustering_loss, array_ops.shape(
            clustering_loss)])

  # Clustering specific summary.
  summary.scalar('losses/score_pred', score_pred)
  summary.scalar('losses/' + margin_type, clustering_score_pred)
  summary.scalar('losses/score_gt', score_gt)

  return clustering_loss
