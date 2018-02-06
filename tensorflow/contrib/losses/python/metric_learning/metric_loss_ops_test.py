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
"""Tests for triplet_semihard_loss."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.contrib.losses.python import metric_learning as metric_loss_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.platform import test
try:
  # pylint: disable=g-import-not-at-top
  from sklearn import datasets
  from sklearn import metrics
  HAS_SKLEARN = True
except ImportError:
  HAS_SKLEARN = False


def pairwise_distance_np(feature, squared=False):
  """Computes the pairwise distance matrix in numpy.

  Args:
    feature: 2-D numpy array of size [number of data, feature dimension]
    squared: Boolean. If true, output is the pairwise squared euclidean
      distance matrix; else, output is the pairwise euclidean distance matrix.

  Returns:
    pairwise_distances: 2-D numpy array of size
      [number of data, number of data].
  """
  triu = np.triu_indices(feature.shape[0], 1)
  upper_tri_pdists = np.linalg.norm(feature[triu[1]] - feature[triu[0]], axis=1)
  if squared:
    upper_tri_pdists **= 2.
  num_data = feature.shape[0]
  pairwise_distances = np.zeros((num_data, num_data))
  pairwise_distances[np.triu_indices(num_data, 1)] = upper_tri_pdists
  # Make symmetrical.
  pairwise_distances = pairwise_distances + pairwise_distances.T - np.diag(
      pairwise_distances.diagonal())
  return pairwise_distances


class ContrastiveLossTest(test.TestCase):

  def testContrastive(self):
    with self.test_session():
      num_data = 10
      feat_dim = 6
      margin = 1.0

      embeddings_anchor = np.random.rand(num_data, feat_dim).astype(np.float32)
      embeddings_positive = np.random.rand(num_data, feat_dim).astype(
          np.float32)
      labels = np.random.randint(0, 2, size=(num_data,)).astype(np.float32)

      # Compute the loss in NP
      dist = np.sqrt(
          np.sum(np.square(embeddings_anchor - embeddings_positive), axis=1))
      loss_np = np.mean(
          labels * np.square(dist) +
          (1.0 - labels) * np.square(np.maximum(margin - dist, 0.0)))
      # Compute the loss with TF
      loss_tf = metric_loss_ops.contrastive_loss(
          labels=ops.convert_to_tensor(labels),
          embeddings_anchor=ops.convert_to_tensor(embeddings_anchor),
          embeddings_positive=ops.convert_to_tensor(embeddings_positive),
          margin=margin)
      loss_tf = loss_tf.eval()
      self.assertAllClose(loss_np, loss_tf)


class TripletSemiHardLossTest(test.TestCase):

  def testTripletSemiHard(self):
    with self.test_session():
      num_data = 10
      feat_dim = 6
      margin = 1.0
      num_classes = 4

      embedding = np.random.rand(num_data, feat_dim).astype(np.float32)
      labels = np.random.randint(
          0, num_classes, size=(num_data)).astype(np.float32)

      # Reshape labels to compute adjacency matrix.
      labels_reshaped = np.reshape(labels, (labels.shape[0], 1))
      # Compute the loss in NP.
      adjacency = np.equal(labels_reshaped, labels_reshaped.T)

      pdist_matrix = pairwise_distance_np(embedding, squared=True)
      loss_np = 0.0
      num_positives = 0.0
      for i in range(num_data):
        for j in range(num_data):
          if adjacency[i][j] > 0.0 and i != j:
            num_positives += 1.0

            pos_distance = pdist_matrix[i][j]
            neg_distances = []

            for k in range(num_data):
              if adjacency[i][k] == 0:
                neg_distances.append(pdist_matrix[i][k])

            # Sort by distance.
            neg_distances.sort()
            chosen_neg_distance = neg_distances[0]

            for l in range(len(neg_distances)):
              chosen_neg_distance = neg_distances[l]
              if chosen_neg_distance > pos_distance:
                break

            loss_np += np.maximum(
                0.0, margin - chosen_neg_distance + pos_distance)

      loss_np /= num_positives

      # Compute the loss in TF.
      loss_tf = metric_loss_ops.triplet_semihard_loss(
          labels=ops.convert_to_tensor(labels),
          embeddings=ops.convert_to_tensor(embedding),
          margin=margin)
      loss_tf = loss_tf.eval()
      self.assertAllClose(loss_np, loss_tf)


class LiftedStructLossTest(test.TestCase):

  def testLiftedStruct(self):
    with self.test_session():
      num_data = 10
      feat_dim = 6
      margin = 1.0
      num_classes = 4

      embedding = np.random.rand(num_data, feat_dim).astype(np.float32)
      labels = np.random.randint(
          0, num_classes, size=(num_data)).astype(np.float32)
      # Reshape labels to compute adjacency matrix.
      labels_reshaped = np.reshape(labels, (labels.shape[0], 1))

      # Compute the loss in NP
      adjacency = np.equal(labels_reshaped, labels_reshaped.T)
      pdist_matrix = pairwise_distance_np(embedding)
      loss_np = 0.0
      num_constraints = 0.0
      for i in range(num_data):
        for j in range(num_data):
          if adjacency[i][j] > 0.0 and i != j:
            d_pos = pdist_matrix[i][j]
            negs = []
            for k in range(num_data):
              if not adjacency[i][k]:
                negs.append(margin - pdist_matrix[i][k])
            for l in range(num_data):
              if not adjacency[j][l]:
                negs.append(margin - pdist_matrix[j][l])

            negs = np.array(negs)
            max_elem = np.max(negs)
            negs -= max_elem
            negs = np.exp(negs)
            soft_maximum = np.log(np.sum(negs)) + max_elem

            num_constraints += 1.0
            this_loss = max(soft_maximum + d_pos, 0)
            loss_np += this_loss * this_loss

      loss_np = loss_np / num_constraints / 2.0

      # Compute the loss in TF
      loss_tf = metric_loss_ops.lifted_struct_loss(
          labels=ops.convert_to_tensor(labels),
          embeddings=ops.convert_to_tensor(embedding),
          margin=margin)
      loss_tf = loss_tf.eval()
      self.assertAllClose(loss_np, loss_tf)


def convert_to_list_of_sparse_tensor(np_matrix):
  list_of_sparse_tensors = []
  nrows, ncols = np_matrix.shape
  for i in range(nrows):
    sp_indices = []
    for j in range(ncols):
      if np_matrix[i][j] == 1:
        sp_indices.append([j])

    num_non_zeros = len(sp_indices)
    list_of_sparse_tensors.append(sparse_tensor.SparseTensor(
        indices=np.array(sp_indices),
        values=np.ones((num_non_zeros,)),
        dense_shape=np.array([ncols,])))

  return list_of_sparse_tensors


class NpairsLossTest(test.TestCase):

  def testNpairs(self):
    with self.test_session():
      num_data = 15
      feat_dim = 6
      num_classes = 5
      reg_lambda = 0.02

      embeddings_anchor = np.random.rand(num_data, feat_dim).astype(np.float32)
      embeddings_positive = np.random.rand(num_data, feat_dim).astype(
          np.float32)

      labels = np.random.randint(
          0, num_classes, size=(num_data)).astype(np.float32)
      # Reshape labels to compute adjacency matrix.
      labels_reshaped = np.reshape(labels, (labels.shape[0], 1))

      # Compute the loss in NP
      reg_term = np.mean(np.sum(np.square(embeddings_anchor), 1))
      reg_term += np.mean(np.sum(np.square(embeddings_positive), 1))
      reg_term *= 0.25 * reg_lambda

      similarity_matrix = np.matmul(embeddings_anchor, embeddings_positive.T)

      labels_remapped = np.equal(
          labels_reshaped, labels_reshaped.T).astype(np.float32)
      labels_remapped /= np.sum(labels_remapped, axis=1, keepdims=True)

      xent_loss = math_ops.reduce_mean(nn.softmax_cross_entropy_with_logits(
          logits=ops.convert_to_tensor(similarity_matrix),
          labels=ops.convert_to_tensor(labels_remapped))).eval()
      loss_np = xent_loss + reg_term

      # Compute the loss in TF
      loss_tf = metric_loss_ops.npairs_loss(
          labels=ops.convert_to_tensor(labels),
          embeddings_anchor=ops.convert_to_tensor(embeddings_anchor),
          embeddings_positive=ops.convert_to_tensor(embeddings_positive),
          reg_lambda=reg_lambda)
      loss_tf = loss_tf.eval()
      self.assertAllClose(loss_np, loss_tf)


class NpairsLossMultiLabelTest(test.TestCase):

  def testNpairsMultiLabelLossWithSingleLabelEqualsNpairsLoss(self):
    with self.test_session():
      num_data = 15
      feat_dim = 6
      reg_lambda = 0.02

      embeddings_anchor = np.random.rand(num_data, feat_dim).astype(np.float32)
      embeddings_positive = np.random.rand(num_data, feat_dim).astype(
          np.float32)
      labels = np.arange(num_data)
      labels = np.reshape(labels, -1)

      # Compute vanila npairs loss.
      loss_npairs = metric_loss_ops.npairs_loss(
          labels=ops.convert_to_tensor(labels),
          embeddings_anchor=ops.convert_to_tensor(embeddings_anchor),
          embeddings_positive=ops.convert_to_tensor(embeddings_positive),
          reg_lambda=reg_lambda).eval()

      # Compute npairs multilabel loss.
      labels_one_hot = np.identity(num_data)
      loss_npairs_multilabel = metric_loss_ops.npairs_loss_multilabel(
          sparse_labels=convert_to_list_of_sparse_tensor(labels_one_hot),
          embeddings_anchor=ops.convert_to_tensor(embeddings_anchor),
          embeddings_positive=ops.convert_to_tensor(embeddings_positive),
          reg_lambda=reg_lambda).eval()

      self.assertAllClose(loss_npairs, loss_npairs_multilabel)

  def testNpairsMultiLabel(self):
    with self.test_session():
      num_data = 15
      feat_dim = 6
      num_classes = 10
      reg_lambda = 0.02

      embeddings_anchor = np.random.rand(num_data, feat_dim).astype(np.float32)
      embeddings_positive = np.random.rand(num_data, feat_dim).astype(
          np.float32)

      labels = np.random.randint(0, 2, (num_data, num_classes))
      # set entire column to one so that each row has at least one bit set.
      labels[:, -1] = 1

      # Compute the loss in NP
      reg_term = np.mean(np.sum(np.square(embeddings_anchor), 1))
      reg_term += np.mean(np.sum(np.square(embeddings_positive), 1))
      reg_term *= 0.25 * reg_lambda

      similarity_matrix = np.matmul(embeddings_anchor, embeddings_positive.T)

      labels_remapped = np.dot(labels, labels.T).astype(np.float)
      labels_remapped /= np.sum(labels_remapped, 1, keepdims=True)

      xent_loss = math_ops.reduce_mean(nn.softmax_cross_entropy_with_logits(
          logits=ops.convert_to_tensor(similarity_matrix),
          labels=ops.convert_to_tensor(labels_remapped))).eval()
      loss_np = xent_loss + reg_term

      # Compute the loss in TF
      loss_tf = metric_loss_ops.npairs_loss_multilabel(
          sparse_labels=convert_to_list_of_sparse_tensor(labels),
          embeddings_anchor=ops.convert_to_tensor(embeddings_anchor),
          embeddings_positive=ops.convert_to_tensor(embeddings_positive),
          reg_lambda=reg_lambda)
      loss_tf = loss_tf.eval()

      self.assertAllClose(loss_np, loss_tf)


def compute_ground_truth_cluster_score(feat, y):
  y_unique = np.unique(y)
  score_gt_np = 0.0
  for c in y_unique:
    feat_subset = feat[y == c, :]
    pdist_subset = pairwise_distance_np(feat_subset)
    score_gt_np += -1.0 * np.min(np.sum(pdist_subset, axis=0))
  score_gt_np = score_gt_np.astype(np.float32)
  return score_gt_np


def compute_cluster_loss_numpy(feat,
                               y,
                               margin_multiplier=1.0,
                               enable_pam_finetuning=True):
  if enable_pam_finetuning:
    facility = ForwardGreedyFacility(
        n_clusters=np.unique(y).size).pam_augmented_fit(feat, y,
                                                        margin_multiplier)
  else:
    facility = ForwardGreedyFacility(
        n_clusters=np.unique(y).size).loss_augmented_fit(feat, y,
                                                         margin_multiplier)

  score_augmented = facility.score_aug_
  score_gt = compute_ground_truth_cluster_score(feat, y)
  return np.maximum(np.float32(0.0), score_augmented - score_gt)


class ForwardGreedyFacility(object):

  def __init__(self, n_clusters=8):
    self.n_clusters = n_clusters
    self.center_ics_ = None

  def _check_init_args(self):
    # Check n_clusters.
    if (self.n_clusters is None or self.n_clusters <= 0 or
        not isinstance(self.n_clusters, int)):
      raise ValueError('n_clusters has to be nonnegative integer.')

  def loss_augmented_fit(self, feat, y, loss_mult):
    """Fit K-Medoids to the provided data."""
    self._check_init_args()
    # Check that the array is good and attempt to convert it to
    # Numpy array if possible.
    feat = self._check_array(feat)
    # Apply distance metric to get the distance matrix.
    pdists = pairwise_distance_np(feat)

    num_data = feat.shape[0]
    candidate_ids = list(range(num_data))
    candidate_scores = np.zeros(num_data,)
    subset = []

    k = 0
    while k < self.n_clusters:
      candidate_scores = []
      for i in candidate_ids:
        # push i to subset.
        subset.append(i)
        marginal_cost = -1.0 * np.sum(np.min(pdists[:, subset], axis=1))
        loss = 1.0 - metrics.normalized_mutual_info_score(
            y, self._get_cluster_ics(pdists, subset))
        candidate_scores.append(marginal_cost + loss_mult * loss)
        # remove i from subset.
        subset.pop()

      # push i_star to subset.
      i_star = candidate_ids[np.argmax(candidate_scores)]
      subset.append(i_star)
      # remove i_star from candidate indices.
      candidate_ids.remove(i_star)
      k += 1

    # Expose labels_ which are the assignments of
    # the training data to clusters.
    self.labels_ = self._get_cluster_ics(pdists, subset)
    # Expose cluster centers, i.e. medoids.
    self.cluster_centers_ = feat.take(subset, axis=0)
    # Expose indices of chosen cluster centers.
    self.center_ics_ = subset
    # Expose the score = -\sum_{i \in V} min_{j \in S} || x_i - x_j ||
    self.score_ = np.float32(-1.0) * self._get_facility_distance(pdists, subset)
    self.score_aug_ = self.score_ + loss_mult * (
        1.0 - metrics.normalized_mutual_info_score(
            y, self._get_cluster_ics(pdists, subset)))
    self.score_aug_ = self.score_aug_.astype(np.float32)
    # Expose the chosen cluster indices.
    self.subset_ = subset
    return self

  def _augmented_update_medoid_ics_in_place(self, pdists, y_gt, cluster_ics,
                                            medoid_ics, loss_mult):
    for cluster_idx in range(self.n_clusters):
      # y_pred = self._get_cluster_ics(D, medoid_ics)
      # Don't prematurely do the assignment step.
      # Do this after we've updated all cluster medoids.
      y_pred = cluster_ics

      if sum(y_pred == cluster_idx) == 0:
        # Cluster is empty.
        continue

      curr_score = (
          -1.0 * np.sum(
              pdists[medoid_ics[cluster_idx], y_pred == cluster_idx]) +
          loss_mult * (1.0 - metrics.normalized_mutual_info_score(
              y_gt, y_pred)))

      pdist_in = pdists[y_pred == cluster_idx, :]
      pdist_in = pdist_in[:, y_pred == cluster_idx]

      all_scores_fac = np.sum(-1.0 * pdist_in, axis=1)
      all_scores_loss = []
      for i in range(y_pred.size):
        if y_pred[i] != cluster_idx:
          continue
        # remove this cluster's current centroid
        medoid_ics_i = medoid_ics[:cluster_idx] + medoid_ics[cluster_idx + 1:]
        # add this new candidate to the centroid list
        medoid_ics_i += [i]
        y_pred_i = self._get_cluster_ics(pdists, medoid_ics_i)
        all_scores_loss.append(loss_mult * (
            1.0 - metrics.normalized_mutual_info_score(y_gt, y_pred_i)))

      all_scores = all_scores_fac + all_scores_loss
      max_score_idx = np.argmax(all_scores)
      max_score = all_scores[max_score_idx]

      if max_score > curr_score:
        medoid_ics[cluster_idx] = np.where(
            y_pred == cluster_idx)[0][max_score_idx]

  def pam_augmented_fit(self, feat, y, loss_mult):
    pam_max_iter = 5
    self._check_init_args()
    feat = self._check_array(feat)
    pdists = pairwise_distance_np(feat)
    self.loss_augmented_fit(feat, y, loss_mult)
    print('PAM -1 (before PAM): score: %f, score_aug: %f' % (
        self.score_, self.score_aug_))
    # Initialize from loss augmented facility location
    subset = self.center_ics_
    for iter_ in range(pam_max_iter):
      # update the cluster assignment
      cluster_ics = self._get_cluster_ics(pdists, subset)
      # update the medoid for each clusters
      self._augmented_update_medoid_ics_in_place(pdists, y, cluster_ics, subset,
                                                 loss_mult)
      self.score_ = np.float32(-1.0) * self._get_facility_distance(
          pdists, subset)
      self.score_aug_ = self.score_ + loss_mult * (
          1.0 - metrics.normalized_mutual_info_score(
              y, self._get_cluster_ics(pdists, subset)))
      self.score_aug_ = self.score_aug_.astype(np.float32)
      print('PAM iter: %d, score: %f, score_aug: %f' % (iter_, self.score_,
                                                        self.score_aug_))

    self.center_ics_ = subset
    self.labels_ = cluster_ics
    return self

  def _check_array(self, feat):
    # Check that the number of clusters is less than or equal to
    # the number of samples
    if self.n_clusters > feat.shape[0]:
      raise ValueError('The number of medoids ' + '({}) '.format(
          self.n_clusters) + 'must be larger than the number ' +
                       'of samples ({})'.format(feat.shape[0]))
    return feat

  def _get_cluster_ics(self, pdists, subset):
    """Returns cluster indices for pdist and current medoid indices."""
    # Assign data points to clusters based on
    # which cluster assignment yields
    # the smallest distance`
    cluster_ics = np.argmin(pdists[subset, :], axis=0)
    return cluster_ics

  def _get_facility_distance(self, pdists, subset):
    return np.sum(np.min(pdists[subset, :], axis=0))


class ClusterLossTest(test.TestCase):

  def _genClusters(self, n_samples, n_clusters):
    blobs = datasets.make_blobs(
        n_samples=n_samples, centers=n_clusters)
    embedding, labels = blobs
    embedding = (embedding - embedding.mean(axis=0)) / embedding.std(axis=0)
    embedding = embedding.astype(np.float32)
    return embedding, labels

  def testClusteringLossPAMOff(self):
    if not HAS_SKLEARN:
      return
    with self.test_session():
      margin_multiplier = 10.0
      embeddings, labels = self._genClusters(n_samples=128, n_clusters=64)

      loss_np = compute_cluster_loss_numpy(
          embeddings, labels, margin_multiplier, enable_pam_finetuning=False)
      loss_tf = metric_loss_ops.cluster_loss(
          labels=ops.convert_to_tensor(labels),
          embeddings=ops.convert_to_tensor(embeddings),
          margin_multiplier=margin_multiplier,
          enable_pam_finetuning=False)
      loss_tf = loss_tf.eval()
      self.assertAllClose(loss_np, loss_tf)

  def testClusteringLossPAMOn(self):
    if not HAS_SKLEARN:
      return
    with self.test_session():
      margin_multiplier = 10.0
      embeddings, labels = self._genClusters(n_samples=128, n_clusters=64)

      loss_np = compute_cluster_loss_numpy(
          embeddings, labels, margin_multiplier, enable_pam_finetuning=True)
      loss_tf = metric_loss_ops.cluster_loss(
          labels=ops.convert_to_tensor(labels),
          embeddings=ops.convert_to_tensor(embeddings),
          margin_multiplier=margin_multiplier,
          enable_pam_finetuning=True)
      loss_tf = loss_tf.eval()
      self.assertAllClose(loss_np, loss_tf)

if __name__ == '__main__':
  test.main()
