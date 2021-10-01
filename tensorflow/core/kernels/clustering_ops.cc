// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License.  You may obtain a copy
// of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
// License for the specific language governing permissions and limitations under
// the License.
// ==============================================================================

#define EIGEN_USE_THREADS

#include <algorithm>
#include <memory>
#include <numeric>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/top_n.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/byte_order.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace {
using errors::InvalidArgument;

template <typename Scalar>
using RowMajorMatrix =
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

using MatrixXfRowMajor = RowMajorMatrix<float>;
using MatrixXi64RowMajor = RowMajorMatrix<int64_t>;

// Ideally this should be computed by dividing L3 cache size by the number of
// physical CPUs. Since there isn't a portable method to do this, we are using
// a conservative estimate here.
const int64_t kDefaultL3CachePerCpu = 1 << 20;

// These values were determined by performing a parameter sweep on the
// NearestNeighborsOp benchmark.
const int64_t kNearestNeighborsCentersMaxBlockSize = 1024;
const int64_t kNearestNeighborsPointsMinBlockSize = 16;

// Returns the smallest multiple of a that is not smaller than b.
int64_t NextMultiple(int64_t a, int64_t b) {
  const int64_t remainder = b % a;
  return remainder == 0 ? b : (b + a - remainder);
}

// Returns a / b rounded up to the next higher integer.
int64_t CeilOfRatio(int64_t a, int64_t b) { return (a + b - 1) / b; }

}  // namespace

// Implementation of K-means++ initialization. Samples points iteratively in
// proportion to the squared distances from selected points.
// TODO(ands): Add support for other distance metrics.
class KmeansPlusPlusInitializationOp : public OpKernel {
 public:
  explicit KmeansPlusPlusInitializationOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->MatchSignature(
                       {DT_FLOAT, DT_INT64, DT_INT64, DT_INT64}, {DT_FLOAT}));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& points_tensor = context->input(0);
    const Tensor& num_to_sample_tensor = context->input(1);
    const Tensor& seed_tensor = context->input(2);
    const Tensor& num_retries_per_sample_tensor = context->input(3);

    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(points_tensor.shape()),
                InvalidArgument("Input points should be a matrix."));
    OP_REQUIRES(context,
                TensorShapeUtils::IsScalar(num_to_sample_tensor.shape()),
                InvalidArgument("Input num_to_sample should be a scalar."));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(seed_tensor.shape()),
                InvalidArgument("Input seed should be a scalar."));
    OP_REQUIRES(
        context,
        TensorShapeUtils::IsScalar(num_retries_per_sample_tensor.shape()),
        InvalidArgument("Input num_retries_per_sample should be a scalar."));

    const int64_t num_points = points_tensor.dim_size(0);
    const int64_t point_dimensions = points_tensor.dim_size(1);
    const int64_t num_to_sample = num_to_sample_tensor.scalar<int64_t>()();
    const int64_t seed = seed_tensor.scalar<int64_t>()();
    const int64_t num_retries_per_sample = [&]() {
      const int64_t value = num_retries_per_sample_tensor.scalar<int64_t>()();
      return value >= 0 ? value
                        : 2 + static_cast<int64_t>(std::log(num_to_sample));
    }();

    OP_REQUIRES(context, num_points > 0,
                InvalidArgument("Expected points.rows() > 0."));
    OP_REQUIRES(context, num_to_sample > 0,
                InvalidArgument("Expected num_to_sample > 0."));
    OP_REQUIRES(context, num_to_sample <= num_points,
                InvalidArgument("Expected num_to_sample <= points.rows(). ",
                                num_to_sample, " vs ", num_points, "."));

    Tensor* output_sampled_points_tensor;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       0, TensorShape({num_to_sample, point_dimensions}),
                       &output_sampled_points_tensor));

    const Eigen::Map<const MatrixXfRowMajor> points(
        points_tensor.matrix<float>().data(), num_points, point_dimensions);
    const Eigen::VectorXf points_half_squared_norm =
        0.5 * points.rowwise().squaredNorm();

    Eigen::Map<MatrixXfRowMajor> sampled_points(
        output_sampled_points_tensor->matrix<float>().data(), num_to_sample,
        point_dimensions);
    std::unordered_set<int64_t> sampled_indices;

    random::PhiloxRandom random(seed);
    random::SimplePhilox rng(&random);

    auto add_one_point = [&](int64_t from, int64_t to) {
      from = std::min(from, num_points - 1);
      sampled_points.row(to) = points.row(from);
      sampled_indices.insert(from);
    };

    // Distances from all points to nearest selected point. Initialize with
    // distances to first selected point.
    Eigen::VectorXf min_distances(num_points);
    min_distances.fill(std::numeric_limits<float>::infinity());
    Eigen::VectorXf min_distances_cumsum(num_points);

    auto draw_one_sample = [&]() -> int64 {
      if (sampled_indices.empty()) return rng.Uniform64(num_points);
      int64_t index = 0;
      do {
        // If v is drawn from Uniform[0, distances.sum()), then
        // Prob[cumsum(distances)(i - 1) <= v < cumsum(distances)(i)] is
        // proportional to distances(i).
        index = std::upper_bound(
                    min_distances_cumsum.data(),
                    min_distances_cumsum.data() + num_points,
                    rng.RandFloat() * min_distances_cumsum(num_points - 1)) -
                min_distances_cumsum.data();
      } while (sampled_indices.find(index) != sampled_indices.end());
      return index;
    };

    auto sample_one_point = [&]() {
      const int64_t sampled_index = draw_one_sample();
      min_distances = min_distances.cwiseMin(GetHalfSquaredDistancesToY(
          points, points_half_squared_norm, points.row(sampled_index),
          points_half_squared_norm(sampled_index)));
      return sampled_index;
    };

    auto sample_one_point_with_retries = [&]() {
      Eigen::VectorXf best_new_min_distances(num_points);
      float best_potential = std::numeric_limits<float>::infinity();
      int64_t best_sampled_index = 0;
      for (int i = 1 + num_retries_per_sample; i > 0; --i) {
        const int64_t sampled_index = draw_one_sample();
        Eigen::VectorXf new_min_distances =
            min_distances.cwiseMin(GetHalfSquaredDistancesToY(
                points, points_half_squared_norm, points.row(sampled_index),
                points_half_squared_norm(sampled_index)));
        const float potential = new_min_distances.sum();
        if (potential < best_potential) {
          best_potential = potential;
          best_sampled_index = sampled_index;
          best_new_min_distances.swap(new_min_distances);
        }
      }
      min_distances.swap(best_new_min_distances);
      return best_sampled_index;
    };

    for (int64_t i = 0; i < num_to_sample; ++i) {
      if (i > 0) {
        std::partial_sum(min_distances.data(),
                         min_distances.data() + num_points,
                         min_distances_cumsum.data());
      }
      int64_t next = num_retries_per_sample == 0
                         ? sample_one_point()
                         : sample_one_point_with_retries();
      add_one_point(next, i);
    }
  }

 private:
  // Returns a column vector with the i-th element set to half the squared
  // euclidean distance between the i-th row of xs, and y. Precomputed norms for
  // each row of xs and y must be provided for efficiency.
  // TODO(ands): Parallelize this for large xs.
  static Eigen::VectorXf GetHalfSquaredDistancesToY(
      const Eigen::Ref<const MatrixXfRowMajor>& xs,
      const Eigen::Ref<const Eigen::VectorXf>& xs_half_squared_norm,
      const Eigen::Ref<const Eigen::RowVectorXf>& y,
      float y_half_squared_norm) {
    // Squared distance between points xs_i and y is:
    //   || xs_i ||^2 - 2 <xs_i, y> + || y ||^2
    return (xs_half_squared_norm - xs * y.transpose()).array() +
           y_half_squared_norm;
  }
};

REGISTER_KERNEL_BUILDER(Name("KmeansPlusPlusInitialization").Device(DEVICE_CPU),
                        KmeansPlusPlusInitializationOp);

// Implementation of one single Markov Chain for the k-MC^2 algorithm
class KMC2ChainInitializationOp : public OpKernel {
 public:
  explicit KMC2ChainInitializationOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->MatchSignature({DT_FLOAT, DT_INT64}, {DT_INT64}));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& distances_tensor = context->input(0);
    const Tensor& seed_tensor = context->input(1);
    OP_REQUIRES(context, TensorShapeUtils::IsVector(distances_tensor.shape()),
                InvalidArgument("Input distances should be a vector."));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(seed_tensor.shape()),
                InvalidArgument("Input seed should be a scalar."));
    const int64_t num_points = distances_tensor.dim_size(0);
    const int64_t seed = seed_tensor.scalar<int64_t>()();
    OP_REQUIRES(context, num_points > 0,
                InvalidArgument("Expected distances_tensor.size() > 0."));

    random::PhiloxRandom random(seed);
    random::SimplePhilox rng(&random);

    auto distances = distances_tensor.flat<float>();
    // Set the initial state of the Markov chain to be the first candidate.
    int64_t selected_index = 0;
    float selected_distance = distances(selected_index);
    // Build a Markov chain of length num_points.
    for (int64_t i = 1; i < num_points; ++i) {
      const float candidate_distance = distances(i);
      // Set the next state of the Markov chain to be the candidate with
      // probability min(1, candidate_distance/selected_distance).
      if (candidate_distance > rng.RandFloat() * selected_distance) {
        selected_index = i;
        selected_distance = candidate_distance;
      }
    }

    Tensor* output_sampled_index_tensor;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({}),
                                            &output_sampled_index_tensor));
    auto output = output_sampled_index_tensor->scalar<int64_t>();
    // Return the last state of the Markov chain as the new center.
    output() = selected_index;
  }
};

REGISTER_KERNEL_BUILDER(Name("KMC2ChainInitialization").Device(DEVICE_CPU),
                        KMC2ChainInitializationOp);

// Operator for computing the nearest neighbors for a set of points.
class NearestNeighborsOp : public OpKernel {
 public:
  explicit NearestNeighborsOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->MatchSignature({DT_FLOAT, DT_FLOAT, DT_INT64},
                                           {DT_INT64, DT_FLOAT}));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& points_tensor = context->input(0);
    const Tensor& centers_tensor = context->input(1);
    const Tensor& k_tensor = context->input(2);

    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(points_tensor.shape()),
                InvalidArgument("Input points should be a matrix."));
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(centers_tensor.shape()),
                InvalidArgument("Input centers should be a matrix."));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(k_tensor.shape()),
                InvalidArgument("Input k should be a scalar."));

    const int64_t num_points = points_tensor.dim_size(0);
    const int64_t point_dimensions = points_tensor.dim_size(1);
    const int64_t num_centers = centers_tensor.dim_size(0);
    const int64_t center_dimensions = centers_tensor.dim_size(1);

    OP_REQUIRES(context, num_points > 0,
                InvalidArgument("Expected points.rows() > 0."));
    OP_REQUIRES(
        context, point_dimensions == center_dimensions,
        InvalidArgument("Expected point_dimensions == center_dimensions: ",
                        point_dimensions, " vs ", center_dimensions, "."));

    const Eigen::Map<const MatrixXfRowMajor> points(
        points_tensor.matrix<float>().data(), num_points, point_dimensions);
    const Eigen::Map<const MatrixXfRowMajor> centers(
        centers_tensor.matrix<float>().data(), num_centers, center_dimensions);
    const int64_t k =
        std::min<int64_t>(num_centers, k_tensor.scalar<int64_t>()());

    Tensor* output_nearest_center_indices_tensor;
    Tensor* output_nearest_center_distances_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({num_points, k}),
                                &output_nearest_center_indices_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(
                                1, TensorShape({num_points, k}),
                                &output_nearest_center_distances_tensor));

    if (k == 0) return;

    Eigen::Map<MatrixXi64RowMajor> nearest_center_indices(
        output_nearest_center_indices_tensor->matrix<int64_t>().data(),
        num_points, k);
    Eigen::Map<MatrixXfRowMajor> nearest_center_distances(
        output_nearest_center_distances_tensor->matrix<float>().data(),
        num_points, k);

    const Eigen::VectorXf centers_half_squared_norm =
        0.5 * centers.rowwise().squaredNorm();

    // The distance computation is sharded to take advantage of multiple cores
    // and to allow intermediate values to reside in L3 cache. This is done by
    // sharding the points and centers as follows:
    //
    // 1. Centers are sharded such that each block of centers has at most
    //    kNearestNeighborsCentersMaxBlockSize rows.
    // 2. Points are sharded, and each block of points is multiplied with each
    //    block of centers. The block size of points is chosen such that the
    //    point coordinates (point_dimensions) and the matrix of distances to
    //    each center in one block -- the intermediate data -- fits in L3 cache.
    // 3. After performing each block-block distance computation, the results
    //    are reduced to a set of k nearest centers as soon as possible. This
    //    decreases total memory I/O.
    auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
    const int64_t num_threads = worker_threads.num_threads;
    // This kernel might be configured to use fewer than the total number of
    // available CPUs on the host machine. To avoid destructive interference
    // with other jobs running on the host machine, we must only use a fraction
    // of total available L3 cache. Unfortunately, we cannot query the host
    // machine to get the number of physical CPUs. So, we use a fixed per-CPU
    // budget and scale it by the number of CPUs available to this operation.
    const int64_t total_memory_budget =
        kDefaultL3CachePerCpu * port::NumSchedulableCPUs();
    // Compute the number of blocks into which rows of points must be split so
    // that the distance matrix and the block of points can fit in cache. One
    // row of points will yield a vector of distances to each center in a block.
    const int64_t bytes_per_row =
        (std::min(kNearestNeighborsCentersMaxBlockSize,
                  num_centers) /* centers in a block */
         + point_dimensions /* coordinates of one point */) *
        sizeof(float);
    // The memory needed for storing the centers being processed. This is shared
    // by all workers. Adding slack to the number of threads to avoid incorrect
    // cache eviction when a new block of centers is loaded.
    const int64_t bytes_for_centers =
        std::min(num_centers,
                 (num_threads + 2) * kNearestNeighborsCentersMaxBlockSize) *
        point_dimensions * sizeof(float);
    // The memory budget available for workers to store their distance matrices.
    const int64_t available_memory_budget =
        total_memory_budget - bytes_for_centers;
    // That memory budget is shared by all threads.
    const int64_t rows_per_block = std::max<int64_t>(
        kNearestNeighborsPointsMinBlockSize,
        available_memory_budget / num_threads / bytes_per_row);
    // Divide rows into almost uniformly-sized units of work that are small
    // enough for the memory budget (rows_per_block). Round up to a multiple of
    // the number of threads.
    const int64_t num_units =
        NextMultiple(num_threads, CeilOfRatio(num_points, rows_per_block));
    auto work = [&](int64_t start, int64_t limit) {
      for (; start < limit; ++start) {
        const int64_t start_row = num_points * start / num_units;
        const int64_t limit_row = num_points * (start + 1) / num_units;
        DCHECK_LE(limit_row, num_points);
        const int64_t num_rows = limit_row - start_row;
        auto points_shard = points.middleRows(start_row, num_rows);
        const Eigen::VectorXf points_half_squared_norm =
            0.5 * points_shard.rowwise().squaredNorm();
        auto nearest_center_indices_shard =
            nearest_center_indices.middleRows(start_row, num_rows);
        auto nearest_center_distances_shard =
            nearest_center_distances.middleRows(start_row, num_rows);
        FindKNearestCenters(k, points_shard, points_half_squared_norm, centers,
                            centers_half_squared_norm,
                            nearest_center_indices_shard,
                            nearest_center_distances_shard);
      }
    };

    const int64_t units_per_thread = num_units / num_threads;
    BlockingCounter counter(num_threads - 1);
    for (int64_t i = 1; i < num_threads; ++i) {
      const int64_t start = i * units_per_thread;
      const int64_t limit = start + units_per_thread;
      worker_threads.workers->Schedule([work, &counter, start, limit]() {
        work(start, limit);
        counter.DecrementCount();
      });
    }
    work(0, units_per_thread);
    counter.Wait();
  }

 private:
  static void FindKNearestCenters(
      int64_t k, const Eigen::Ref<const MatrixXfRowMajor>& points,
      const Eigen::Ref<const Eigen::VectorXf>& points_half_squared_norm,
      const Eigen::Ref<const MatrixXfRowMajor>& centers,
      const Eigen::Ref<const Eigen::VectorXf>& centers_half_squared_norm,
      const Eigen::Ref<MatrixXi64RowMajor>& nearest_center_indices,
      const Eigen::Ref<MatrixXfRowMajor>& nearest_center_distances) {
    DCHECK_LE(k, centers.rows());
    if (centers.rows() <= kNearestNeighborsCentersMaxBlockSize) {
      FindKNearestCentersOneBlock(k, points, points_half_squared_norm, centers,
                                  centers_half_squared_norm,
                                  nearest_center_indices,
                                  nearest_center_distances);
    } else {
      FindKNearestCentersBlockwise(k, points, points_half_squared_norm, centers,
                                   centers_half_squared_norm,
                                   nearest_center_indices,
                                   nearest_center_distances);
    }
  }

  static void FindKNearestCentersOneBlock(
      int64_t k, const Eigen::Ref<const MatrixXfRowMajor>& points,
      const Eigen::Ref<const Eigen::VectorXf>& points_half_squared_norm,
      const Eigen::Ref<const MatrixXfRowMajor>& centers,
      const Eigen::Ref<const Eigen::VectorXf>& centers_half_squared_norm,
      Eigen::Ref<MatrixXi64RowMajor> nearest_center_indices,
      Eigen::Ref<MatrixXfRowMajor> nearest_center_distances) {
    DCHECK_LE(k, centers.rows());
    const int64_t num_points = points.rows();
    const MatrixXfRowMajor inner_product = points * centers.transpose();
    // Find nearest neighbors.
    if (k == 1) {
      for (int i = 0; i < num_points; ++i) {
        int64_t index;
        nearest_center_distances(i, 0) =
            2.0 *
            (points_half_squared_norm(i) +
             (centers_half_squared_norm.transpose() - inner_product.row(i))
                 .minCoeff(&index));
        nearest_center_indices(i, 0) = index;
      }
    } else {
      // Select k nearest centers for each point.
      using Center = std::pair<float, int64_t>;
      const int64_t num_centers = centers.rows();
      gtl::TopN<Center, std::less<Center>> selector(k);
      std::unique_ptr<std::vector<Center>> nearest_centers;
      for (int i = 0; i < num_points; ++i) {
        selector.reserve(num_centers);
        for (int j = 0; j < num_centers; ++j) {
          const float partial_distance =
              centers_half_squared_norm(j) - inner_product(i, j);
          selector.push(Center(partial_distance, j));
        }
        nearest_centers.reset(selector.Extract());
        selector.Reset();
        const float point_half_squared_norm = points_half_squared_norm(i);
        for (int j = 0; j < k; ++j) {
          const Center& center = (*nearest_centers)[j];
          nearest_center_distances(i, j) =
              2.0 * (point_half_squared_norm + center.first);
          nearest_center_indices(i, j) = center.second;
        }
      }
    }
  }

  static void FindKNearestCentersBlockwise(
      int64_t k, const Eigen::Ref<const MatrixXfRowMajor>& points,
      const Eigen::Ref<const Eigen::VectorXf>& points_half_squared_norm,
      const Eigen::Ref<const MatrixXfRowMajor>& centers,
      const Eigen::Ref<const Eigen::VectorXf>& centers_half_squared_norm,
      Eigen::Ref<MatrixXi64RowMajor> nearest_center_indices,
      Eigen::Ref<MatrixXfRowMajor> nearest_center_distances) {
    const int64_t num_points = points.rows();
    const int64_t num_centers = centers.rows();
    DCHECK_LE(k, num_centers);
    DCHECK_GT(num_centers, kNearestNeighborsCentersMaxBlockSize);
    // Store nearest neighbors with first block of centers directly into the
    // output matrices.
    int64_t out_k = std::min(k, kNearestNeighborsCentersMaxBlockSize);
    FindKNearestCentersOneBlock(
        out_k, points, points_half_squared_norm,
        centers.topRows(kNearestNeighborsCentersMaxBlockSize),
        centers_half_squared_norm.head(kNearestNeighborsCentersMaxBlockSize),
        nearest_center_indices, nearest_center_distances);
    // Iteratively compute nearest neighbors with other blocks of centers, and
    // update the output matrices.
    MatrixXi64RowMajor block_nearest_center_indices(num_points, k);
    MatrixXfRowMajor block_nearest_center_distances(num_points, k);
    Eigen::Matrix<int64_t, 1, Eigen::Dynamic> merged_indices(k);
    Eigen::Matrix<float, 1, Eigen::Dynamic> merged_distances(k);
    for (int64_t centers_start = kNearestNeighborsCentersMaxBlockSize;
         centers_start < num_centers;
         centers_start += kNearestNeighborsCentersMaxBlockSize) {
      const int64_t centers_block_size = std::min(
          kNearestNeighborsCentersMaxBlockSize, num_centers - centers_start);
      const int64_t block_k = std::min(k, centers_block_size);
      FindKNearestCentersOneBlock(
          block_k, points, points_half_squared_norm,
          centers.middleRows(centers_start, centers_block_size),
          centers_half_squared_norm.segment(centers_start, centers_block_size),
          block_nearest_center_indices, block_nearest_center_distances);
      if (k == 1) {
        for (int i = 0; i < num_points; ++i) {
          if (block_nearest_center_distances(i, 0) <
              nearest_center_distances(i, 0)) {
            nearest_center_indices(i, 0) =
                block_nearest_center_indices(i, 0) + centers_start;
            nearest_center_distances(i, 0) =
                block_nearest_center_distances(i, 0);
          }
        }
      } else {
        for (int i = 0; i < num_points; ++i) {
          // Merge and accumulate top-k list from block_nearest_center_indices
          // into nearest_center_indices.
          for (int64_t j_out = 0, j_block = 0, j_merged = 0;
               (j_out < out_k || j_block < block_k) && j_merged < k;
               ++j_merged) {
            const float distance_out =
                j_out < out_k ? nearest_center_distances(i, j_out)
                              : std::numeric_limits<float>::infinity();
            const float distance_block =
                j_block < block_k ? block_nearest_center_distances(i, j_block)
                                  : std::numeric_limits<float>::infinity();
            if (distance_out <= distance_block) {
              merged_indices(j_merged) = nearest_center_indices(i, j_out);
              merged_distances(j_merged) = distance_out;
              ++j_out;
            } else {
              merged_indices(j_merged) =
                  block_nearest_center_indices(i, j_block) + centers_start;
              merged_distances(j_merged) = distance_block;
              ++j_block;
            }
          }
          nearest_center_indices.row(i) = merged_indices;
          nearest_center_distances.row(i) = merged_distances;
          out_k = std::min(k, out_k + block_k);
        }
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("NearestNeighbors").Device(DEVICE_CPU),
                        NearestNeighborsOp);

}  // namespace tensorflow
