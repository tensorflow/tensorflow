/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_NEAREST_NEIGHBOR_KERNELS_HYPERPLANE_LSH_PROBES_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_NEAREST_NEIGHBOR_KERNELS_HYPERPLANE_LSH_PROBES_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/contrib/nearest_neighbor/kernels/heap.h"

namespace tensorflow {
namespace nearest_neighbor {

// This class implements hyperplane multiprobe LSH as described in the
// following paper:
//
//   Multi-probe LSH: efficient indexing for high-dimensional similarity search
//   Qin Lv, William Josephson, Zhe Wang, Moses Charikar, Kai Li
//
// The class is only responsible for generating the probing sequence of given
// length for a given batch of points. The actual hash table lookups are
// implemented in other classes.
template <typename CoordinateType, typename HashType>
class HyperplaneMultiprobe {
 public:
  using Matrix = Eigen::Matrix<CoordinateType, Eigen::Dynamic, Eigen::Dynamic,
                               Eigen::RowMajor>;
  using ConstMatrixMap = Eigen::Map<const Matrix>;
  using MatrixMap = Eigen::Map<Matrix>;
  using Vector =
      Eigen::Matrix<CoordinateType, Eigen::Dynamic, 1, Eigen::ColMajor>;

  HyperplaneMultiprobe(int num_hyperplanes_per_table, int num_tables)
      : num_hyperplanes_per_table_(num_hyperplanes_per_table),
        num_tables_(num_tables),
        num_probes_(0),
        cur_probe_counter_(0),
        sorted_hyperplane_indices_(0),
        main_table_probe_(num_tables) {}

  // The first input hash_vector is the matrix-vector product between the
  // hyperplane matrix and the vector for which we want to generate a probing
  // sequence. We assume that each index in hash_vector is proportional to the
  // distance between vector and hyperplane (i.e., the hyperplane vectors should
  // all have the same norm).
  //
  // The second input is the number of probes we want to retrieve. If this
  // number is fixed in advance, it should be passed in here in order to enable
  // some (minor) internal optimizations. If the number of probes it not known
  // in advance, the multiprobe sequence can still produce an arbitrary length
  // probing sequence (up to the maximum number of probes) by calling
  // get_next_probe multiple times.
  //
  // If num_probes is at most num_tables, it is not necessary to generate an
  // actual multiprobe sequence and the multiprobe object will simply return
  // the "standard" LSH probes without incurring any multiprobe overhead.
  void SetupProbing(const Vector& hash_vector, int_fast64_t num_probes) {
    // We accept a copy here for now.
    hash_vector_ = hash_vector;
    num_probes_ = num_probes;
    cur_probe_counter_ = -1;

    // Compute the initial probes for each table, i.e., the "true" hash
    // locations LSH without multiprobe would give.
    for (int_fast32_t ii = 0; ii < num_tables_; ++ii) {
      main_table_probe_[ii] = 0;
      for (int_fast32_t jj = 0; jj < num_hyperplanes_per_table_; ++jj) {
        main_table_probe_[ii] = main_table_probe_[ii] << 1;
        main_table_probe_[ii] =
            main_table_probe_[ii] |
            (hash_vector_[ii * num_hyperplanes_per_table_ + jj] >= 0.0);
      }
    }

    if (num_probes_ >= 0 && num_probes_ <= num_tables_) {
      return;
    }

    if (sorted_hyperplane_indices_.size() == 0) {
      sorted_hyperplane_indices_.resize(num_tables_);
      for (int_fast32_t ii = 0; ii < num_tables_; ++ii) {
        sorted_hyperplane_indices_[ii].resize(num_hyperplanes_per_table_);
        for (int_fast32_t jj = 0; jj < num_hyperplanes_per_table_; ++jj) {
          sorted_hyperplane_indices_[ii][jj] = jj;
        }
      }
    }

    for (int_fast32_t ii = 0; ii < num_tables_; ++ii) {
      HyperplaneComparator comp(hash_vector_, ii * num_hyperplanes_per_table_);
      std::sort(sorted_hyperplane_indices_[ii].begin(),
                sorted_hyperplane_indices_[ii].end(), comp);
    }

    if (num_probes_ >= 0) {
      heap_.Resize(2 * num_probes_);
    }
    heap_.Reset();
    for (int_fast32_t ii = 0; ii < num_tables_; ++ii) {
      int_fast32_t best_index = sorted_hyperplane_indices_[ii][0];
      CoordinateType score =
          hash_vector_[ii * num_hyperplanes_per_table_ + best_index];
      score = score * score;
      HashType hash_mask = 1;
      hash_mask = hash_mask << (num_hyperplanes_per_table_ - best_index - 1);
      heap_.InsertUnsorted(score, ProbeCandidate(ii, hash_mask, 0));
    }
    heap_.Heapify();
  }

  // This method stores the current probe (= hash table location) and
  // corresponding table in the output parameters. The return value indicates
  // whether this succeeded (true) or the current probing sequence is exhausted
  // (false). Here, we say a probing sequence is exhausted if one of the
  // following two conditions occurs:
  // - We have used a non-negative value for num_probes in setup_probing, and
  //   we have produced this many number of probes in the current sequence.
  // - We have used a negative value for num_probes in setup_probing, and we
  //   have produced all possible probes in the probing sequence.
  bool GetNextProbe(HashType* cur_probe, int_fast32_t* cur_table) {
    cur_probe_counter_ += 1;

    if (num_probes_ >= 0 && cur_probe_counter_ >= num_probes_) {
      // We are out of probes in the current probing sequence.
      return false;
    }

    // For the first num_tables_ probes, we directly return the "standard LSH"
    // probes to guarantee that they always come first and we avoid any
    // multiprobe overhead.
    if (cur_probe_counter_ < num_tables_) {
      *cur_probe = main_table_probe_[cur_probe_counter_];
      *cur_table = cur_probe_counter_;
      return true;
    }

    // If the heap is empty, the current probing sequence is exhausted.
    if (heap_.IsEmpty()) {
      return false;
    }

    CoordinateType cur_score;
    ProbeCandidate cur_candidate;
    heap_.ExtractMin(&cur_score, &cur_candidate);
    *cur_table = cur_candidate.table_;
    int_fast32_t cur_index =
        sorted_hyperplane_indices_[*cur_table][cur_candidate.last_index_];
    *cur_probe = main_table_probe_[*cur_table] ^ cur_candidate.hash_mask_;

    if (cur_candidate.last_index_ != num_hyperplanes_per_table_ - 1) {
      // swapping out the last flipped index
      int_fast32_t next_index =
          sorted_hyperplane_indices_[*cur_table][cur_candidate.last_index_ + 1];

      // xor out previous bit, xor in new bit.
      HashType next_mask =
          cur_candidate.hash_mask_ ^
          (HashType(1) << (num_hyperplanes_per_table_ - cur_index - 1)) ^
          (HashType(1) << (num_hyperplanes_per_table_ - next_index - 1));

      CoordinateType cur_coord =
          hash_vector_[*cur_table * num_hyperplanes_per_table_ + cur_index];
      CoordinateType next_coord =
          hash_vector_[*cur_table * num_hyperplanes_per_table_ + next_index];
      CoordinateType next_score =
          cur_score - cur_coord * cur_coord + next_coord * next_coord;

      heap_.Insert(next_score, ProbeCandidate(*cur_table, next_mask,
                                              cur_candidate.last_index_ + 1));

      // adding a new flipped index
      next_mask =
          cur_candidate.hash_mask_ ^
          (HashType(1) << (num_hyperplanes_per_table_ - next_index - 1));
      next_score = cur_score + next_coord * next_coord;

      heap_.Insert(next_score, ProbeCandidate(*cur_table, next_mask,
                                              cur_candidate.last_index_ + 1));
    }

    return true;
  }

 private:
  class ProbeCandidate {
   public:
    ProbeCandidate(int_fast32_t table = 0, HashType hash_mask = 0,
                   int_fast32_t last_index = 0)
        : table_(table), hash_mask_(hash_mask), last_index_(last_index) {}

    int_fast32_t table_;
    HashType hash_mask_;
    int_fast32_t last_index_;
  };

  class HyperplaneComparator {
   public:
    HyperplaneComparator(const Vector& values, int_fast32_t offset)
        : values_(values), offset_(offset) {}

    bool operator()(int_fast32_t ii, int_fast32_t jj) const {
      return std::abs(values_[offset_ + ii]) < std::abs(values_[offset_ + jj]);
    }

   private:
    const Vector& values_;
    int_fast32_t offset_;
  };

  int_fast32_t num_hyperplanes_per_table_;
  int_fast32_t num_tables_;
  int_fast64_t num_probes_;
  int_fast64_t cur_probe_counter_;
  std::vector<std::vector<int_fast32_t>> sorted_hyperplane_indices_;
  std::vector<HashType> main_table_probe_;
  SimpleHeap<CoordinateType, ProbeCandidate> heap_;
  Vector hash_vector_;
};

}  // namespace nearest_neighbor
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_NEAREST_NEIGHBOR_KERNELS_HYPERPLANE_LSH_PROBES_H_
