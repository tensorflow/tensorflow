/* Copyright 2015 Google Inc. All Rights Reserved.

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

// An abstraction to pick from one of N elements with a specified
// weight per element.
//
// The weight for a given element can be changed in O(lg N) time
// An element can be picked in O(lg N) time.
//
// Uses O(N) bytes of memory.
//
// Alternative: distribution-sampler.h allows O(1) time picking, but no weight
// adjustment after construction.

#ifndef TENSORFLOW_LIB_RANDOM_WEIGHTED_PICKER_H_
#define TENSORFLOW_LIB_RANDOM_WEIGHTED_PICKER_H_

#include <assert.h>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace random {

class SimplePhilox;

class WeightedPicker {
 public:
  // REQUIRES   N >= 0
  // Initializes the elements with a weight of one per element
  explicit WeightedPicker(int N);

  // Releases all resources
  ~WeightedPicker();

  // Pick a random element with probability proportional to its weight.
  // If total weight is zero, returns -1.
  int Pick(SimplePhilox* rnd) const;

  // Deterministically pick element x whose weight covers the
  // specified weight_index.
  // Returns -1 if weight_index is not in the range [ 0 .. total_weight()-1 ]
  int PickAt(int32 weight_index) const;

  // Get the weight associated with an element
  // REQUIRES 0 <= index < N
  int32 get_weight(int index) const;

  // Set the weight associated with an element
  // REQUIRES weight >= 0.0f
  // REQUIRES 0 <= index < N
  void set_weight(int index, int32 weight);

  // Get the total combined weight of all elements
  int32 total_weight() const;

  // Get the number of elements in the picker
  int num_elements() const;

  // Set weight of each element to "weight"
  void SetAllWeights(int32 weight);

  // Resizes the picker to N and
  // sets the weight of each element i to weight[i].
  // The sum of the weights should not exceed 2^31 - 2
  // Complexity O(N).
  void SetWeightsFromArray(int N, const int32* weights);

  // REQUIRES   N >= 0
  //
  // Resize the weighted picker so that it has "N" elements.
  // Any newly added entries have zero weight.
  //
  // Note: Resizing to a smaller size than num_elements() will
  // not reclaim any memory.  If you wish to reduce memory usage,
  // allocate a new WeightedPicker of the appropriate size.
  //
  // It is efficient to use repeated calls to Resize(num_elements() + 1)
  // to grow the picker to size X (takes total time O(X)).
  void Resize(int N);

  // Grow the picker by one and set the weight of the new entry to "weight".
  //
  // Repeated calls to Append() in order to grow the
  // picker to size X takes a total time of O(X lg(X)).
  // Consider using SetWeightsFromArray instead.
  void Append(int32 weight);

 private:
  // We keep a binary tree with N leaves.  The "i"th leaf contains
  // the weight of the "i"th element.  An internal node contains
  // the sum of the weights of its children.
  int N_;           // Number of elements
  int num_levels_;  // Number of levels in tree (level-0 is root)
  int32** level_;   // Array that holds nodes per level

  // Size of each level
  static int LevelSize(int level) { return 1 << level; }

  // Rebuild the tree weights using the leaf weights
  void RebuildTreeWeights();

  TF_DISALLOW_COPY_AND_ASSIGN(WeightedPicker);
};

inline int32 WeightedPicker::get_weight(int index) const {
  DCHECK_GE(index, 0);
  DCHECK_LT(index, N_);
  return level_[num_levels_ - 1][index];
}

inline int32 WeightedPicker::total_weight() const { return level_[0][0]; }

inline int WeightedPicker::num_elements() const { return N_; }

}  // namespace random
}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_RANDOM_WEIGHTED_PICKER_H_
