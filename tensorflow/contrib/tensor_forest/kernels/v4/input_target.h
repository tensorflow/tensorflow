// Copyright 2017 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_TENSOR_FOREST_KERNELS_V4_INPUT_TARGET_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_TENSOR_FOREST_KERNELS_V4_INPUT_TARGET_H_
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace tensorforest {

typedef TTypes<float, 1>::UnalignedConstTensor SingleDimStorageType;

// Base class for classes that hold labels and weights. Mostly for testing
// purposes, because it's inconvenient to construct nasty Eigen::things.
class InputTarget {
 public:
  virtual ~InputTarget() {}
  virtual int32 GetTargetAsClassIndex(int example_index,
                                      int target_index) const = 0;

  virtual float GetTargetWeight(int example_index) const = 0;

  virtual float GetTargetAsContinuous(int example_index,
                                      int target_index) const = 0;
};

template <typename T>
class StoredInputTarget : public InputTarget {
 protected:
  // Takes ownership of t and w with a std::unique_ptr.
  StoredInputTarget(const T* t, const T* w, int num_targets)
      : target_(t), weight_(w), num_targets_(num_targets) {}

  const std::unique_ptr<const T> target_;
  const std::unique_ptr<const T> weight_;
  int num_targets_;
};

// Holds labels/targets and weights. Assumes that tensors are passed as
// t.unaligned_flat<float>(). For multi-output, specifying the number of
// outputs will correctly index the flattened data.
class TensorInputTarget : public StoredInputTarget<SingleDimStorageType> {
 public:
  TensorInputTarget(const Tensor& target, const Tensor& weight, int num_targets)
      : StoredInputTarget(
            new SingleDimStorageType(target.unaligned_flat<float>()),
            new SingleDimStorageType(weight.unaligned_flat<float>()),
            num_targets),
        original_tensor_(target) {}

  int32 GetTargetAsClassIndex(int example_index,
                              int target_index) const override {
    return static_cast<int32>(
        GetTargetAsContinuous(example_index, target_index));
  }

  float GetTargetWeight(int example_index) const override {
    const size_t num_weights = weight_->size();
    return num_weights > 0 && example_index < num_weights
               ? (*weight_)(example_index)
               : 1.0;
  }

  float GetTargetAsContinuous(int example_index,
                              int target_index) const override {
    QCHECK_LT(target_index, num_targets_);
    return (*target_)(example_index * num_targets_ + target_index);
  }

  const Tensor& original_tensor() const {
    return original_tensor_;
  }

 protected:
  Tensor original_tensor_;
};
}  // namespace tensorforest
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_TENSOR_FOREST_KERNELS_V4_INPUT_TARGET_H_
