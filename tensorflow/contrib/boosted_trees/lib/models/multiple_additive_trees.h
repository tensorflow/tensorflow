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
#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_MODELS_MULTIPLE_ADDITIVE_TREES_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_MODELS_MULTIPLE_ADDITIVE_TREES_H_

#include <vector>

#include "tensorflow/contrib/boosted_trees/lib/utils/batch_features.h"
#include "tensorflow/contrib/boosted_trees/proto/tree_config.pb.h"  // NOLINT
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace boosted_trees {
namespace models {

// Multiple additive trees prediction model.
// This class does not hold state and is thread safe.
class MultipleAdditiveTrees {
 public:
  // Predict runs tree ensemble on the given batch and updates
  // output predictions accordingly, for the given list of trees.
  static void Predict(
      const boosted_trees::trees::DecisionTreeEnsembleConfig& config,
      const std::vector<int32>& trees_to_include,
      const boosted_trees::utils::BatchFeatures& features,
      tensorflow::thread::ThreadPool* const worker_threads,
      tensorflow::TTypes<float>::Matrix output_predictions);
};

}  // namespace models
}  // namespace boosted_trees
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_MODELS_MULTIPLE_ADDITIVE_TREES_H_
