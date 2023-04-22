/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_GENERIC_LAYOUT_OPTIMIZER_TRANSPOSER_FACTORY_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_GENERIC_LAYOUT_OPTIMIZER_TRANSPOSER_FACTORY_H_

#include <memory>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.h"

namespace tensorflow {
namespace grappler {

class TransposerFactory {
 public:
  explicit TransposerFactory() {}

  std::shared_ptr<Transposer> GetTransposer(const NodeDef& node);

 protected:
  template <typename T>
  std::shared_ptr<Transposer> GetOrCreateIfNotFound(const string& key) {
    auto& transposer = transposer_map_[key];
    if (transposer == nullptr) {
      transposer = std::make_shared<T>();
    }
    return transposer;
  }

  absl::flat_hash_map<string, std::shared_ptr<Transposer>> transposer_map_;
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_GENERIC_LAYOUT_OPTIMIZER_TRANSPOSER_FACTORY_H_
