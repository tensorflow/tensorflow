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

// The file input provides a mechanism to feed grappler with existing TensorFlow
// graphs stored in TensorFlow checkpoints. Note that at this point the weights
// that may be stored in the checkpoint are not restored in order to speedup the
// initialization.

#ifndef LEARNING_BRAIN_EXPERIMENTAL_GRAPPLER_INPUTS_FILE_INPUT_YIELDER_H_
#define LEARNING_BRAIN_EXPERIMENTAL_GRAPPLER_INPUTS_FILE_INPUT_YIELDER_H_

#include <stddef.h>
#include <limits>
#include <vector>
#include "tensorflow/core/grappler/inputs/input_yielder.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace grappler {

class GrapplerItem;

class FileInputYielder : public InputYielder {
 public:
  // Iterates over the files specified in the list of 'filename' up to
  // 'max_iterations' times.
  explicit FileInputYielder(
      const std::vector<string>& filenames,
      size_t max_iterations = std::numeric_limits<size_t>::max());
  bool NextItem(GrapplerItem* item) override;

 private:
  const std::vector<string> filenames_;
  size_t current_file_;
  size_t current_iteration_;
  size_t max_iterations_;

  size_t bad_inputs_;
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // LEARNING_BRAIN_EXPERIMENTAL_GRAPPLER_INPUTS_FILE_INPUT_YIELDER_H_
