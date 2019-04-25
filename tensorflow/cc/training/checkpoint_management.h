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

#ifndef TENSORFLOW_CC_TRAINING_CHECKPOINT_MANAGEMENT_H_
#define TENSORFLOW_CC_TRAINING_CHECKPOINT_MANAGEMENT_H_

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/python/training/checkpoint_state.pb.h"

namespace tensorflow {

// Gets the path to the latest checkpoint saved to ckpt_dir, with the
// CheckpointState filename given by ckpt_state_filename.
Status LatestCheckpointPath(const std::string& ckpt_dir,
                            const std::string& ckpt_state_filename,
                            std::string* latest_ckpt);

// Overloads LatestCheckpointPath in order to make ckpt_state_filename an
// optional argument.
Status LatestCheckpointPath(const std::string& ckpt_dir,
                            std::string* latest_ckpt) {
  return LatestCheckpointPath(ckpt_dir, "checkpoint", latest_ckpt);
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CC_TRAINING_CHECKPOINT_MANAGEMENT_H_
