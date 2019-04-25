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

#include "tensorflow/cc/training/checkpoint_management.h"

#include <vector>

#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/python/training/checkpoint_state.pb.h"

namespace tensorflow {

Status LatestCheckpointPath(const std::string& ckpt_dir,
                            const std::string& ckpt_state_filename,
                            std::string* latest_ckpt) {
  // Reading the CheckpointState proto containing the paths to the checkpoints.
  CheckpointState ckpt_state;
  const std::string ckpt_state_path =
      io::JoinPath(ckpt_dir, ckpt_state_filename);
  TF_RETURN_IF_ERROR(
      ReadTextProto(Env::Default(), ckpt_state_path, &ckpt_state));

  const std::string& ckpt_path = ckpt_state.model_checkpoint_path();
  if (ckpt_path.empty()) {
    return Status(
        error::Code::FAILED_PRECONDITION,
        strings::StrCat("CheckpointState contains no checkpoint paths."));
  }

  if (io::IsAbsolutePath(ckpt_path)) {
    *latest_ckpt = ckpt_path;
  } else {
    *latest_ckpt = io::JoinPath(ckpt_dir, ckpt_path);
  }

  // Looking for checkpoints stored with either V1 or V2 format.
  const std::string& ckpt_path_v1 = *latest_ckpt;
  const std::string ckpt_path_v2 = strings::StrCat(*latest_ckpt, ".index");
  std::vector<std::string> files_v1, files_v2;
  if (!Env::Default()->GetMatchingPaths(ckpt_path_v1, &files_v1).ok() ||
      !Env::Default()->GetMatchingPaths(ckpt_path_v2, &files_v2).ok() ||
      (files_v1.empty() && files_v2.empty())) {
    return Status(error::Code::FAILED_PRECONDITION,
                  strings::StrCat("No files matching ", ckpt_path_v1, " or ",
                                  ckpt_path_v2, "."));
  }

  return Status::OK();
}

}  // namespace tensorflow
