/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0(the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_CORE_KERNELS_CHECKPOINT_CALLBACK_MANAGER_H_
#define TENSORFLOW_CORE_KERNELS_CHECKPOINT_CALLBACK_MANAGER_H_

#include <functional>
#include <string>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/resource_base.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace checkpoint {

ABSL_CONST_INIT extern const absl::string_view
    kCheckpointCallbackManagerResourceName;

// StatusOr<std::string> save_callback(absl::string_view checkpoint_id);
using SaveCallback = std::function<StatusOr<std::string>(absl::string_view)>;

// Status restore_callback(absl::string_view checkpoint_id,
//                         absl::string_view content_from_checkpoint);
using RestoreCallback =
    std::function<Status(absl::string_view, absl::string_view)>;

// A class to save and restore additional information for checkpointing.
class CheckpointCallbackManager : public ResourceBase {
 public:
  CheckpointCallbackManager() = default;

  // Not copyable or movable
  CheckpointCallbackManager(const CheckpointCallbackManager&) = delete;
  CheckpointCallbackManager& operator=(const CheckpointCallbackManager&) =
      delete;

  std::string DebugString() const override {
    return "CheckpointCallbackManager";
  }

  // Infers a checkpoint id and directory from a prefix
  // passed to SaveV2 / RestoreV2 Ops
  static StatusOr<std::pair<std::string, std::string>>
  GetCheckpointIdAndPathFromPrefix(absl::string_view prefix);

  // Register a save callback.
  // The passed callback will be triggered with an identified checkpoint id.
  // The callback should return a string content needs to be stored
  // as a part of a checkpoint, and then the content is stored as a file
  // with the registered the file_extension.
  Status RegisterSaveCallback(absl::string_view file_extension,
                              SaveCallback callback);

  // Checks if a registered save callback exists for an extension.
  bool DoesSaveCallbackExist(absl::string_view file_extension);

  // Register a restore callback.
  // The passed file_extension is used to generate a file name together with
  // an identified checkpoint_id. If the file exists, the registered callback
  // is triggered with the content of the file.
  Status RegisterRestoreCallback(absl::string_view file_extension,
                                 RestoreCallback callback);

  // Checks if a registered restore callback exists for an extension.
  bool DoesRestoreCallbackExist(absl::string_view file_extension);

  // Should be triggered from SaveV2()::Compute().
  void Save(absl::string_view prefix);

  // Should be triggered from RestoreV2()::Compute().
  void Restore(absl::string_view prefix);

 private:
  mutable mutex mu_;

  absl::flat_hash_map<std::string, SaveCallback> save_callbacks_
      TF_GUARDED_BY(mu_);
  absl::flat_hash_map<std::string, RestoreCallback> restore_callbacks_
      TF_GUARDED_BY(mu_);

  // Checkpoint save and restore could happen before save / restore callbacks
  // are registered. The last checkpoint information is kept in these variables
  // to trigger the registered callback lazily.
  std::pair<std::string, std::string> last_restored_checkpoint_id_and_dir_
      TF_GUARDED_BY(mu_);

  std::pair<std::string, std::string> last_saved_checkpoint_id_and_dir_
      TF_GUARDED_BY(mu_);
};

}  // namespace checkpoint
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_CHECKPOINT_CALLBACK_MANAGER_H_
