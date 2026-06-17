/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_DELEGATES_FLEX_DELEGATE_DATA_H_
#define TENSORFLOW_LITE_DELEGATES_FLEX_DELEGATE_DATA_H_

#include <functional>
#include <string>

#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/delegates/flex/buffer_map.h"
#include "tensorflow/lite/delegates/flex/subgraph_resource.h"

namespace tflite {
namespace flex {

// Data kept by the Flex delegate for the lifetime of an Interpreter.
//
// Note: This class is *not* thread-safe; any dependent delegates should not be
// used concurrently.
class DelegateData {
 public:
  DelegateData();
  ~DelegateData();

  // Prepare the necessary EagerContext and data for execution.
  // This must be called at least once before execution. After preparation
  // succeeds, redundant calls will be ignored (even if the session_options
  // differ).
  // When `main_subgraph` parameter is provided, this function will register
  // FunctionDefs associated with each of the subgraphs attached to the
  // `main_subgraph` which is delegated by 'flex_delegate'.
  // 'flex_delegate' should always be non-null when 'main_subgraph' is
  // non-null.
  absl::Status Prepare(const tensorflow::SessionOptions& session_options,
                       Subgraph* main_subgraph = nullptr,
                       TfLiteDelegate* flex_delegate = nullptr);

  // The EagerContext that is required for execution of Flex Ops.
  // Note: The context is lazily created after the first call to |Prepare()|.
  tensorflow::EagerContext* GetEagerContext() { return eager_context_; }

  tensorflow::CancellationManager* GetCancellationManager() {
    return cancellation_manager_;
  }

  void SetCancellationManager(
      tensorflow::CancellationManager* cancellation_manager) {
    cancellation_manager_ = cancellation_manager;
  }

  // Map from TF Lite tensor index to TensorFlow tensor for a given context.
  BufferMap* GetBufferMap(const TfLiteContext* context) {
    return &buffer_map_[context];
  }

  // Returns the mapping between tensor index and last node index for a given
  // context.
  std::map<int, int>* GetTensorReleaseMap(const TfLiteContext* context) {
    return &tensor_release_map_[context];
  }

 private:
  // Will be null until Prepare() is called and completes successfully.
  tensorflow::EagerContext* eager_context_ = nullptr;
  // Not owned by DelegateData.
  tensorflow::CancellationManager* cancellation_manager_ = nullptr;
  // TODO(b/112439500): Clean up stale BufferMap instances after adding the
  // necessary cleanup hook from a TfLiteContext to a TfLiteDelegate.
  std::unordered_map<const TfLiteContext*, BufferMap> buffer_map_;
  // Maps between context and the tensor release map. The map will be filled
  // during delegate initialization, and queried during eval to look up tensor
  // lifetime information.
  std::unordered_map<const TfLiteContext*, std::map<int, int>>
      tensor_release_map_;
};

// Creates a `TFLiteSubgraphResource` for each subgraph (execpt
// for main subgraph) in the model and adds it in the eager context's resource
// manager. It also registers FunctionDefs in the function library runtime for
// subgraphs which are used by a list of flex ops.
absl::Status RegisterFunctionDefForSubgraphs(
    Subgraph& main_subgraph,
    const std::function<absl::Status(
        const std::vector<std::unique_ptr<Subgraph>>&,
        std::set<std::string>* result)>& select_subgraphs_to_register,
    tensorflow::ResourceMgr* resource_mgr,
    tensorflow::EagerContext* eager_context, TfLiteDelegate* flex_delegate);

}  // namespace flex
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_FLEX_DELEGATE_DATA_H_
