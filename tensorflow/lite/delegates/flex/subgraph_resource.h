/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_DELEGATES_FLEX_SUBGRAPH_RESOURCE_H_
#define TENSORFLOW_LITE_DELEGATES_FLEX_SUBGRAPH_RESOURCE_H_

#include <memory>
#include <string>

#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/subgraph.h"

namespace tflite {
namespace flex {

// This object stores a pointer for a TfLite subgraph and the associated mutex
// to access the subgraph. Before accessing the TF Lite subgraph, the caller
// needs to first acquire a lock on the mutex object.
class TFLiteSubgraphResource : public tensorflow::ResourceBase {
 public:
  explicit TFLiteSubgraphResource(Subgraph& subgraph, TfLiteDelegate* delegate)
      : subgraph_(subgraph), delegate_(delegate) {}

  // This class is movable but not copyable.
  TFLiteSubgraphResource(TFLiteSubgraphResource&&) = default;
  TFLiteSubgraphResource& operator=(TFLiteSubgraphResource&&) = default;

  TFLiteSubgraphResource(const TFLiteSubgraphResource&) = delete;
  TFLiteSubgraphResource& operator=(TFLiteSubgraphResource&) = delete;

  std::string DebugString() const override { return "TFLiteSubgraphResource"; }

  // Returns the TFLite subgraph. Before calling
  // this method, the caller needs to acquire the underlying mutex lock.
  Subgraph& GetSubgraphResource() const TF_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    return subgraph_;
  }

  tensorflow::mutex& GetExclusiveLock() TF_LOCK_RETURNED(mutex_) {
    return mutex_;
  }

  // Returns a pointer to the TfLiteDelegate which this instance of subgraph
  // is running as part of it.
  TfLiteDelegate* GetFlexDelegate() TF_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    return delegate_;
  }

 private:
  tensorflow::mutex mutex_;
  Subgraph& subgraph_ TF_GUARDED_BY(mutex_);
  TfLiteDelegate* delegate_ TF_GUARDED_BY(mutex_) = nullptr;
};

}  // namespace flex
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_FLEX_SUBGRAPH_RESOURCE_H_
