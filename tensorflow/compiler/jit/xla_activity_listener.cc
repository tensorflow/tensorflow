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

#include "tensorflow/compiler/jit/xla_activity_listener.h"

#include "absl/synchronization/mutex.h"
#include "tensorflow/compiler/jit/xla_activity.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {
namespace {
// The list of all registered `XlaActivityListener`s.
struct XlaActivityListenerList {
  absl::Mutex mutex;
  std::vector<std::unique_ptr<XlaActivityListener>> listeners
      TF_GUARDED_BY(mutex);
};

void FlushAllListeners();

XlaActivityListenerList* GetXlaActivityListenerList() {
  static XlaActivityListenerList* listener_list = new XlaActivityListenerList;
  static int unused = std::atexit(FlushAllListeners);
  (void)unused;
  return listener_list;
}

template <typename FnTy>
absl::Status ForEachListener(FnTy fn) {
  XlaActivityListenerList* listener_list = GetXlaActivityListenerList();
  absl::ReaderMutexLock reader_lock(listener_list->mutex);

  for (const std::unique_ptr<XlaActivityListener>& listener :
       listener_list->listeners) {
    TF_RETURN_IF_ERROR(fn(listener.get()));
  }

  return absl::OkStatus();
}

void FlushAllListeners() {
  absl::Status s = ForEachListener([](XlaActivityListener* listener) {
    listener->Flush();
    return absl::OkStatus();
  });
  CHECK(s.ok());
}
}  // namespace

absl::Status BroadcastXlaActivity(
    XlaAutoClusteringActivity auto_clustering_activity) {
  return ForEachListener([&](XlaActivityListener* listener) {
    return listener->Listen(auto_clustering_activity);
  });
}

absl::Status BroadcastXlaActivity(
    XlaJitCompilationActivity jit_compilation_activity) {
  return ForEachListener([&](XlaActivityListener* listener) {
    return listener->Listen(jit_compilation_activity);
  });
}

absl::Status BroadcastOptimizationRemark(
    XlaOptimizationRemark optimization_remark) {
  VLOG(2) << "OptimizationRemark: " << optimization_remark.DebugString();
  return ForEachListener([&](XlaActivityListener* listener) {
    return listener->Listen(optimization_remark);
  });
}

absl::Status BroadcastOptimizationRemark(
    XlaOptimizationRemark::Warning optimization_warning,
    string debug_information) {
  XlaOptimizationRemark remark;
  remark.set_warning(optimization_warning);
  remark.set_debug_information(std::move(debug_information));
  return BroadcastOptimizationRemark(std::move(remark));
}
void RegisterXlaActivityListener(
    std::unique_ptr<XlaActivityListener> listener) {
  XlaActivityListenerList* listener_list = GetXlaActivityListenerList();
  absl::WriterMutexLock writer_lock(listener_list->mutex);

  listener_list->listeners.push_back(std::move(listener));
}

void XlaActivityListener::Flush() {}

XlaActivityListener::~XlaActivityListener() {}

}  // namespace tensorflow
