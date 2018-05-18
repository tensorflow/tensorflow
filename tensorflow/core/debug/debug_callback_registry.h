/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_DEBUG_CALLBACK_REGISTRY_H_
#define TENSORFLOW_DEBUG_CALLBACK_REGISTRY_H_

#include <functional>
#include <map>
#include <string>
#include <vector>

#include "tensorflow/core/debug/debug_node_key.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {

// Supports exporting observed debug events to clients using registered
// callbacks.  Users can register a callback for each debug_url stored using
// DebugTensorWatch.  The callback key be equivalent to what follows
// "memcbk:///".
//
// All events generated for a watched node will be sent to the call back in the
// order that they are observed.
//
// This callback router should not be used in production or training steps.  It
// is optimized for deep inspection of graph state rather than performance.
class DebugCallbackRegistry {
 public:
  using EventCallback = std::function<void(const DebugNodeKey&, const Tensor&)>;

  // Provides singleton access to the in memory event store.
  static DebugCallbackRegistry* singleton();

  // Returns the registered callback, or nullptr, for key.
  EventCallback* GetCallback(const string& key);

  // Associates callback with key.  This must be called by clients observing
  // nodes to be exported by this callback router before running a session.
  void RegisterCallback(const string& key, EventCallback callback);

  // Removes the callback associated with key.
  void UnregisterCallback(const string& key);

 private:
  DebugCallbackRegistry();

  // Mutex to ensure that keyed events are never updated in parallel.
  mutex mu_;

  // Maps debug_url keys to callbacks for routing observed tensors.
  std::map<string, EventCallback> keyed_callback_ GUARDED_BY(mu_);

  static DebugCallbackRegistry* instance_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_DEBUG_CALLBACK_REGISTRY_H_
