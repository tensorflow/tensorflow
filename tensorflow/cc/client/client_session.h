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

#ifndef TENSORFLOW_CC_CLIENT_CLIENT_SESSION_H_
#define TENSORFLOW_CC_CLIENT_CLIENT_SESSION_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

namespace thread {

struct ThreadPoolOptions;

}

/// @addtogroup core
/// @{

/// A `ClientSession` object lets the caller drive the evaluation of the
/// TensorFlow graph constructed with the C++ API.
///
/// Example:
///
///     Scope root = Scope::NewRootScope();
///     auto a = Placeholder(root, DT_INT32);
///     auto c = Add(root, a, {41});
///
///     ClientSession session(root);
///     std::vector<Tensor> outputs;
///
///     Status s = session.Run({ {a, {1}} }, {c}, &outputs);
///     if (!s.ok()) { ... }
class ClientSession {
 public:
  /// A data type to represent feeds to a Run call.
  ///
  /// This is a map of `Output` objects returned by op-constructors to the value
  /// to feed them with. See `Input::Initializer` for details on what can be
  /// used as feed values.
  typedef std::unordered_map<Output, Input::Initializer, OutputHash> FeedType;

  /// Create a new session to evaluate the graph contained in `scope` by
  /// connecting to the TensorFlow runtime specified by `target`.
  ClientSession(const Scope& scope, const string& target);

  /// Same as above, but use the empty string ("") as the target specification.
  explicit ClientSession(const Scope& scope);

  /// Create a new session, configuring it with `session_options`.
  ClientSession(const Scope& scope, const SessionOptions& session_options);

  ~ClientSession();

  /// Evaluate the tensors in `fetch_outputs`. The values are returned as
  /// `Tensor` objects in `outputs`. The number and order of `outputs` will
  /// match `fetch_outputs`.
  Status Run(const std::vector<Output>& fetch_outputs,
             std::vector<Tensor>* outputs) const;

  /// Same as above, but use the mapping in `inputs` as feeds.
  Status Run(const FeedType& inputs, const std::vector<Output>& fetch_outputs,
             std::vector<Tensor>* outputs) const;

  /// Same as above. Additionally runs the operations ins `run_outputs`.
  Status Run(const FeedType& inputs, const std::vector<Output>& fetch_outputs,
             const std::vector<Operation>& run_outputs,
             std::vector<Tensor>* outputs) const;

  /// Use `run_options` to turn on performance profiling. `run_metadata`, if not
  /// null, is filled in with the profiling results.
  Status Run(const RunOptions& run_options, const FeedType& inputs,
             const std::vector<Output>& fetch_outputs,
             const std::vector<Operation>& run_outputs,
             std::vector<Tensor>* outputs, RunMetadata* run_metadata) const;

  /// Same as above. Additionally allows user to provide custom threadpool
  /// implementation via ThreadPoolOptions.
  Status Run(const RunOptions& run_options, const FeedType& inputs,
             const std::vector<Output>& fetch_outputs,
             const std::vector<Operation>& run_outputs,
             std::vector<Tensor>* outputs, RunMetadata* run_metadata,
             const thread::ThreadPoolOptions& threadpool_options) const;

  /// \brief A handle to a subgraph, created with
  /// `ClientSession::MakeCallable()`.
  typedef int64 CallableHandle;

  /// \brief Creates a `handle` for invoking the subgraph defined by
  /// `callable_options`.
  /// NOTE: This API is still experimental and may change.
  Status MakeCallable(const CallableOptions& callable_options,
                      CallableHandle* out_handle);

  /// \brief Invokes the subgraph named by `handle` with the given options and
  /// input tensors.
  ///
  /// The order of tensors in `feed_tensors` must match the order of names in
  /// `CallableOptions::feed()` and the order of tensors in `fetch_tensors` will
  /// match the order of names in `CallableOptions::fetch()` when this subgraph
  /// was created.
  /// NOTE: This API is still experimental and may change.
  Status RunCallable(CallableHandle handle,
                     const std::vector<Tensor>& feed_tensors,
                     std::vector<Tensor>* fetch_tensors,
                     RunMetadata* run_metadata);

  /// \brief Invokes the subgraph named by `handle` with the given options and
  /// input tensors.
  ///
  /// The order of tensors in `feed_tensors` must match the order of names in
  /// `CallableOptions::feed()` and the order of tensors in `fetch_tensors` will
  /// match the order of names in `CallableOptions::fetch()` when this subgraph
  /// was created.
  /// NOTE: This API is still experimental and may change.
  Status RunCallable(CallableHandle handle,
                     const std::vector<Tensor>& feed_tensors,
                     std::vector<Tensor>* fetch_tensors,
                     RunMetadata* run_metadata,
                     const thread::ThreadPoolOptions& options);

  /// \brief Releases resources associated with the given `handle` in this
  /// session.
  /// NOTE: This API is still experimental and may change.
  Status ReleaseCallable(CallableHandle handle);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
  Impl* impl() { return impl_.get(); }
  const Impl* impl() const { return impl_.get(); }
};

/// @}

}  // end namespace tensorflow

#endif  // TENSORFLOW_CC_CLIENT_CLIENT_SESSION_H_
