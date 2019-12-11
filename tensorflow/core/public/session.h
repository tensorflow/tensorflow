/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PUBLIC_SESSION_H_
#define TENSORFLOW_CORE_PUBLIC_SESSION_H_

#include <string>
#include <vector>

#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
class DeviceMgr;

namespace thread {

struct ThreadPoolOptions;

}

/// \brief A Session instance lets a caller drive a TensorFlow graph
/// computation.
///
/// When a Session is created with a given target, a new Session object
/// is bound to the universe of resources specified by that target.
/// Those resources are available to this session to perform
/// computation described in the GraphDef.  After extending the session
/// with a graph, the caller uses the Run() API to perform the
/// computation and potentially fetch outputs as Tensors.
///
/// Example:
///
/// ```c++
///
///     tensorflow::GraphDef graph;
///     // ... Create or load graph into "graph".
///
///     // This example uses the default options which connects
///     // to a local runtime.
///     tensorflow::SessionOptions options;
///     std::unique_ptr<tensorflow::Session>
///     session(tensorflow::NewSession(options));
///
///     // Create the session with this graph.
///     tensorflow::Status s = session->Create(graph);
///     if (!s.ok()) { ... }
///
///     // Run the graph and fetch the first output of the "output"
///     // operation, and also run to but do not return anything
///     // for the "update_state" operation.
///     std::vector<tensorflow::Tensor> outputs;
///     s = session->Run({}, {"output:0"}, {"update_state"}, &outputs);
///     if (!s.ok()) { ... }
///
///     // Map the output as a flattened float tensor, and do something
///     // with it.
///     auto output_tensor = outputs[0].flat<float>();
///     if (output_tensor(0) > 0.5) { ... }
///
///     // Close the session to release the resources associated with
///     // this session.
///     session->Close();
///
/// ```
///
/// A Session allows concurrent calls to Run(), though a Session must
/// be created / extended by a single thread.
///
/// Only one thread must call Close(), and Close() must only be called
/// after all other calls to Run() have returned.
class Session {
 public:
  Session();
  virtual ~Session();

  /// \brief Create the graph to be used for the session.
  ///
  /// Returns an error if this session has already been created with a
  /// graph. To re-use the session with a different graph, the caller
  /// must Close() the session first.
  virtual Status Create(const GraphDef& graph) = 0;
#ifndef SWIG
  virtual Status Create(GraphDef&& graph) { return Create(graph); }
#endif

  /// \brief Adds operations to the graph that is already registered with the
  /// Session.
  ///
  /// The names of new operations in "graph" must not exist in the
  /// graph that is already registered.
  virtual Status Extend(const GraphDef& graph) = 0;
#ifndef SWIG
  virtual Status Extend(GraphDef&& graph) { return Extend(graph); }
#endif

  /// \brief Runs the graph with the provided input tensors and fills
  /// `outputs` for the endpoints specified in `output_tensor_names`.
  /// Runs to but does not return Tensors for the nodes in
  /// `target_node_names`.
  ///
  /// The order of tensors in `outputs` will match the order provided
  /// by `output_tensor_names`.
  ///
  /// If `Run` returns `OK()`, then `outputs->size()` will be equal to
  /// `output_tensor_names.size()`.  If `Run` does not return `OK()`, the
  /// state of `outputs` is undefined.
  ///
  /// REQUIRES: The name of each Tensor of the input or output must
  /// match a "Tensor endpoint" in the `GraphDef` passed to `Create()`.
  ///
  /// REQUIRES: At least one of `output_tensor_names` and
  /// `target_node_names` must be non-empty.
  ///
  /// REQUIRES: outputs is not nullptr if `output_tensor_names` is non-empty.
  virtual Status Run(const std::vector<std::pair<string, Tensor> >& inputs,
                     const std::vector<string>& output_tensor_names,
                     const std::vector<string>& target_node_names,
                     std::vector<Tensor>* outputs) = 0;

  /// \brief Implementations which support `RunOptions`.
  //
  /// NOTE: This API is still experimental and may change.
  virtual Status Create(const RunOptions& run_options, const GraphDef& graph) {
    return errors::Unimplemented(
        "Create(const RunOptions& run_options, const GraphDef& graph) is not "
        "supported for this session.");
  }
  virtual Status Extend(const RunOptions& run_options, const GraphDef& graph) {
    return errors::Unimplemented(
        "Extend(const RunOptions& run_options, const GraphDef& graph) is not "
        "supported for this session.");
  }
#ifndef SWIG
  virtual Status Create(const RunOptions& run_options, GraphDef&& graph) {
    return Create(run_options, graph);
  }
  virtual Status Extend(const RunOptions& run_options, GraphDef&& graph) {
    return Extend(run_options, graph);
  }
#endif
  virtual Status Close(const RunOptions& run_options) {
    return errors::Unimplemented(
        "Close(const RunOptions& run_options) is not supported for this "
        "session.");
  }

  /// \brief Like `Run`, but allows users to pass in a `RunOptions` proto and
  /// to retrieve non-Tensor metadata output via a `RunMetadata` proto for this
  /// step.  `run_metadata` may be nullptr, in which case any metadata output is
  /// discarded.
  /// NOTE: This API is still experimental and may change.
  virtual Status Run(const RunOptions& run_options,
                     const std::vector<std::pair<string, Tensor> >& inputs,
                     const std::vector<string>& output_tensor_names,
                     const std::vector<string>& target_node_names,
                     std::vector<Tensor>* outputs, RunMetadata* run_metadata);

  /// \brief Sets up a graph for partial execution. All future feeds and
  /// fetches are specified by `input_names` and `output_names`. Returns
  /// `handle` that can be used to perform a sequence of partial feeds and
  /// fetches.
  /// NOTE: This API is still experimental and may change.
  virtual Status PRunSetup(const std::vector<string>& input_names,
                           const std::vector<string>& output_names,
                           const std::vector<string>& target_nodes,
                           string* handle);

  /// \brief Continues the pending execution specified by `handle` with the
  /// provided input tensors and fills `outputs` for the endpoints specified
  /// in `output_names`.
  /// NOTE: This API is still experimental and may change.
  virtual Status PRun(const string& handle,
                      const std::vector<std::pair<string, Tensor> >& inputs,
                      const std::vector<string>& output_names,
                      std::vector<Tensor>* outputs);

  /// \brief List devices in the session.
  ///
  /// Retrieves the list of available devices within the session, and populates
  /// *response. This API is optional. If it is unimplemented, Status will
  /// return a corresponding error message, and *response will be unmodified.
  virtual Status ListDevices(std::vector<DeviceAttributes>* response) = 0;

  /// \brief Closes this session.
  ///
  /// Closing a session releases the resources used by this session
  /// on the TensorFlow runtime (specified during session creation by
  /// the `SessionOptions::target` field).
  virtual Status Close() = 0;

  // NOTE(ashankar): As of July 2017, this method was added to facilitate some
  // experimentation. Reconsider/re-evaluate after September 2017.
  //
  // Sets `*output` to the `DeviceMgr` that owns accessible devices in the
  // address-space of the caller.
  virtual Status LocalDeviceManager(const DeviceMgr** output) {
    return errors::Unimplemented(
        "LocalDeviceManager is not supported for this session.");
  }

  /// \brief A handle to a subgraph, created with `Session::MakeCallable()`.
  typedef int64 CallableHandle;

  /// \brief Creates a `handle` for invoking the subgraph defined by
  /// `callable_options`.
  /// NOTE: This API is still experimental and may change.
  virtual Status MakeCallable(const CallableOptions& callable_options,
                              CallableHandle* out_handle) {
    return errors::Unimplemented(
        "MakeCallable is not supported for this session.");
  }

  /// \brief Invokes the subgraph named by `handle` with the given options and
  /// input tensors.
  ///
  /// The order of tensors in `feed_tensors` must and `fetch_tensors` will
  /// match the order of names in `CallableOptions::feed()` and
  /// `CallableOptions::fetch()` when this subgraph was created.
  /// NOTE: This API is still experimental and may change.
  virtual Status RunCallable(CallableHandle handle,
                             const std::vector<Tensor>& feed_tensors,
                             std::vector<Tensor>* fetch_tensors,
                             RunMetadata* run_metadata) {
    return errors::Unimplemented(
        "RunCallable is not supported for this session.");
  }

  /// \brief Invokes the subgraph named by `handle` with the given options and
  /// input tensors.
  ///
  /// The order of tensors in `feed_tensors` must and `fetch_tensors` will
  /// match the order of names in `CallableOptions::feed()` and
  /// `CallableOptions::fetch()` when this subgraph was created.
  /// NOTE: This API is still experimental and may change.
  virtual Status RunCallable(
      CallableHandle handle, const std::vector<Tensor>& feed_tensors,
      std::vector<Tensor>* fetch_tensors, RunMetadata* run_metadata,
      const thread::ThreadPoolOptions& threadpool_options) {
    return errors::Unimplemented(
        "RunCallable with threadpool is not supported for this session.");
  }

  /// \brief Releases resources associated with the given `handle` in this
  /// session.
  /// NOTE: This API is still experimental and may change.
  virtual Status ReleaseCallable(CallableHandle handle) {
    return errors::Unimplemented(
        "ReleaseCallable is not supported for this session.");
  }

  /// \brief Release global graph-related state in this session.
  ///
  /// After calling `this->Finalize()`, calls to `this->Run()` with previously
  /// unseen feeds and fetches, and calls to `this->MakeCallable()` will fail.
  /// Using `MakeCallable()` and `RunCallable()` is recommended, because
  /// explicit callable creation makes it clearer where the `Finalize()` call
  /// should be placed.
  ///
  /// This API can be used in conjunction with a "warmup" phase to reduce the
  /// memory consumed by the session:
  ///
  /// 1. Call `Session::Create()`.
  /// 2. Call `Session::MakeCallable()` for all subgraphs that you will execute
  ///    in the session.
  /// 3. Call `Session::Finalize()` to release global graph-related state.
  /// 4. Call `Session::RunCallable()` with the handle(s) created in step 2.
  ///
  /// NOTE: This API is still experimental and may change.
  virtual Status Finalize() {
    return errors::Unimplemented("Finalize is not supported for this session.");
  }
};

/// \brief Create a new session with the given options.
///
/// If session creation succeeds, the new `Session` will be stored in
/// `*out_session`, the caller will take ownership of the returned
/// `*out_session`, and this function will return `OK()`. Otherwise, this
/// function will return an error status and set *out_session to nullptr.
Status NewSession(const SessionOptions& options, Session** out_session);

/// \brief Resets resource containers associated with a target.
///
/// Reset() allows misbehaving or slow sessions to be aborted and closed, and
/// causes their resources eventually to be released.  Reset() does not wait
/// for the computations in old sessions to cease; it merely starts the
/// process of tearing them down.  However, if a new session is started after
/// a Reset(), the new session is isolated from changes that old sessions
/// (started prior to the Reset()) may continue to make to resources, provided
/// all those resources are in containers listed in "containers".
///
/// Old sessions may continue to have side-effects on resources not in
/// containers listed in "containers", and thus may affect future
/// sessions' results in ways that are hard to predict.  Thus, if well-defined
/// behavior is desired, it is recommended that all containers be listed in
/// "containers".
///
/// `containers` is a vector of string representation of resource container
/// names. When a resource container is reset, the resources held by the
/// container will be released. In particular, all Variables in the container
/// will become undefined.  If the "containers" vector is empty, the default
/// container is assumed.  If the "containers" vector is non-empty, the
/// default container should be listed explicitly.
///
/// If Reset succeeds, this function will return `OK()`. Otherwise, this
/// function will return an error status.
Status Reset(const SessionOptions& options,
             const std::vector<string>& containers);

/// \brief Create a new session with the given options.
///
/// If a new `Session` object could not be created, this function will
/// return nullptr.
///
/// *Strongly prefer* the version of NewSession that returns Status,
/// which contains more helpful error information.
Session* NewSession(const SessionOptions& options);

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_PUBLIC_SESSION_H_
