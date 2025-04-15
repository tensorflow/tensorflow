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

#include "tensorflow/cc/saved_model/loader.h"

#include <memory>
#include <string>
#include <unordered_set>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/cc/saved_model/fingerprinting.h"
#include "tensorflow/cc/saved_model/loader_util.h"
#include "tensorflow/cc/saved_model/metrics.h"
#include "tensorflow/cc/saved_model/reader.h"
#include "tensorflow/cc/saved_model/util.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph_debug_info.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/lib/monitoring/sampler.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/file_system_helper.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saver.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/tensor_bundle/naming.h"

namespace tensorflow {
namespace {

auto* load_attempt_count = monitoring::Counter<2>::New(
    "/tensorflow/cc/saved_model/load_attempt_count",
    "The number of times a SavedModel was successfully loaded.", "model_path",
    "status");
auto* load_latency = monitoring::Counter<1>::New(
    "/tensorflow/cc/saved_model/load_latency",
    "Latency in microseconds for SavedModels that were successfully loaded.",
    "model_path");
auto* load_latency_by_stage = monitoring::Sampler<2>::New(
    {
        "/tensorflow/cc/saved_model/load_latency_by_stage",  // metric name
        "Distribution of wall time spent (in microseconds) in each stage "
        "(restore graph from disk, run init graph op, etc) when loading the "
        "model",
        "model_path",
        "stage",
    },
    // Scale of 10, power of 1.8 with bucket count 37 (~258 minutes).
    monitoring::Buckets::Exponential(10, 1.8, 37));

constexpr char kLoadAttemptFail[] = "fail";
constexpr char kLoadAttemptSuccess[] = "success";
// `tensorflow::LoadSavedModel` API label.
constexpr char kCCLoadLabel[] = "cc_load";

uint64 GetLatencyMicroseconds(const uint64 start_microseconds) {
  const uint64 end_microseconds = EnvTime::NowMicros();
  // Avoid clock skew.
  if (end_microseconds < start_microseconds) return 0;
  return end_microseconds - start_microseconds;
}

// Ensure that constant tensors loaded from the saved model have valid shape.
// Also ensure that constant nodes have a value assigned to them.
// TODO(b/154763635): this is temporary and will be replaced with a better audit
static absl::Status ValidateNode(const NodeDef& node) {
  const auto node_iterator = node.attr().find("value");
  if (node_iterator != node.attr().end()) {
    AttrValue node_value = node_iterator->second;
    if (node_value.has_tensor()) {
      const PartialTensorShape node_shape(node_value.tensor().tensor_shape());
      if (node_shape.num_elements() < 0) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Saved model contains node \"", node.name(), "\" (op \"", node.op(),
            "\") which initializes from a tensor with ",
            node_shape.num_elements(), " elements"));
      }
    }
  } else if (node.op() == "Const") {
    return absl::FailedPreconditionError(absl::StrCat(
        "Saved model contains node \"", node.name(),
        "\" which is a constant tensor but no value has been provided"));
  }
  return absl::OkStatus();
}

static absl::Status ValidateFunctionNotRecursive(const FunctionDef& function) {
  const auto& function_name = function.signature().name();
  for (const auto& node : function.node_def()) {
    if (node.op() == function_name) {
      return absl::FailedPreconditionError(absl::StrCat(
          "Function ", function_name,
          " is self recursive and TensorFlow does not support this scenario."));
    }
  }

  return absl::OkStatus();
}

static absl::Status ValidateSavedTensors(const GraphDef& graph_def) {
  for (const auto& node : graph_def.node()) {
    TF_RETURN_IF_ERROR(ValidateNode(node));
  }

  if (graph_def.has_library()) {
    const FunctionDefLibrary& library = graph_def.library();
    for (const auto& function : library.function()) {
      for (const auto& node : function.node_def()) {
        TF_RETURN_IF_ERROR(ValidateNode(node));
      }

      // Also check that there is no recursivity in the library
      TF_RETURN_IF_ERROR(ValidateFunctionNotRecursive(function));
    }
  }

  return absl::OkStatus();
}

Tensor CreateStringTensor(const string& value) {
  Tensor tensor(DT_STRING, TensorShape({}));
  tensor.scalar<tstring>()() = value;
  return tensor;
}

void AddAssetsTensorsToInputs(const absl::string_view export_dir,
                              const std::vector<AssetFileDef>& asset_file_defs,
                              std::vector<std::pair<string, Tensor>>* inputs) {
  if (asset_file_defs.empty()) {
    return;
  }
  for (auto& asset_file_def : asset_file_defs) {
    Tensor assets_file_path_tensor = CreateStringTensor(io::JoinPath(
        export_dir, kSavedModelAssetsDirectory, asset_file_def.filename()));
    inputs->push_back(
        {asset_file_def.tensor_info().name(), assets_file_path_tensor});
  }
}

// Like Session::Run(), but uses the Make/Run/ReleaseCallable() API to avoid
// leaving behind non-GC'ed state.
//
// Detailed motivation behind this approach, from ashankar@:
//
// Each call to Session::Run() that identifies a new subgraph (based on feeds
// and fetches) creates some datastructures that live as long as the session
// (the partitioned graph, associated executors etc.).
//
// A pathological case of this would be if say the initialization op
// (main_op/legacy_init_op) involves the use of a large constant. Then we
// allocate memory for that large constant that will just stick around till the
// session dies. With this Callable mechanism, that memory will be released
// right after ReleaseCallable returns.
//
// However, the resource manager state remains.
absl::Status RunOnce(const RunOptions& run_options,
                     const std::vector<std::pair<string, Tensor>>& inputs,
                     const std::vector<string>& output_tensor_names,
                     const std::vector<string>& target_node_names,
                     std::vector<Tensor>* outputs, RunMetadata* run_metadata,
                     Session* session) {
  CallableOptions callable_options;
  std::vector<Tensor> feed_tensors;
  *callable_options.mutable_run_options() = run_options;
  for (const auto& input : inputs) {
    const string& name = input.first;
    const Tensor& tensor = input.second;
    callable_options.add_feed(name);
    feed_tensors.push_back(tensor);
  }
  for (const string& output_tensor_name : output_tensor_names) {
    callable_options.add_fetch(output_tensor_name);
  }
  for (const string& target_node_name : target_node_names) {
    callable_options.add_target(target_node_name);
  }

  Session::CallableHandle callable_handle;
  TF_RETURN_IF_ERROR(session->MakeCallable(callable_options, &callable_handle));
  const absl::Status run_status = session->RunCallable(
      callable_handle, feed_tensors, outputs, run_metadata);
  // Be sure to call ReleaseCallable() regardless of the outcome of
  // RunCallable().
  session->ReleaseCallable(callable_handle).IgnoreError();
  return run_status;
}

// RunInitOp will return OK if the initialization op was run successfully.
// An empty init_op_name indicates that there are no init ops to run.
absl::Status RunInitOp(const RunOptions& run_options, const string& export_dir,
                       const MetaGraphDef& meta_graph_def,
                       const std::vector<AssetFileDef>& asset_file_defs,
                       Session* session, const string& init_op_name) {
  if (!init_op_name.empty()) {
    LOG(INFO) << "Running initialization op on SavedModel bundle at path: "
              << export_dir;
    std::vector<std::pair<string, Tensor>> inputs;
    AddAssetsTensorsToInputs(export_dir, asset_file_defs, &inputs);
    RunMetadata run_metadata;
    return RunOnce(run_options, inputs, {}, {init_op_name},
                   nullptr /* outputs */, &run_metadata, session);
  }
  return absl::OkStatus();
}

absl::Status RunRestore(const RunOptions& run_options, const string& export_dir,
                        const absl::string_view restore_op_name,
                        const absl::string_view variable_filename_const_op_name,
                        const std::vector<AssetFileDef>& asset_file_defs,
                        Session* session) {
  LOG(INFO) << "Restoring SavedModel bundle.";
  // Find path to variables to be restored in export directory.
  const string variables_directory =
      io::JoinPath(export_dir, kSavedModelVariablesDirectory);
  // Check for saver checkpoints in v2 format. Models exported in the checkpoint
  // v2 format will have a variables.index file. The corresponding
  // variables are stored in the variables.data-?????-of-????? files.
  const string variables_index_path = io::JoinPath(
      variables_directory, MetaFilename(kSavedModelVariablesFilename));
  TF_ASSIGN_OR_RETURN(
      bool variables_index_exists,
      internal::FileExists(Env::Default(), variables_index_path));
  if (!variables_index_exists) {
    LOG(INFO) << "The specified SavedModel has no variables; no checkpoints "
                 "were restored. File does not exist: "
              << variables_index_path;
    return absl::OkStatus();
  }
  const string variables_path =
      io::JoinPath(variables_directory, kSavedModelVariablesFilename);

  // Add variables to the graph.
  Tensor variables_path_tensor(DT_STRING, TensorShape({}));
  variables_path_tensor.scalar<tstring>()() = variables_path;

  std::vector<std::pair<string, Tensor>> inputs = {
      {string(variable_filename_const_op_name), variables_path_tensor}};

  AddAssetsTensorsToInputs(export_dir, asset_file_defs, &inputs);

  RunMetadata run_metadata;
  return RunOnce(run_options, inputs, {}, {string(restore_op_name)},
                 nullptr /* outputs */, &run_metadata, session);
}

}  // namespace

SavedModelBundleInterface::~SavedModelBundleInterface() = default;

absl::Status LoadMetagraphIntoSession(const SessionOptions& session_options,
                                      const MetaGraphDef& meta_graph,
                                      std::unique_ptr<Session>* session) {
  Session* session_p = nullptr;
  TF_RETURN_IF_ERROR(NewSession(session_options, &session_p));
  session->reset(session_p);
  TF_RETURN_IF_ERROR(ValidateSavedTensors(meta_graph.graph_def()));
  return (*session)->Create(meta_graph.graph_def());
}

absl::Status LoadGraphDefIntoSession(const SessionOptions& session_options,
                                     GraphDef graph_def,
                                     std::unique_ptr<Session>* session) {
  Session* session_p = nullptr;
  TF_RETURN_IF_ERROR(NewSession(session_options, &session_p));
  session->reset(session_p);
  TF_RETURN_IF_ERROR(ValidateSavedTensors(graph_def));
  return (*session)->Create(std::move(graph_def));
}

absl::Status LoadSavedModelInternal(const SessionOptions& session_options,
                                    const RunOptions& run_options,
                                    const string& export_dir,
                                    const std::unordered_set<string>& tags,
                                    SavedModelBundle* const bundle) {
  TF_RETURN_IF_ERROR(ReadMetaGraphDefFromSavedModel(export_dir, tags,
                                                    &bundle->meta_graph_def));
  TF_RETURN_IF_ERROR(
      ReadSavedModelDebugInfoIfPresent(export_dir, &bundle->debug_info));
  TF_RETURN_IF_ERROR(LoadMetagraphIntoSession(
      session_options, bundle->meta_graph_def, &bundle->session));
  TF_RETURN_IF_ERROR(RestoreSession(run_options, bundle->meta_graph_def,
                                    export_dir, &bundle->session));
  return absl::OkStatus();
}

namespace {
// Session wrapper that prevents calls to Session::Create(), Session::Extend(),
// and the deprecated partial-run methods.
//
// Limiting the available methods on a returned Session gives us the option
// to replace the Session with a cut-down implementation, without breaking any
// users.
class LiteSessionWrapper : public Session {
 public:
  explicit LiteSessionWrapper(std::unique_ptr<Session> wrapped)
      : wrapped_(std::move(wrapped)) {}

  absl::Status Create(const GraphDef& graph) override {
    return absl::UnimplementedError("Session::Create()");
  }
  absl::Status Create(GraphDef&& graph) override {
    return absl::UnimplementedError("Session::Create()");
  }

  absl::Status Extend(const GraphDef& graph) override {
    return absl::UnimplementedError("Session::Extend()");
  }
  absl::Status Extend(GraphDef&& graph) override {
    return absl::UnimplementedError("Session::Extend()");
  }

  absl::Status Run(const std::vector<std::pair<string, Tensor>>& inputs,
                   const std::vector<string>& output_tensor_names,
                   const std::vector<string>& target_node_names,
                   std::vector<Tensor>* outputs) override {
    return wrapped_->Run(inputs, output_tensor_names, target_node_names,
                         outputs);
  }

  absl::Status Create(const RunOptions& run_options,
                      const GraphDef& graph) override {
    return absl::UnimplementedError("Session::Create()");
  }
  absl::Status Extend(const RunOptions& run_options,
                      const GraphDef& graph) override {
    return absl::UnimplementedError("Session::Extend()");
  }
  absl::Status Create(const RunOptions& run_options,
                      GraphDef&& graph) override {
    return absl::UnimplementedError("Session::Create()");
  }
  absl::Status Extend(const RunOptions& run_options,
                      GraphDef&& graph) override {
    return absl::UnimplementedError("Session::Extend()");
  }
  absl::Status Close(const RunOptions& run_options) override {
    return wrapped_->Close(run_options);
  }

  absl::Status Run(const RunOptions& run_options,
                   const std::vector<std::pair<string, Tensor>>& inputs,
                   const std::vector<string>& output_tensor_names,
                   const std::vector<string>& target_node_names,
                   std::vector<Tensor>* outputs,
                   RunMetadata* run_metadata) override {
    return wrapped_->Run(run_options, inputs, output_tensor_names,
                         target_node_names, outputs, run_metadata);
  }

  absl::Status PRunSetup(const std::vector<string>& input_names,
                         const std::vector<string>& output_names,
                         const std::vector<string>& target_nodes,
                         string* handle) override {
    return absl::UnimplementedError("Session::PRunSetup()");
  }

  absl::Status PRun(const string& handle,
                    const std::vector<std::pair<string, Tensor>>& inputs,
                    const std::vector<string>& output_names,
                    std::vector<Tensor>* outputs) override {
    return absl::UnimplementedError("Session::PRun()");
  }

  absl::Status ListDevices(std::vector<DeviceAttributes>* response) override {
    return wrapped_->ListDevices(response);
  }

  absl::Status Close() override { return wrapped_->Close(); }

  absl::Status LocalDeviceManager(const DeviceMgr** device_mgr) override {
    return wrapped_->LocalDeviceManager(device_mgr);
  }

  absl::Status MakeCallable(const CallableOptions& callable_options,
                            CallableHandle* out_handle) override {
    return wrapped_->MakeCallable(callable_options, out_handle);
  }

  absl::Status RunCallable(CallableHandle handle,
                           const std::vector<Tensor>& feed_tensors,
                           std::vector<Tensor>* fetch_tensors,
                           RunMetadata* run_metadata) override {
    return wrapped_->RunCallable(handle, feed_tensors, fetch_tensors,
                                 run_metadata);
  }

  absl::Status RunCallable(
      CallableHandle handle, const std::vector<Tensor>& feed_tensors,
      std::vector<Tensor>* fetch_tensors, RunMetadata* run_metadata,
      const thread::ThreadPoolOptions& threadpool_options) override {
    return wrapped_->RunCallable(handle, feed_tensors, fetch_tensors,
                                 run_metadata, threadpool_options);
  }

  absl::Status ReleaseCallable(CallableHandle handle) override {
    return wrapped_->ReleaseCallable(handle);
  }

  absl::Status Finalize() override { return wrapped_->Finalize(); }

 private:
  const std::unique_ptr<Session> wrapped_;
};
}  // namespace

absl::Status LoadSavedModelInternal(const SessionOptions& session_options,
                                    const RunOptions& run_options,
                                    const string& export_dir,
                                    const std::unordered_set<string>& tags,
                                    SavedModelBundleLite* const bundle) {
  MetaGraphDef meta_graph_def;
  TF_RETURN_IF_ERROR(
      ReadMetaGraphDefFromSavedModel(export_dir, tags, &meta_graph_def));
  std::unique_ptr<Session> session;
  TF_RETURN_IF_ERROR(LoadGraphDefIntoSession(
      session_options, std::move(*meta_graph_def.mutable_graph_def()),
      &session));
  TF_RETURN_IF_ERROR(
      RestoreSession(run_options, meta_graph_def, export_dir, &session));
  *bundle = SavedModelBundleLite(
      std::make_unique<LiteSessionWrapper>(std::move(session)),
      std::move(*meta_graph_def.mutable_signature_def()));
  return absl::OkStatus();
}

template <typename BundleType>
absl::Status LoadSavedModelGeneric(const SessionOptions& session_options,
                                   const RunOptions& run_options,
                                   const string& export_dir,
                                   const std::unordered_set<string>& tags,
                                   BundleType* const bundle) {
  metrics::SavedModelReadApi(kCCLoadLabel).IncrementBy(1);
  auto fingerprint_proto =
      saved_model::fingerprinting::ReadSavedModelFingerprint(export_dir);
  if (fingerprint_proto.ok()) {
    // Set gauge cell with saved_model_checksum.
    metrics::SavedModelReadFingerprint().Set(
        std::to_string(fingerprint_proto->saved_model_checksum()));
  }

  // TODO(robson): Add tests for the counters.
  const uint64 start_microseconds = Env::Default()->NowMicros();
  const absl::Status status = LoadSavedModelInternal(
      session_options, run_options, export_dir, tags, bundle);
  auto log_and_count = [&](const string& status_str) {
    LOG(INFO) << "SavedModel load for tags { " << absl::StrJoin(tags, " ")
              << " }; Status: " << status_str << ": " << status << ". Took "
              << GetLatencyMicroseconds(start_microseconds) << " microseconds.";
    load_attempt_count->GetCell(export_dir, status_str)->IncrementBy(1);
  };
  if (status.ok()) {
    log_and_count(kLoadAttemptSuccess);
    metrics::SavedModelReadPath().Set(export_dir);
  } else {
    log_and_count(kLoadAttemptFail);
  }
  load_latency->GetCell(export_dir)
      ->IncrementBy(GetLatencyMicroseconds(start_microseconds));
  return status;
}

absl::Status LoadSavedModel(const SessionOptions& session_options,
                            const RunOptions& run_options,
                            const string& export_dir,
                            const std::unordered_set<string>& tags,
                            SavedModelBundle* const bundle) {
  return LoadSavedModelGeneric<SavedModelBundle>(session_options, run_options,
                                                 export_dir, tags, bundle);
}

absl::Status RestoreSession(const RunOptions& run_options,
                            const MetaGraphDef& meta_graph,
                            const string& export_dir,
                            std::unique_ptr<Session>* session) {
  const uint64 read_start_microseconds = Env::Default()->NowMicros();
  std::vector<AssetFileDef> asset_file_defs;
  TF_RETURN_IF_ERROR(internal::GetAssetFileDefs(meta_graph, &asset_file_defs));
  if (meta_graph.has_saver_def()) {
    TF_RETURN_IF_ERROR(RunRestore(run_options, export_dir,
                                  meta_graph.saver_def().restore_op_name(),
                                  meta_graph.saver_def().filename_tensor_name(),
                                  asset_file_defs, session->get()));
  }
  // Record walltime spent in restoring graph from disk, but postpone metric
  // increments until graph init finishes.
  const uint64 restore_graph_walltime =
      GetLatencyMicroseconds(read_start_microseconds);

  const uint64 graph_init_start_microseconds = Env::Default()->NowMicros();
  string init_op_name;
  TF_RETURN_IF_ERROR(
      internal::GetInitOp(export_dir, meta_graph, &init_op_name));
  TF_RETURN_IF_ERROR(RunInitOp(run_options, export_dir, meta_graph,
                               asset_file_defs, session->get(), init_op_name));
  load_latency_by_stage->GetCell(export_dir, "restore_graph")
      ->Add(restore_graph_walltime);
  // Record wall time spent in init op.
  load_latency_by_stage->GetCell(export_dir, "init_graph")
      ->Add(GetLatencyMicroseconds(graph_init_start_microseconds));
  return absl::OkStatus();
}

absl::Status LoadSavedModel(const SessionOptions& session_options,
                            const RunOptions& run_options,
                            const string& export_dir,
                            const std::unordered_set<string>& tags,
                            SavedModelBundleLite* const bundle) {
  SessionOptions rewritten_options(session_options);
  // We disallow calls to Session::Extend() on the returned session, so we can
  // reduce memory consumption by not storing the original GraphDef.
  rewritten_options.config.mutable_experimental()
      ->set_optimize_for_static_graph(true);
  // Disallowing the `RunOptions.output_partition_graphs` option (typically used
  // in debugging and tests) allows us to reduce memory consumption further by
  // not storing the rewritten subgraph for each signature.
  rewritten_options.config.mutable_experimental()
      ->set_disable_output_partition_graphs(true);
  // TODO(mrry): Consider specializing the session creation to reduce peak
  // RAM consumption by using `Session::Create(GraphDef&&)`.
  TF_RETURN_IF_ERROR(LoadSavedModelGeneric(rewritten_options, run_options,
                                           export_dir, tags, bundle));
  return absl::OkStatus();
}

bool MaybeSavedModelDirectory(const string& export_dir) {
  const string saved_model_pb_path =
      io::JoinPath(export_dir, kSavedModelFilenamePb);
  const string saved_model_cpb_path =
      io::JoinPath(export_dir, kSavedModelFilenameCpb);
  const string saved_model_pbtxt_path =
      io::JoinPath(export_dir, kSavedModelFilenamePbTxt);
  return Env::Default()->FileExists(saved_model_pb_path).ok() ||
         Env::Default()->FileExists(saved_model_cpb_path).ok() ||
         Env::Default()->FileExists(saved_model_pbtxt_path).ok();
}

}  // namespace tensorflow
