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

#include "tensorflow/contrib/session_bundle/session_bundle.h"

#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/any.pb.h"
#include "tensorflow/contrib/session_bundle/manifest.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/protobuf_internal.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saver.pb.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/tensor_bundle/naming.h"

namespace tensorflow {
namespace serving {
namespace {

auto* load_attempt_count = monitoring::Counter<2>::New(
    "/tensorflow/contrib/session_bundle/load_attempt_count",
    "The number of times a SessionBundle was requested to be loaded.",
    "model_path", "status");
auto* load_latency = monitoring::Counter<1>::New(
    "/tensorflow/contrib/session_bundle/load_latency",
    "Latency in microseconds for SessionBundles that were successfully loaded.",
    "model_path");
constexpr char kLoadAttemptFail[] = "fail";
constexpr char kLoadAttemptSuccess[] = "success";

// Create a session using the given options and load the graph.
Status CreateSessionFromGraphDef(const SessionOptions& options,
                                 const GraphDef& graph,
                                 std::unique_ptr<Session>* session) {
  session->reset(NewSession(options));
  return (*session)->Create(graph);
}

Status GetMetaGraphDefFromExport(const StringPiece export_dir,
                                 MetaGraphDef* meta_graph_def) {
  const string meta_graph_def_path =
      io::JoinPath(export_dir, kMetaGraphDefFilename);
  return ReadBinaryProto(Env::Default(), meta_graph_def_path, meta_graph_def);
}

// Creates a string tensor.
Tensor CreateStringTensor(const string& value) {
  Tensor tensor(DT_STRING, TensorShape({}));
  tensor.scalar<string>()() = value;
  return tensor;
}

// Adds Assets related tensors (assets_dir and asset files) to the inputs.
void AddAssetsTensorsToInputs(const StringPiece export_dir,
                              const std::vector<AssetFile>& asset_files,
                              std::vector<std::pair<string, Tensor>>* inputs) {
  if (asset_files.empty()) {
    return;
  }
  for (auto& asset : asset_files) {
    Tensor assets_file_tensor = CreateStringTensor(
        io::JoinPath(export_dir, kAssetsDirectory, asset.filename()));
    inputs->push_back(
        {asset.tensor_binding().tensor_name(), assets_file_tensor});
  }
}

// Historically, model exporter(exporter.py) takes only saver with sharded=True,
// and therefore always exports checkpoint in pattern file names.  In practice,
// instead of training from scratch and export directly, we usually want to
// restore from existing checkpoints and then export directly.  To support such
// case, model exporter now supports reusing saver object restored from existing
// checkpoint, that may have sharded=False - it will then export checkpoint file
// in plain file name.  This method is to support models exported by both types
// of saver object.  The change is backward-compatible, therefore no changes are
// needed for existing model exports.
//
// Checkpoint v2 support: Variables exported using tf-exporter in the checkpoint
// v2 format will have export.index and export.data-?????-of-????? files as
// opposed to just an export or export-?????-of-????? file. The V2 save/restore
// code accepts a filename prefix and assumes both prefix.index and
// prefix.data-* are present in the filesystem. So if we see export.index
// present in the export_dir, we know the export is in V2 format and we return
// <export_dir>/export as this prefix.
string GetVariablesFilename(const StringPiece export_dir) {
  const char kVariablesFilename[] = "export";
  const string kVariablesIndexFilename = MetaFilename("export");  // V2 ckpts
  const char kVariablesFilenamePattern[] = "export-\?\?\?\?\?-of-\?\?\?\?\?";
  if (Env::Default()
          ->FileExists(io::JoinPath(export_dir, kVariablesFilename))
          .ok() ||
      // This works for the case of V2 because the variables filename is taken
      // as a prefix in the save/restore abstraction, and the index and actual
      // variables are meant to be present as prefix.index and
      // prefix.data-?????-of-?????.
      Env::Default()
          ->FileExists(io::JoinPath(export_dir, kVariablesIndexFilename))
          .ok()) {
    return io::JoinPath(export_dir, kVariablesFilename);
  } else {
    return io::JoinPath(export_dir, kVariablesFilenamePattern);
  }
}

Status RunRestoreOp(const RunOptions& run_options, const StringPiece export_dir,
                    const std::vector<AssetFile>& asset_files,
                    const StringPiece restore_op_name,
                    const StringPiece variables_filename_const_op_name,
                    Session* session) {
  LOG(INFO) << "Running restore op for SessionBundle: " << restore_op_name
            << ", " << variables_filename_const_op_name;
  Tensor variables_tensor =
      CreateStringTensor(GetVariablesFilename(export_dir));
  std::vector<std::pair<string, Tensor>> inputs = {
      {string(variables_filename_const_op_name), variables_tensor}};
  AddAssetsTensorsToInputs(export_dir, asset_files, &inputs);
  RunMetadata run_metadata;
  return session->Run(run_options, inputs, {}, {string(restore_op_name)},
                      nullptr /* outputs */, &run_metadata);
}

Status RunInitOp(const RunOptions& run_options, const StringPiece export_dir,
                 const std::vector<AssetFile>& asset_files,
                 const StringPiece init_op_name, Session* session) {
  LOG(INFO) << "Running init op for SessionBundle";
  std::vector<std::pair<string, Tensor>> inputs;
  AddAssetsTensorsToInputs(export_dir, asset_files, &inputs);
  RunMetadata run_metadata;
  return session->Run(run_options, inputs, {}, {string(init_op_name)},
                      nullptr /* outputs */, &run_metadata);
}

Status LoadSessionBundleFromPathUsingRunOptionsInternal(
    const SessionOptions& options, const RunOptions& run_options,
    const StringPiece export_dir, SessionBundle* const bundle) {
  LOG(INFO) << "Attempting to load a SessionBundle from: " << export_dir;
  LOG(INFO) << "Using RunOptions: " << DebugStringIfAvailable(run_options);
  TF_RETURN_IF_ERROR(
      GetMetaGraphDefFromExport(export_dir, &(bundle->meta_graph_def)));

  // Deprecated SessionBundle models may fail to load because newly added
  // attributes are not added to the Graph in the default Session initialization
  // flow. Add an explicit call here when first loading the graph from disk.
  TF_RETURN_IF_ERROR(
      AddDefaultAttrsToGraphDef(bundle->meta_graph_def.mutable_graph_def(),
                                *OpRegistry::Global(), 0 /* node_offset */));

  const auto& collection_def_map = bundle->meta_graph_def.collection_def();
  const auto graph_it = bundle->meta_graph_def.collection_def().find(kGraphKey);
  if (graph_it != collection_def_map.end()) {
    const CollectionDef& graph_collection_def = graph_it->second;
    // Use serving graph_def in MetaGraphDef collection_def.
    if (graph_collection_def.any_list().value_size() != 1) {
      return errors::FailedPrecondition(
          "Expected exactly one serving GraphDef in : ", export_dir);
    }
    const auto& any = graph_collection_def.any_list().value(0);
    GraphDef graph_def;
    TF_RETURN_IF_ERROR(ParseAny(any, &graph_def, "tensorflow.GraphDef"));
    TF_RETURN_IF_ERROR(
        CreateSessionFromGraphDef(options, graph_def, &bundle->session));
  } else {
    // Fallback to use the graph_def in the MetaGraphDef.
    const GraphDef& graph_def = bundle->meta_graph_def.graph_def();
    TF_RETURN_IF_ERROR(
        CreateSessionFromGraphDef(options, graph_def, &bundle->session));
  }

  std::vector<AssetFile> asset_files;
  const auto assets_it = collection_def_map.find(kAssetsKey);
  if (assets_it != collection_def_map.end()) {
    const auto& any_assets = assets_it->second.any_list().value();
    for (const auto& any_asset : any_assets) {
      AssetFile asset_file;
      TF_RETURN_IF_ERROR(
          ParseAny(any_asset, &asset_file, "tensorflow.serving.AssetFile"));
      asset_files.push_back(asset_file);
    }
  }

  TF_RETURN_IF_ERROR(
      RunRestoreOp(run_options, export_dir, asset_files,
                   bundle->meta_graph_def.saver_def().restore_op_name(),
                   bundle->meta_graph_def.saver_def().filename_tensor_name(),
                   bundle->session.get()));

  const auto init_op_it = collection_def_map.find(kInitOpKey);
  if (init_op_it != collection_def_map.end()) {
    if (init_op_it->second.node_list().value_size() != 1) {
      return errors::FailedPrecondition(strings::StrCat(
          "Expected exactly one serving init op in : ", export_dir));
    }
    TF_RETURN_IF_ERROR(RunInitOp(run_options, export_dir, asset_files,
                                 init_op_it->second.node_list().value(0),
                                 bundle->session.get()));
  }

  return Status::OK();
}

}  // namespace

Status LoadSessionBundleFromPath(const SessionOptions& options,
                                 const StringPiece export_dir,
                                 SessionBundle* const bundle) {
  TF_RETURN_IF_ERROR(LoadSessionBundleFromPathUsingRunOptions(
      options, RunOptions(), export_dir, bundle));
  return Status::OK();
}

Status LoadSessionBundleFromPathUsingRunOptions(const SessionOptions& options,
                                                const RunOptions& run_options,
                                                const StringPiece export_dir,
                                                SessionBundle* const bundle) {
  const uint64 start_microseconds = Env::Default()->NowMicros();
  const Status status = LoadSessionBundleFromPathUsingRunOptionsInternal(
      options, run_options, export_dir, bundle);

  const uint64 load_latency_microsecs = [&]() -> uint64 {
    const uint64 end_microseconds = Env::Default()->NowMicros();
    // Avoid clock skew.
    if (end_microseconds < start_microseconds) return 0;
    return end_microseconds - start_microseconds;
  }();
  auto log_and_count = [&](const string& status_str) {
    LOG(INFO) << "Loading SessionBundle: " << status_str << ". Took "
              << load_latency_microsecs << " microseconds.";
    load_attempt_count->GetCell(string(export_dir), status_str)->IncrementBy(1);
  };
  if (status.ok()) {
    log_and_count(kLoadAttemptSuccess);
  } else {
    log_and_count(kLoadAttemptFail);
  }
  load_latency->GetCell(string(export_dir))
      ->IncrementBy(load_latency_microsecs);
  return status;
}

bool IsPossibleExportDirectory(const StringPiece directory) {
  const string meta_graph_def_path =
      io::JoinPath(directory, kMetaGraphDefFilename);
  return Env::Default()->FileExists(meta_graph_def_path).ok();
}

}  // namespace serving
}  // namespace tensorflow
