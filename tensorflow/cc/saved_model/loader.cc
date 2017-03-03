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

#include <unordered_set>

#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/protobuf_internal.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"
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
    "Latency in microseconds for SavedModels that were succesfully loaded.",
    "model_path");
constexpr char kLoadAttemptFail[] = "fail";
constexpr char kLoadAttemptSuccess[] = "success";

Status ReadSavedModel(const string& export_dir, SavedModel* saved_model_proto) {
  const string saved_model_pb_path =
      io::JoinPath(export_dir, kSavedModelFilenamePb);
  if (Env::Default()->FileExists(saved_model_pb_path).ok()) {
    return ReadBinaryProto(Env::Default(), saved_model_pb_path,
                           saved_model_proto);
  }
  const string saved_model_pbtxt_path =
      io::JoinPath(export_dir, kSavedModelFilenamePbTxt);
  if (Env::Default()->FileExists(saved_model_pbtxt_path).ok()) {
    return ReadTextProto(Env::Default(), saved_model_pbtxt_path,
                         saved_model_proto);
  }
  return Status(error::Code::NOT_FOUND,
                "Could not find SavedModel .pb or .pbtxt at supplied export "
                "directory path: " +
                    export_dir);
}

Status FindMetaGraphDefToLoad(const SavedModel& saved_model_proto,
                              const std::unordered_set<string>& tags,
                              MetaGraphDef* meta_graph_def_to_load) {
  for (const MetaGraphDef& meta_graph_def : saved_model_proto.meta_graphs()) {
    // Get tags from the meta_graph_def.
    std::unordered_set<string> graph_tags;
    for (const string& tag : meta_graph_def.meta_info_def().tags()) {
      graph_tags.insert(tag);
    }
    // Match with the set of tags provided.
    if (graph_tags == tags) {
      *meta_graph_def_to_load = meta_graph_def;
      return Status::OK();
    }
  }
  return Status(error::Code::NOT_FOUND,
                "Could not find meta graph def matching supplied tags.");
}

Status LoadMetaGraphIntoSession(const MetaGraphDef& meta_graph_def,
                                const SessionOptions& session_options,
                                std::unique_ptr<Session>* session) {
  session->reset(NewSession(session_options));
  return (*session)->Create(meta_graph_def.graph_def());
}

Tensor CreateStringTensor(const string& value) {
  Tensor tensor(DT_STRING, TensorShape({}));
  tensor.scalar<string>()() = value;
  return tensor;
}

void AddAssetsTensorsToInputs(const StringPiece export_dir,
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

Status RunRestore(const RunOptions& run_options, const string& export_dir,
                  const StringPiece restore_op_name,
                  const StringPiece variable_filename_const_op_name,
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
  if (!Env::Default()->FileExists(variables_index_path).ok()) {
    LOG(INFO) << "The specified SavedModel has no variables; no checkpoints "
                 "were restored.";
    return Status::OK();
  }
  const string variables_path =
      io::JoinPath(variables_directory, kSavedModelVariablesFilename);

  // Add variables to the graph.
  Tensor variables_path_tensor(DT_STRING, TensorShape({}));
  variables_path_tensor.scalar<string>()() = variables_path;

  std::vector<std::pair<string, Tensor>> inputs = {
      {variable_filename_const_op_name.ToString(), variables_path_tensor}};

  AddAssetsTensorsToInputs(export_dir, asset_file_defs, &inputs);

  RunMetadata run_metadata;
  return session->Run(run_options, inputs, {}, {restore_op_name.ToString()},
                      nullptr /* outputs */, &run_metadata);
}

Status RunLegacyInitOp(const RunOptions& run_options, const string& export_dir,
                       const MetaGraphDef& meta_graph_def,
                       const std::vector<AssetFileDef>& asset_file_defs,
                       Session* session) {
  LOG(INFO) << "Running LegacyInitOp on SavedModel bundle.";
  const auto& collection_def_map = meta_graph_def.collection_def();
  const auto init_op_it = collection_def_map.find(kSavedModelLegacyInitOpKey);
  if (init_op_it != collection_def_map.end()) {
    if (init_op_it->second.node_list().value_size() != 1) {
      return errors::FailedPrecondition(strings::StrCat(
          "Expected exactly one serving init op in : ", export_dir));
    }
    std::vector<std::pair<string, Tensor>> inputs;
    AddAssetsTensorsToInputs(export_dir, asset_file_defs, &inputs);
    RunMetadata run_metadata;
    const StringPiece legacy_init_op_name =
        init_op_it->second.node_list().value(0);
    return session->Run(run_options, inputs, {},
                        {legacy_init_op_name.ToString()}, nullptr /* outputs */,
                        &run_metadata);
  }
  return Status::OK();
}

Status GetAssetFileDefs(const MetaGraphDef& meta_graph_def,
                        std::vector<AssetFileDef>* asset_file_defs) {
  const auto& collection_def_map = meta_graph_def.collection_def();
  const auto assets_it = collection_def_map.find(kSavedModelAssetsKey);
  if (assets_it == collection_def_map.end()) {
    return Status::OK();
  }
  const auto& any_assets = assets_it->second.any_list().value();
  for (const auto& any_asset : any_assets) {
    AssetFileDef asset_file_def;
    TF_RETURN_IF_ERROR(
        ParseAny(any_asset, &asset_file_def, "tensorflow.AssetFileDef"));
    asset_file_defs->push_back(asset_file_def);
  }
  return Status::OK();
}

Status LoadSavedModelInternal(const SessionOptions& session_options,
                              const RunOptions& run_options,
                              const string& export_dir,
                              const std::unordered_set<string>& tags,
                              SavedModelBundle* const bundle) {
  if (!MaybeSavedModelDirectory(export_dir)) {
    return Status(error::Code::NOT_FOUND,
                  "SavedModel not found in export directory: " + export_dir);
  }
  LOG(INFO) << "Loading SavedModel from: " << export_dir;

  SavedModel saved_model_proto;
  TF_RETURN_IF_ERROR(ReadSavedModel(export_dir, &saved_model_proto));

  TF_RETURN_IF_ERROR(
      FindMetaGraphDefToLoad(saved_model_proto, tags, &bundle->meta_graph_def));

  TF_RETURN_IF_ERROR(LoadMetaGraphIntoSession(
      bundle->meta_graph_def, session_options, &bundle->session));

  std::vector<AssetFileDef> asset_file_defs;
  TF_RETURN_IF_ERROR(
      GetAssetFileDefs(bundle->meta_graph_def, &asset_file_defs));
  TF_RETURN_IF_ERROR(
      RunRestore(run_options, export_dir,
                 bundle->meta_graph_def.saver_def().restore_op_name(),
                 bundle->meta_graph_def.saver_def().filename_tensor_name(),
                 asset_file_defs, bundle->session.get()));
  // TODO(sukritiramesh): Add support for a single main op to run upon load,
  // which will supersede the legacy_init_op and separate RunRestore.
  TF_RETURN_IF_ERROR(RunLegacyInitOp(run_options, export_dir,
                                     bundle->meta_graph_def, asset_file_defs,
                                     bundle->session.get()));
  return Status::OK();
}

}  // namespace

Status LoadSavedModel(const SessionOptions& session_options,
                      const RunOptions& run_options, const string& export_dir,
                      const std::unordered_set<string>& tags,
                      SavedModelBundle* const bundle) {
  // TODO(robson): Add tests for the counters.
  const uint64 start_microseconds = Env::Default()->NowMicros();
  const Status status = LoadSavedModelInternal(session_options, run_options,
                                               export_dir, tags, bundle);
  const uint64 load_latency_microsecs = [&]() -> uint64 {
    const uint64 end_microseconds = Env::Default()->NowMicros();
    // Avoid clock skew.
    if (end_microseconds < start_microseconds) return 0;
    return end_microseconds - start_microseconds;
  }();
  auto log_and_count = [&](const string& status_str) {
    LOG(INFO) << "Loading SavedModel: " << status_str << ". Took "
              << load_latency_microsecs << " microseconds.";
    load_attempt_count->GetCell(export_dir, status_str)->IncrementBy(1);
  };
  if (status.ok()) {
    log_and_count(kLoadAttemptSuccess);
  } else {
    log_and_count(kLoadAttemptFail);
  }
  load_latency->GetCell(export_dir)->IncrementBy(load_latency_microsecs);
  return status;
}

bool MaybeSavedModelDirectory(const string& export_dir) {
  const string saved_model_pb_path =
      io::JoinPath(export_dir, kSavedModelFilenamePb);
  const string saved_model_pbtxt_path =
      io::JoinPath(export_dir, kSavedModelFilenamePbTxt);
  return Env::Default()->FileExists(saved_model_pb_path).ok() ||
         Env::Default()->FileExists(saved_model_pbtxt_path).ok();
}

}  // namespace tensorflow
