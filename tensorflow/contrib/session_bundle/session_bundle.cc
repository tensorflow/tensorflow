/* Copyright 2016 Google Inc. All Rights Reserved.

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
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saver.pb.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace contrib {
namespace {

// Create a session using the given options and load the graph.
Status CreateSessionFromGraphDef(
    const tensorflow::SessionOptions& options, const GraphDef& graph,
    std::unique_ptr<tensorflow::Session>* session) {
  session->reset(NewSession(options));
  return (*session)->Create(graph);
}

Status GetMetaGraphDefFromExport(const StringPiece export_dir,
                                 tensorflow::MetaGraphDef* meta_graph_def) {
  const string meta_graph_def_path =
      tensorflow::io::JoinPath(export_dir, kMetaGraphDefFilename);
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
  if (!asset_files.empty()) {
    for (auto& asset : asset_files) {
      Tensor assets_file_tensor = CreateStringTensor(tensorflow::io::JoinPath(
          tensorflow::io::JoinPath(export_dir, kAssetsDirectory),
          asset.filename()));
      inputs->push_back(
          {asset.tensor_binding().tensor_name(), assets_file_tensor});
    }
  }
}

// Historically, model exporter(exporter.py) takes only saver with
// sharded=True, and therefore always exports checkpoint in pattern file names.
// In practice, instead of training from scratch and export directly, we
// usually want to restore from existing checkpoints and then export directly.
// To support such case, model exporter now supports reusing saver object
// restored from existing checkpoint, that may have sharded=False - it will
// then export checkpoint file in plain file name.
// This method is to support models exported by both types of saver object.
// The change is backward-compatible, therefore no changes are needed for
// existing model exports.
string GetVariablesFilename(const StringPiece export_dir) {
  const char kVariablesFilename[] = "export";
  const char kVariablesFilenamePattern[] = "export-\?\?\?\?\?-of-\?\?\?\?\?";
  if (Env::Default()->FileExists(
      tensorflow::io::JoinPath(export_dir, kVariablesFilename))) {
    return tensorflow::io::JoinPath(export_dir, kVariablesFilename);
  } else {
    return tensorflow::io::JoinPath(export_dir, kVariablesFilenamePattern);
  }
}

Status RunRestoreOp(const StringPiece export_dir,
                    const std::vector<AssetFile>& asset_files,
                    const StringPiece restore_op_name,
                    const StringPiece variables_filename_const_op_name,
                    tensorflow::Session* session) {
  LOG(INFO) << "Running restore op for SessionBundle";
  Tensor variables_tensor = CreateStringTensor(
      GetVariablesFilename(export_dir));
  std::vector<std::pair<string, Tensor>> inputs = {
      {variables_filename_const_op_name.ToString(), variables_tensor}};
  AddAssetsTensorsToInputs(export_dir, asset_files, &inputs);
  return session->Run(inputs, {}, {restore_op_name.ToString()}, nullptr);
}

Status RunInitOp(const StringPiece export_dir,
                 const std::vector<AssetFile>& asset_files,
                 const StringPiece init_op_name, tensorflow::Session* session) {
  LOG(INFO) << "Running init op for SessionBundle";
  std::vector<std::pair<string, Tensor>> inputs;
  AddAssetsTensorsToInputs(export_dir, asset_files, &inputs);
  return session->Run(inputs, {}, {init_op_name.ToString()}, nullptr);
}

}  // namespace

tensorflow::Status LoadSessionBundleFromPath(
    const tensorflow::SessionOptions& options, const StringPiece export_dir,
    SessionBundle* bundle) {
  LOG(INFO) << "Attempting to load a SessionBundle from: " << export_dir;
  TF_RETURN_IF_ERROR(
      GetMetaGraphDefFromExport(export_dir, &(bundle->meta_graph_def)));

  auto collection_def = bundle->meta_graph_def.collection_def();
  if (collection_def.find(kGraphKey) != collection_def.end()) {
    // Use serving graph_def in MetaGraphDef collection_def.
    if (collection_def[kGraphKey].any_list().value_size() != 1) {
      return errors::FailedPrecondition(
          strings::StrCat("Expected exactly one serving GraphDef in : ",
                          bundle->meta_graph_def.DebugString()));
    }
    tensorflow::GraphDef graph_def;
    collection_def[kGraphKey].any_list().value(0).UnpackTo(&graph_def);
    TF_RETURN_IF_ERROR(
        CreateSessionFromGraphDef(options, graph_def, &bundle->session));
  } else {
    // Fallback to use the graph_def in the MetaGraphDef.
    const tensorflow::GraphDef& graph_def = bundle->meta_graph_def.graph_def();
    TF_RETURN_IF_ERROR(
        CreateSessionFromGraphDef(options, graph_def, &bundle->session));
  }

  std::vector<AssetFile> asset_files;
  auto any_assets = collection_def[kAssetsKey].any_list().value();
  for (const auto any_asset : any_assets) {
    AssetFile asset_file;
    any_asset.UnpackTo(&asset_file);
    asset_files.push_back(asset_file);
  }

  TF_RETURN_IF_ERROR(
      RunRestoreOp(export_dir, asset_files,
                   bundle->meta_graph_def.saver_def().restore_op_name(),
                   bundle->meta_graph_def.saver_def().filename_tensor_name(),
                   bundle->session.get()));

  if (collection_def.find(kInitOpKey) != collection_def.end()) {
    if (collection_def[kInitOpKey].node_list().value_size() != 1) {
      return errors::FailedPrecondition(
          strings::StrCat("Expected exactly one serving init op in : ",
                          bundle->meta_graph_def.DebugString()));
    }
    return RunInitOp(export_dir, asset_files,
                     collection_def[kInitOpKey].node_list().value(0),
                     bundle->session.get());
  }

  LOG(INFO) << "Done loading SessionBundle";
  return Status::OK();
}

}  // namespace contrib
}  // namespace tensorflow
