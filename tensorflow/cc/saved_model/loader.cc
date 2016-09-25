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

#include "tensorflow/cc/saved_model/loader.h"

#include <unordered_set>

#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace {

Status ReadSavedModel(const string& export_dir, SavedModel* saved_model_proto) {
  const string saved_model_path =
      io::JoinPath(export_dir, kSavedModelFilenamePb);
  return ReadBinaryProto(Env::Default(), saved_model_path, saved_model_proto);
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

Status Restore(const RunOptions& run_options, const string& export_dir,
               const StringPiece restore_op_name,
               const StringPiece variable_filename_const_op_name,
               Session* session) {
  const string variables_path =
      io::JoinPath(export_dir, kSavedModelVariablesFilename);
  if (!Env::Default()->FileExists(variables_path)) {
    return Status(error::Code::NOT_FOUND,
                  "Could not find checkpointed variables.");
  }

  // Add variables to the graph.
  Tensor variables_path_tensor(DT_STRING, TensorShape({}));
  variables_path_tensor.scalar<string>()() = variables_path;

  std::vector<std::pair<string, Tensor>> inputs = {
      {variable_filename_const_op_name.ToString(), variables_path_tensor}};

  RunMetadata run_metadata;
  return session->Run(run_options, inputs, {}, {restore_op_name.ToString()},
                      nullptr /* outputs */, &run_metadata);
}

}  // namespace

Status LoadSavedModel(const string& export_dir,
                      const std::unordered_set<string>& tags,
                      const SessionOptions& session_options,
                      const RunOptions& run_options,
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

  TF_RETURN_IF_ERROR(
      Restore(run_options, export_dir,
              bundle->meta_graph_def.saver_def().restore_op_name(),
              bundle->meta_graph_def.saver_def().filename_tensor_name(),
              bundle->session.get()));

  LOG(INFO) << "Done loading SavedModel.";
  return Status::OK();
}

bool MaybeSavedModelDirectory(const string& export_dir) {
  const string saved_model_pb_path =
      io::JoinPath(export_dir, kSavedModelFilenamePb);
  const string saved_model_pbtxt_path =
      io::JoinPath(export_dir, kSavedModelFilenamePbTxt);
  return Env::Default()->FileExists(saved_model_pb_path) ||
         Env::Default()->FileExists(saved_model_pbtxt_path);
}

}  // namespace tensorflow
