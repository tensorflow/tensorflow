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

/// SavedModel loading functions and SavedModelBundle struct.

#ifndef TENSORFLOW_CC_SAVED_MODEL_LOADER_H_
#define TENSORFLOW_CC_SAVED_MODEL_LOADER_H_

#include <string>
#include <unordered_set>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/graph_debug_info.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {

/// Represents a SavedModel that is loaded from storage.
class SavedModelBundleInterface {
 public:
  virtual ~SavedModelBundleInterface();

  /// Returns the TensorFlow Session that can be used to interact with the
  /// SavedModel.
  virtual Session* GetSession() const = 0;

  /// Returns a map from signature name to SignatureDef for all signatures in
  /// in the SavedModel.
  virtual const protobuf::Map<string, SignatureDef>& GetSignatures() const = 0;
};

/// SavedModel representation once the SavedModel is loaded from storage.
///
/// NOTE: Prefer to use SavedModelBundleLite in new code, as it consumes less
/// RAM.
struct SavedModelBundle : public SavedModelBundleInterface {
  /// A TensorFlow Session does not Close itself on destruction. To avoid
  /// resource leaks, we explicitly call Close on Sessions that we create.
  ~SavedModelBundle() override {
    if (session) {
      session->Close().IgnoreError();
    }
  }

  SavedModelBundle() = default;

  Session* GetSession() const override { return session.get(); }
  const protobuf::Map<string, SignatureDef>& GetSignatures() const override {
    return meta_graph_def.signature_def();
  }

  std::unique_ptr<Session> session;
  MetaGraphDef meta_graph_def;
  std::unique_ptr<GraphDebugInfo> debug_info;
};

// A version of SavedModelBundle that avoids storing a potentially large
// MetaGraphDef. Prefer to use SavedModelBundleLite in new code.
class SavedModelBundleLite : public SavedModelBundleInterface {
 public:
  SavedModelBundleLite() = default;
  SavedModelBundleLite& operator=(SavedModelBundleLite&& other) = default;

  SavedModelBundleLite(std::unique_ptr<Session> session,
                       protobuf::Map<string, SignatureDef> signatures)
      : session_(std::move(session)), signatures_(std::move(signatures)) {}

  /// A TensorFlow Session does not Close itself on destruction. To avoid
  /// resource leaks, we explicitly call Close on Sessions that we create.
  ~SavedModelBundleLite() override {
    if (session_) {
      session_->Close().IgnoreError();
    }
  }

  Session* GetSession() const override { return session_.get(); }
  const protobuf::Map<string, SignatureDef>& GetSignatures() const override {
    return signatures_;
  }

 private:
  std::unique_ptr<Session> session_;
  protobuf::Map<string, SignatureDef> signatures_;
};

// Restore variable and resources in the SavedModel export dir for the
// indicated metagraph.
// The recommended way to load a saved model is to call LoadSavedModel,
// which provides an already initialized Metagraph, Session, and DebugInfo.
Status RestoreSession(const RunOptions& run_options,
                      const MetaGraphDef& meta_graph, const string& export_dir,
                      std::unique_ptr<Session>* session);

// Initialize a session which wraps this metagraph.
// The recommended way to load a saved model is to call LoadSavedModel,
// which provides an already initialized Metagraph, Session, and DebugInfo.
Status LoadMetagraphIntoSession(const SessionOptions& session_options,
                                const MetaGraphDef& meta_graph,
                                std::unique_ptr<Session>* session);

/// Loads a SavedModel from the specified export directory. The MetaGraphDef
/// to be loaded is identified by the supplied tags, corresponding exactly to
/// the set of tags used at SavedModel build time. Stores a SavedModel bundle in
/// *bundle with a session and the requested MetaGraphDef, if found.
///
/// NOTE: Prefer the overload that takes a SavedModelBundleLite* in new code.
Status LoadSavedModel(const SessionOptions& session_options,
                      const RunOptions& run_options, const string& export_dir,
                      const std::unordered_set<string>& tags,
                      SavedModelBundle* const bundle);

/// Loads a SavedModel from the specified export directory. The MetaGraphDef
/// to be loaded is identified by the supplied tags, corresponding exactly to
/// the set of tags used at SavedModel build time. Stores a SavedModel bundle
/// in *bundle with a session created from the requested MetaGraphDef if found.
///
/// This overload creates a SavedModelBundleLite, which consumes less RAM than
/// an equivalent SavedModelBundle.
Status LoadSavedModel(const SessionOptions& session_options,
                      const RunOptions& run_options, const string& export_dir,
                      const std::unordered_set<string>& tags,
                      SavedModelBundleLite* const bundle);

/// Checks whether the provided directory could contain a SavedModel. Note that
/// the method does not load any data by itself. If the method returns `false`,
/// the export directory definitely does not contain a SavedModel. If the method
/// returns `true`, the export directory may contain a SavedModel but provides
/// no guarantee that it can be loaded.
bool MaybeSavedModelDirectory(const std::string& export_dir);

}  // namespace tensorflow

#endif  // TENSORFLOW_CC_SAVED_MODEL_LOADER_H_
