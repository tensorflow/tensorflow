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

// Low-level functionality for setting up a inference Session.

#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_SESSION_BUNDLE_SESSION_BUNDLE_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_SESSION_BUNDLE_SESSION_BUNDLE_H_

#include <memory>

#include "tensorflow/contrib/session_bundle/manifest.pb.h"
#include "tensorflow/contrib/session_bundle/signature.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saver.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace serving {

const char kMetaGraphDefFilename[] = "export.meta";
const char kAssetsDirectory[] = "assets";
const char kInitOpKey[] = "serving_init_op";
const char kAssetsKey[] = "serving_assets";
const char kGraphKey[] = "serving_graph";

// Data and objects loaded from a python Exporter export.
// WARNING(break-tutorial-inline-code): The following code snippet is
// in-lined in tutorials, please update tutorial documents accordingly
// whenever code changes.
struct SessionBundle {
  std::unique_ptr<Session> session;
  MetaGraphDef meta_graph_def;

  // A TensorFlow Session does not Close itself on destruction. To avoid
  // resource leaks, we explicitly call Close on Sessions that we create.
  ~SessionBundle() {
    if (session) {
      session->Close().IgnoreError();
    }
  }

  SessionBundle(SessionBundle&&) = default;
  SessionBundle() = default;
};

// Loads a manifest and initialized session using the output of an Exporter.
Status LoadSessionBundleFromPath(const SessionOptions& options,
                                 const StringPiece export_dir,
                                 SessionBundle* bundle);

// Similar to the LoadSessionBundleFromPath(), but also allows the session run
// invocations for the restore and init ops to be configured with
// tensorflow::RunOptions.
//
// This method is EXPERIMENTAL and may change or be removed.
Status LoadSessionBundleFromPathUsingRunOptions(
    const SessionOptions& session_options, const RunOptions& run_options,
    const StringPiece export_dir, SessionBundle* bundle);

// Sanity checks whether the directory looks like an export directory. Note that
// we don't try to load any data in this method.
//
// If the method returns false this is definitely not an export directory, if it
// returns true, it is no guarantee that the model will load.
bool IsPossibleExportDirectory(const StringPiece export_dir);

}  // namespace serving
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_SESSION_BUNDLE_SESSION_BUNDLE_H_
