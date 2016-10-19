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

// Shim for systems that need to load both SessionBundle and
// SavedModelBundle interchangeably during migration to SavedModel.
#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_SESSION_BUNDLE_BUNDLE_SHIM_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_SESSION_BUNDLE_BUNDLE_SHIM_H_

#include <memory>

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/contrib/session_bundle/session_bundle.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace serving {
namespace internal {

// Loads a SavedModel from a session-bundle path. Uses the provided
// session-options to set up the session in the bundle. Also, uses the supplied
// run-options for the session run calls.
Status LoadSavedModelFromLegacySessionBundlePath(
    const SessionOptions& session_options, const RunOptions& run_options,
    const StringPiece session_bundle_export_dir, SavedModelBundle* bundle);

// Adds an entry (key and value) to the input map of the signature def.
void AddInputToSignatureDef(const string& tensor_name,
                            const string& input_map_key,
                            SignatureDef* signature_def);

// Adds an entry (key and value) to the output map of the signature def.
void AddOutputToSignatureDef(const string& tensor_name,
                             const string& output_map_key,
                             SignatureDef* signature_def);

// Converts a SessionBundle to a SavedModel bundle.
Status ConvertSessionBundleToSavedModelBundle(
    SessionBundle& session_bundle, SavedModelBundle* saved_model_bundle);

// Converts signatures in the MetaGraphDef into a SignatureDef in the
// MetaGraphDef.
Status ConvertSignaturesToSignatureDef(MetaGraphDef* meta_graph_def);

// Converts the default signature, if any, from the MetaGraphDef into a
// SignatureDef in the MetaGraphDef. Only supports up conversion for
// `ClassificationSignature` and `RegressionSignature`.
Status ConvertDefaultSignatureToSignatureDef(const Signatures& signatures,
                                             MetaGraphDef* meta_graph_def);

// Converts the named signatures, if any, from the MetaGraphDef into the
// SignatureDef in the MetaGraphDef. Up conversion, in this case, requires the
// set of named signatures to contain at least two GenericSignatures
// corresponding to `inputs` and `outputs`.
Status ConvertNamedSignaturesToSignatureDef(const Signatures& signatures,
                                            MetaGraphDef* meta_graph_def);

}  // namespace internal

// Loads a SavedModel from either a session-bundle path or a SavedModel bundle
// path.
Status LoadSessionBundleOrSavedModelBundle(
    const SessionOptions& session_options, const RunOptions& run_options,
    const string& export_dir, const std::unordered_set<string>& tags,
    SavedModelBundle* bundle);

}  // namespace serving
}  // namespace tensorflow
#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_SESSION_BUNDLE_BUNDLE_SHIM_H_
