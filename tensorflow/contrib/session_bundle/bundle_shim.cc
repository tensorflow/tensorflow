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

#include "tensorflow/contrib/session_bundle/bundle_shim.h"

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/contrib/session_bundle/manifest.pb.h"
#include "tensorflow/contrib/session_bundle/session_bundle.h"
#include "tensorflow/contrib/session_bundle/signature.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace serving {
namespace {
///////////////////////////////////////////////////////////////////////////////
// Helper functions to check Signature type.

bool IsClassificationSignature(const Signature& signature) {
  return signature.type_case() == Signature::kClassificationSignature;
}

bool IsRegressionSignature(const Signature& signature) {
  return signature.type_case() == Signature::kRegressionSignature;
}

///////////////////////////////////////////////////////////////////////////////
// Helper functions to build `Classification`, `Regression` and `Predict`
// SignatureDefs.

SignatureDef BuildRegressionSignatureDef(
    const RegressionSignature& regression_signature,
    const std::unordered_map<string, DataType>& tensor_name_to_dtype) {
  SignatureDef signature_def;
  signature_def.set_method_name(kRegressMethodName);
  internal::AddInputToSignatureDef(regression_signature.input().tensor_name(),
                                   tensor_name_to_dtype, kRegressInputs,
                                   &signature_def);
  internal::AddOutputToSignatureDef(regression_signature.output().tensor_name(),
                                    tensor_name_to_dtype, kRegressOutputs,
                                    &signature_def);
  return signature_def;
}

SignatureDef BuildClassificationSignatureDef(
    const ClassificationSignature& classification_signature,
    const std::unordered_map<string, DataType>& tensor_name_to_dtype) {
  SignatureDef signature_def;
  signature_def.set_method_name(kClassifyMethodName);
  internal::AddInputToSignatureDef(
      classification_signature.input().tensor_name(), tensor_name_to_dtype,
      kClassifyInputs, &signature_def);
  internal::AddOutputToSignatureDef(
      classification_signature.classes().tensor_name(), tensor_name_to_dtype,
      kClassifyOutputClasses, &signature_def);
  internal::AddOutputToSignatureDef(
      classification_signature.scores().tensor_name(), tensor_name_to_dtype,
      kClassifyOutputScores, &signature_def);
  return signature_def;
}

Status MaybeBuildPredictSignatureDef(
    const std::unordered_map<string, DataType>& tensor_name_to_dtype,
    MetaGraphDef* meta_graph_def) {
  Signature input_signature, output_signature;
  // Ensure that named signatures corresponding to `inputs` and `outputs` keys
  // exist.
  if (!GetNamedSignature(kPredictInputs, *meta_graph_def, &input_signature)
           .ok() ||
      !GetNamedSignature(kPredictOutputs, *meta_graph_def, &output_signature)
           .ok()) {
    return Status(error::Code::INVALID_ARGUMENT,
                  "Named signatures can only be up-converted if entries "
                  "corresponding to both `inputs` and `outputs` exist.");
  }
  // Ensure the `inputs` and `outputs` named signatures are generic signatures.
  if (input_signature.type_case() != Signature::TypeCase::kGenericSignature ||
      output_signature.type_case() != Signature::TypeCase::kGenericSignature) {
    return Status(error::Code::INVALID_ARGUMENT,
                  "Named signatures corresponding to `inputs` and `outputs` "
                  "can only be up-converted if they are GenericSignatures.");
  }
  SignatureDef signature_def;
  signature_def.set_method_name(kPredictMethodName);
  // Add map entries from the `inputs` generic signature to the input map in the
  // signature def.
  for (const auto& map_entry : input_signature.generic_signature().map()) {
    internal::AddInputToSignatureDef(map_entry.second.tensor_name(),
                                     tensor_name_to_dtype, map_entry.first,
                                     &signature_def);
  }
  // Add map entries from the `outputs` generic signature to the output map in
  // the signature def.
  for (const auto& map_entry : output_signature.generic_signature().map()) {
    internal::AddOutputToSignatureDef(map_entry.second.tensor_name(),
                                      tensor_name_to_dtype, map_entry.first,
                                      &signature_def);
  }
  // Add the constructed signature def to the signature def map of the meta
  // graph def. Use the default key if it isn't already in use.
  const bool already_has_default_signature =
      meta_graph_def->signature_def().find(kDefaultServingSignatureDefKey) !=
      meta_graph_def->signature_def().end();
  const string signature_def_key =
      already_has_default_signature
          ? strings::StrCat(kDefaultServingSignatureDefKey, "_from_named")
          : kDefaultServingSignatureDefKey;
  (*meta_graph_def->mutable_signature_def())[signature_def_key] = signature_def;
  return Status::OK();
}

Status LoadSavedModelFromLegacySessionBundlePath(
    const SessionOptions& session_options, const RunOptions& run_options,
    const StringPiece session_bundle_export_dir,
    SavedModelBundle* saved_model_bundle) {
  if (session_bundle_export_dir.empty()) {
    return Status(error::Code::NOT_FOUND, "Export directory path is empty.");
  }
  if (!IsPossibleExportDirectory(session_bundle_export_dir)) {
    return Status(
        error::Code::NOT_FOUND,
        "Export directory does not contain a valid SessionBundle export.");
  }

  // Build the session-bundle.
  SessionBundle session_bundle;
  TF_RETURN_IF_ERROR(LoadSessionBundleFromPathUsingRunOptions(
      session_options, run_options, session_bundle_export_dir,
      &session_bundle));

  // Convert the session-bundle to a saved-model-bundle.
  return internal::ConvertSessionBundleToSavedModelBundle(session_bundle,
                                                          saved_model_bundle);
}

///////////////////////////////////////////////////////////////////////////////
// Helper functions to convert `Default` and `Named` signatures to
// SignatureDefs.

// Up-conversion of default signatures is supported for classification and
// regression.
Status ConvertDefaultSignatureToSignatureDef(
    const Signatures& signatures,
    const std::unordered_map<string, DataType>& tensor_name_to_dtype,
    MetaGraphDef* meta_graph_def) {
  if (!signatures.has_default_signature()) {
    return Status::OK();
  }
  const bool already_has_default_signature =
      meta_graph_def->signature_def().find(kDefaultServingSignatureDefKey) !=
      meta_graph_def->signature_def().end();
  if (already_has_default_signature) {
    return Status(error::Code::ALREADY_EXISTS,
                  strings::StrCat(
                      "Default signature cannot be up-converted since ",
                      kDefaultServingSignatureDefKey, " key already exists."));
  }
  const Signature& signature = signatures.default_signature();
  if (IsRegressionSignature(signature)) {
    (*meta_graph_def->mutable_signature_def())[kDefaultServingSignatureDefKey] =
        BuildRegressionSignatureDef(signature.regression_signature(),
                                    tensor_name_to_dtype);
  } else if (IsClassificationSignature(signature)) {
    (*meta_graph_def->mutable_signature_def())[kDefaultServingSignatureDefKey] =
        BuildClassificationSignatureDef(signature.classification_signature(),
                                        tensor_name_to_dtype);
  } else {
    LOG(WARNING) << "Default signature up-conversion to SignatureDef is only "
                    "supported for `Classification` and `Regression`. Could "
                    "not up-convert signature: "
                 << signature.DebugString()
                 << ". (If using SessionRun with the SessionBundle export "
                    "format please ignore this warning.)";
  }
  return Status::OK();
}

Status ConvertNamedSignaturesToSignatureDef(
    const Signatures& signatures,
    const std::unordered_map<string, DataType>& tensor_name_to_dtype,
    MetaGraphDef* meta_graph_def) {
  if (signatures.named_signatures().empty()) {
    return Status::OK();
  }
  // Check for a Predict signature for up-conversion.
  Status predict_signature_def_status =
      MaybeBuildPredictSignatureDef(tensor_name_to_dtype, meta_graph_def);
  for (const auto& it_named_signature : signatures.named_signatures()) {
    const string key = it_named_signature.first;
    // If a Predict SignatureDef was successfully constructed, skip the entries
    // corresponding to `inputs` and `outputs`.
    if (predict_signature_def_status.ok()) {
      if (key == kPredictInputs || key == kPredictOutputs) {
        continue;
      }
    }
    const Signature signature = it_named_signature.second;
    if (IsRegressionSignature(signature)) {
      (*meta_graph_def->mutable_signature_def())[key] =
          BuildRegressionSignatureDef(signature.regression_signature(),
                                      tensor_name_to_dtype);
    } else if (IsClassificationSignature(signature)) {
      (*meta_graph_def->mutable_signature_def())[key] =
          BuildClassificationSignatureDef(signature.classification_signature(),
                                          tensor_name_to_dtype);
    } else {
      LOG(WARNING)
          << "Named signature up-conversion to SignatureDef is only supported "
             "for `Classification`, `Regression` or if two `GenericSignatures` "
             "signatures  called `inputs` and `outputs` exist, corresponding "
             "to the `Prediction` API. Could not up-convert signature: "
          << signature.DebugString();
    }
  }
  return Status::OK();
}

}  // namespace

namespace internal {
///////////////////////////////////////////////////////////////////////////////
// Helper functions to populate SignatureDef fields.

// Adds an entry to the `inputs` map of the supplied SignatureDef.
void AddInputToSignatureDef(
    const string& tensor_name,
    const std::unordered_map<string, DataType>& tensor_name_to_dtype,
    const string& input_key, SignatureDef* signature_def) {
  if (tensor_name.empty()) {
    LOG(WARNING) << "Tensor name not provided. Cannot add TensorInfo to "
                    "SignatureDef inputs.";
    return;
  }
  // Extract the tensor-name in case the supplied string is a tensor-reference.
  // Example: Extract "x" from "x:0".
  std::size_t pos = tensor_name.find(":");
  const string key =
      (pos != std::string::npos) ? tensor_name.substr(0, pos) : tensor_name;
  const auto it_tensor_info = tensor_name_to_dtype.find(key);
  TensorInfo tensor_info;
  tensor_info.set_name(tensor_name);
  if (it_tensor_info != tensor_name_to_dtype.end()) {
    tensor_info.set_dtype(it_tensor_info->second);
  } else {
    LOG(WARNING)
        << "No dtype found for tensor with name: " << tensor_name << ". "
        << "Building TensorInfo with only name for SignatureDef inputs. "
        << "Downstream functionality including validation may be "
        << "impacted.";
  }
  (*signature_def->mutable_inputs())[input_key] = tensor_info;
}

// Adds an entry to the `outputs` map of the supplied SignatureDef.
void AddOutputToSignatureDef(
    const string& tensor_name,
    const std::unordered_map<string, DataType>& tensor_name_to_dtype,
    const string& output_key, SignatureDef* signature_def) {
  if (tensor_name.empty()) {
    LOG(WARNING) << "Tensor name not provided. Cannot add TensorInfo to "
                    "SignatureDef outputs.";
    return;
  }
  // Extract the tensor-name in case the supplied string is a tensor-reference.
  // Example: Extract "x" from "x:0".
  std::size_t pos = tensor_name.find(":");
  const string key =
      (pos != std::string::npos) ? tensor_name.substr(0, pos) : tensor_name;
  const auto it_tensor_info = tensor_name_to_dtype.find(key);
  TensorInfo tensor_info;
  tensor_info.set_name(tensor_name);
  if (it_tensor_info != tensor_name_to_dtype.end()) {
    tensor_info.set_dtype(it_tensor_info->second);
  } else {
    LOG(WARNING)
        << "No dtype found for tensor with name: " << tensor_name << ". "
        << "Building TensorInfo with only name for SignatureDef outputs."
        << " Downstream functionality including validation may be "
        << "impacted.";
  }
  (*signature_def->mutable_outputs())[output_key] = tensor_info;
}

// Builds a map from tensor name to the corresponding datatype, by parsing the
// MetaGraphDef.
Status BuildTensorNameToDtypeMap(
    const MetaGraphDef& meta_graph_def,
    std::unordered_map<string, DataType>* tensor_name_to_dtype) {
  GraphConstructorOptions opts;
  Graph graph(OpRegistry::Global());
  TF_RETURN_IF_ERROR(
      ConvertGraphDefToGraph(opts, meta_graph_def.graph_def(), &graph));
  for (Node* node : graph.nodes()) {
    for (auto dt : node->output_types()) {
      tensor_name_to_dtype->insert(std::make_pair(node->name(), dt));
    }
  }
  return Status::OK();
}

// Converts SessionBundle signatures to SavedModel signature-defs.
Status ConvertSignaturesToSignatureDefs(MetaGraphDef* meta_graph_def) {
  Signatures signatures;
  GetSignatures(*meta_graph_def, &signatures).IgnoreError();

  // Build a map of tensor-names to the corresponding tensor-info with `name`
  // and `dtype` fields.
  std::unordered_map<string, DataType> tensor_name_to_dtype;
  TF_RETURN_IF_ERROR(
      BuildTensorNameToDtypeMap(*meta_graph_def, &tensor_name_to_dtype));

  TF_RETURN_IF_ERROR(ConvertDefaultSignatureToSignatureDef(
      signatures, tensor_name_to_dtype, meta_graph_def));
  TF_RETURN_IF_ERROR(ConvertNamedSignaturesToSignatureDef(
      signatures, tensor_name_to_dtype, meta_graph_def));
  return Status::OK();
}

// Converts a SessionBundle to a SavedModelBundle.
Status ConvertSessionBundleToSavedModelBundle(
    SessionBundle& session_bundle, SavedModelBundle* saved_model_bundle) {
  // Transfer ownership of the session from old to new.
  saved_model_bundle->session = std::move(session_bundle.session);

  // Copy the meta graph def from the SessionBundle to the SavedModelBundle.
  saved_model_bundle->meta_graph_def = session_bundle.meta_graph_def;

  // Convert signatures from session-bundle to signature-defs in
  // saved-model-bundle.
  return internal::ConvertSignaturesToSignatureDefs(
      &saved_model_bundle->meta_graph_def);
}

}  // namespace internal

Status LoadSessionBundleOrSavedModelBundle(
    const SessionOptions& session_options, const RunOptions& run_options,
    const string& export_dir,
    const std::unordered_set<string>& saved_model_tags,
    SavedModelBundle* saved_model_bundle) {
  if (MaybeSavedModelDirectory(export_dir)) {
    LOG(INFO)
        << "Attempting to load native SavedModelBundle in bundle-shim from: "
        << export_dir;
    return LoadSavedModel(session_options, run_options, export_dir,
                          saved_model_tags, saved_model_bundle);
  } else if (IsPossibleExportDirectory(export_dir)) {
    LOG(ERROR) << "Found possible SessionBundle in export directory. "
                  "SessionBundle is deprecated. Use SavedModel instead.";
    LOG(INFO) << "Attempting to up-convert SessionBundle to SavedModelBundle "
                 "in bundle-shim from: "
              << export_dir;
    return LoadSavedModelFromLegacySessionBundlePath(
        session_options, run_options, export_dir, saved_model_bundle);
  }
  return Status(
      error::Code::NOT_FOUND,
      strings::StrCat(
          "Specified file path does not appear to contain a:\n"
          "- Session bundle (should have a file called `export.meta`)\n"
          "- or, SavedModel bundle (should have a file called "
          "`saved_model.pb`)\n"
          "Specified file path: ",
          export_dir));
}

}  // namespace serving
}  // namespace tensorflow
