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

#include "tensorflow/contrib/session_bundle/signature.h"

#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/any.pb.h"
#include "tensorflow/contrib/session_bundle/manifest.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saver.pb.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace serving {
namespace {

// Returns OK if the input and output batch sizes match.
Status BatchSizesMatch(const Tensor& input, const Tensor& output) {
  // Ensure the number of outputs match the number of inputs.
  if (input.dim_size(0) != output.dim_size(0)) {
    return errors::Internal(
        strings::StrCat("Input batch size did not match output batch size: ",
                        input.dim_size(0), " vs. ", output.dim_size(0)));
  }
  return Status::OK();
}
}  // namespace

Status GetSignatures(const tensorflow::MetaGraphDef& meta_graph_def,
                     Signatures* signatures) {
  const auto& collection_def = meta_graph_def.collection_def();
  const auto it = collection_def.find(kSignaturesKey);
  if (it == collection_def.end() || it->second.any_list().value_size() != 1) {
    return errors::FailedPrecondition(
        strings::StrCat("Expected exactly one signatures proto in : ",
                        DebugStringIfAvailable(meta_graph_def)));
  }
  const auto& any = it->second.any_list().value(0);
  return ParseAny(any, signatures, "tensorflow.serving.Signatures");
}

Status SetSignatures(const Signatures& signatures,
                     tensorflow::MetaGraphDef* meta_graph_def) {
  auto& collection_def = *(meta_graph_def->mutable_collection_def());
  auto* any_list = collection_def[kSignaturesKey].mutable_any_list();
  any_list->mutable_value()->Clear();
#ifdef TENSORFLOW_LITE_PROTOS
  signatures.SerializeToString(
      any_list->mutable_value()->Add()->mutable_value());
#else
  any_list->mutable_value()->Add()->PackFrom(signatures);
#endif
  return Status::OK();
}

Status GetClassificationSignature(
    const tensorflow::MetaGraphDef& meta_graph_def,
    ClassificationSignature* signature) {
  Signatures signatures;
  TF_RETURN_IF_ERROR(GetSignatures(meta_graph_def, &signatures));
  if (!signatures.has_default_signature()) {
    return errors::FailedPrecondition(
        strings::StrCat("Expected a default signature in: ",
                        DebugStringIfAvailable(signatures)));
  }
  if (!signatures.default_signature().has_classification_signature()) {
    return errors::FailedPrecondition(strings::StrCat(
        "Expected a classification signature in: ",
        DebugStringIfAvailable(signatures.default_signature())));
  }
  *signature = signatures.default_signature().classification_signature();
  return Status::OK();
}

Status GetNamedClassificationSignature(
    const string& name, const tensorflow::MetaGraphDef& meta_graph_def,
    ClassificationSignature* signature) {
  Signatures signatures;
  TF_RETURN_IF_ERROR(GetSignatures(meta_graph_def, &signatures));
  const auto& it = signatures.named_signatures().find(name);
  if (it == signatures.named_signatures().end()) {
    return errors::NotFound(
        strings::StrCat("Missing signature named \"", name, "\" in: ",
                        DebugStringIfAvailable(signatures)));
  }
  if (!it->second.has_classification_signature()) {
    return errors::FailedPrecondition(
        strings::StrCat("Expected a classification signature for name \"", name,
                        "\" in: ", DebugStringIfAvailable(it->second)));
  }
  *signature = it->second.classification_signature();
  return Status::OK();
}

Status RunClassification(const ClassificationSignature& signature,
                         const Tensor& input, Session* session, Tensor* classes,
                         Tensor* scores) {
  std::vector<string> output_tensor_names;
  if (classes) {
    output_tensor_names.push_back(signature.classes().tensor_name());
  }
  if (scores) {
    output_tensor_names.push_back(signature.scores().tensor_name());
  }
  // Run the graph with our inputs and outputs.
  std::vector<Tensor> outputs;
  const Status run_status =
      session->Run({{signature.input().tensor_name(), input}},
                   output_tensor_names, {}, &outputs);
  if (!run_status.ok()) {
    return run_status;
  }
  // Ensure the output is shaped how we expect.
  // There should be one string Tensor of shape,
  //   [batch_size, num_recommendations].
  if (outputs.size() != output_tensor_names.size()) {
    return errors::Internal(
        strings::StrCat("Expected ", output_tensor_names.size(),
                        " output tensor(s).  Got: ", outputs.size()));
  }
  if (classes) {
    *classes = outputs[0];
    TF_RETURN_IF_ERROR(BatchSizesMatch(input, *classes));
  }
  if (scores) {
    *scores = outputs[classes ? 1 : 0];
    TF_RETURN_IF_ERROR(BatchSizesMatch(input, *scores));
  }
  return Status::OK();
}

Status GetRegressionSignature(const tensorflow::MetaGraphDef& meta_graph_def,
                              RegressionSignature* signature) {
  Signatures signatures;
  TF_RETURN_IF_ERROR(GetSignatures(meta_graph_def, &signatures));
  if (!signatures.has_default_signature()) {
    return errors::FailedPrecondition(
        strings::StrCat("Expected a default signature in: ",
                        DebugStringIfAvailable(signatures)));
  }
  if (!signatures.default_signature().has_regression_signature()) {
    return errors::FailedPrecondition(strings::StrCat(
        "Expected a regression signature in: ",
        DebugStringIfAvailable(signatures.default_signature())));
  }
  *signature = signatures.default_signature().regression_signature();
  return Status::OK();
}

Status RunRegression(const RegressionSignature& signature,
                     const Tensor& regression_input, Session* session,
                     Tensor* regression_output) {
  std::vector<string> output_tensor_names;
  if (regression_output) {
    output_tensor_names.push_back(signature.output().tensor_name());
  }
  // Run the graph with our inputs and outputs.
  std::vector<Tensor> outputs;
  const Status run_status =
      session->Run({{signature.input().tensor_name(), regression_input}},
                   output_tensor_names, {}, &outputs);
  if (!run_status.ok()) {
    return run_status;
  }
  // Ensure the regression score output is shaped how we expect.
  // There should be one float Tensor of shape,
  //   [batch_size, num_recommendations].
  if (outputs.size() != output_tensor_names.size()) {
    return errors::Internal(
        strings::StrCat("Expected ", output_tensor_names.size(),
                        " output tensor(s).  Got: ", outputs.size()));
  }
  if (regression_output) {
    *regression_output = outputs[0];
    TF_RETURN_IF_ERROR(BatchSizesMatch(regression_input, *regression_output));
  }
  return Status::OK();
}

Status GetGenericSignature(const string& name,
                           const tensorflow::MetaGraphDef& meta_graph_def,
                           GenericSignature* signature) {
  Signatures signatures;
  TF_RETURN_IF_ERROR(GetSignatures(meta_graph_def, &signatures));
  const auto& it = signatures.named_signatures().find(name);
  if (it == signatures.named_signatures().end()) {
    return errors::InvalidArgument(
        strings::StrCat("Missing generic signature named \"", name, "\" in ",
                        DebugStringIfAvailable(signatures)));
  }
  if (!it->second.has_generic_signature()) {
    return errors::InvalidArgument(strings::StrCat(
        "Expected a generic signature: ", DebugStringIfAvailable(it->second)));
  }
  *signature = it->second.generic_signature();
  return Status::OK();
}

Status GetDefaultSignature(const tensorflow::MetaGraphDef& meta_graph_def,
                           Signature* default_signature) {
  Signatures signatures;
  TF_RETURN_IF_ERROR(GetSignatures(meta_graph_def, &signatures));
  *default_signature = signatures.default_signature();
  return Status::OK();
}

Status GetNamedSignature(const string& name,
                         const tensorflow::MetaGraphDef& meta_graph_def,
                         Signature* signature) {
  Signatures signatures;
  TF_RETURN_IF_ERROR(GetSignatures(meta_graph_def, &signatures));
  const auto& it = signatures.named_signatures().find(name);
  if (it == signatures.named_signatures().end()) {
    return errors::NotFound(
        strings::StrCat("Missing signature named \"", name, "\" in: ",
                        DebugStringIfAvailable(signatures)));
  }
  *signature = it->second;
  return Status::OK();
}

Status BindGenericInputs(const GenericSignature& signature,
                         const std::vector<std::pair<string, Tensor>>& inputs,
                         std::vector<std::pair<string, Tensor>>* bound_inputs) {
  const protobuf::Map<string, serving::TensorBinding>& bindings =
      signature.map();

  for (const auto& entry : inputs) {
    const auto mapped = bindings.find(entry.first);
    if (mapped == bindings.end()) {
      return errors::NotFound(
          strings::StrCat("Could not find generic binding for: ", entry.first));
    }
    bound_inputs->push_back({mapped->second.tensor_name(), entry.second});
  }
  return Status::OK();
}

Status BindGenericNames(const GenericSignature& signature,
                        const std::vector<string>& input_names,
                        std::vector<string>* bound_names) {
  const protobuf::Map<string, serving::TensorBinding>& bindings =
      signature.map();

  for (const string& entry : input_names) {
    const auto mapped = bindings.find(entry);
    if (mapped == bindings.end()) {
      return errors::NotFound(
          strings::StrCat("Could not find generic binding for: ", entry));
    }
    bound_names->push_back(mapped->second.tensor_name());
  }
  return Status::OK();
}

}  // namespace serving
}  // namespace tensorflow
