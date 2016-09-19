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

// Helpers for working with TensorFlow exports and their signatures.

#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_SESSION_BUNDLE_SIGNATURE_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_SESSION_BUNDLE_SIGNATURE_H_

#include <string>
#include <utility>
#include <vector>

#include "tensorflow/contrib/session_bundle/manifest.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saver.pb.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace serving {

const char kSignaturesKey[] = "serving_signatures";

// Get Signatures from a MetaGraphDef.
Status GetSignatures(const tensorflow::MetaGraphDef& meta_graph_def,
                     Signatures* signatures);

// (Re)set Signatures in a MetaGraphDef.
Status SetSignatures(const Signatures& signatures,
                     tensorflow::MetaGraphDef* meta_graph_def);

// Gets a ClassificationSignature from a MetaGraphDef's default signature.
// Returns an error if the default signature is not a ClassificationSignature,
// or does not exist.
Status GetClassificationSignature(
    const tensorflow::MetaGraphDef& meta_graph_def,
    ClassificationSignature* signature);

// Gets a named ClassificationSignature from a MetaGraphDef.
// Returns an error if a ClassificationSignature with the given name does
// not exist.
Status GetNamedClassificationSignature(
    const string& name, const tensorflow::MetaGraphDef& meta_graph_def,
    ClassificationSignature* signature);

// Gets a RegressionSignature from a MetaGraphDef's default signature.
// Returns an error if the default signature is not a RegressionSignature,
// or does not exist.
Status GetRegressionSignature(const tensorflow::MetaGraphDef& meta_graph_def,
                              RegressionSignature* signature);

// Runs a classification using the provided signature and initialized Session.
//   input: input batch of items to classify
//   classes: output batch of classes; may be null if not needed
//   scores: output batch of scores; may be null if not needed
// Validates sizes of the inputs and outputs are consistent (e.g., input
// batch size equals output batch sizes).
// Does not do any type validation.
Status RunClassification(const ClassificationSignature& signature,
                         const Tensor& input, Session* session, Tensor* classes,
                         Tensor* scores);

// Runs regression using the provided signature and initialized Session.
//   input: input batch of items to run the regression model against
//   output: output targets
// Validates sizes of the inputs and outputs are consistent (e.g., input
// batch size equals output batch sizes).
// Does not do any type validation.
Status RunRegression(const RegressionSignature& signature, const Tensor& input,
                     Session* session, Tensor* output);

// Gets the named GenericSignature from a MetaGraphDef.
// Returns an error if a GenericSignature with the given name does not exist.
Status GetGenericSignature(const string& name,
                           const tensorflow::MetaGraphDef& meta_graph_def,
                           GenericSignature* signature);

// Gets the default signature from a MetaGraphDef.
Status GetDefaultSignature(const tensorflow::MetaGraphDef& meta_graph_def,
                           Signature* default_signature);

// Gets a named Signature from a MetaGraphDef.
// Returns an error if a Signature with the given name does not exist.
Status GetNamedSignature(const string& name,
                         const tensorflow::MetaGraphDef& meta_graph_def,
                         Signature* default_signature);

// Binds TensorFlow inputs specified by the caller using the logical names
// specified at Graph export time, to the actual Graph names.
// Returns an error if any of the inputs do not have a binding in the export's
// MetaGraphDef.
Status BindGenericInputs(const GenericSignature& signature,
                         const std::vector<std::pair<string, Tensor>>& inputs,
                         std::vector<std::pair<string, Tensor>>* bound_inputs);

// Binds the input names specified by the caller using the logical names
// specified at Graph export time, to the actual Graph names. This is useful
// for binding names of both the TensorFlow output tensors and target nodes,
// with the latter (target nodes) being optional and rarely used (if ever) at
// serving time.
// Returns an error if any of the input names do not have a binding in the
// export's MetaGraphDef.
Status BindGenericNames(const GenericSignature& signature,
                        const std::vector<string>& input_names,
                        std::vector<string>* bound_names);

}  // namespace serving
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_SESSION_BUNDLE_SIGNATURE_H_
