/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_EXPERIMENTAL_UTILS_MODEL_OPTIM_H_
#define TENSORFLOW_COMPILER_TF2TENSORRT_EXPERIMENTAL_UTILS_MODEL_OPTIM_H_

#include <string>
#include <vector>

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/compiler/tf2tensorrt/experimental/trt_convert_api.h"
#include "tensorflow/core/grappler/optimizers/meta_optimizer.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/public/session.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT

namespace tensorflow {
namespace tensorrt {

// Run the Grappler meta optimizer on the given MetaGraphDef.
Status RunGrappler(const MetaGraphDef& meta_graph_def,
                   const std::vector<std::string>& input_names,
                   const std::vector<std::string>& output_names,
                   const ConfigProto& config_proto, GraphDef* out_graph_def);

// Apply an inlining optimization to the function's graph definition.
Status ApplyInlining(MetaGraphDef& meta_graph_def,
                     const std::string& saved_model_dir,
                     const std::vector<std::string>& input_names,
                     const std::vector<std::string>& output_names,
                     std::unique_ptr<Session>* session);

// Annotates variable operations with custom `_shape` attribute.
// This is required for the converters and shape inference. The graph
// definition is modified in-place.
Status AnnotateVariableOps(GraphDef* graph_def);

// Load a SavedModel from the given path. Additionally attempts to rename
// input node names to match the requested signature.
Status LoadSavedModel(const std::string& model_dir,
                      const std::string& signature_key,
                      const std::unordered_set<std::string>& tags,
                      SavedModelBundle* bundle,
                      std::vector<std::string>* input_names,
                      std::vector<std::string>* output_names);

// Returns a RewriterConfig proto for TRT transformation.
Status GetTrtRewriterConfig(const TrtConversionParams& params,
                            RewriterConfig* opt_config,
                            bool disable_non_trt_optimizers);

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT

#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_EXPERIMENTAL_UTILS_MODEL_OPTIM_H_