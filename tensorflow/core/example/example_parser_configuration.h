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

#ifndef TENSORFLOW_CORE_EXAMPLE_EXAMPLE_PARSER_CONFIGURATION_H_
#define TENSORFLOW_CORE_EXAMPLE_EXAMPLE_PARSER_CONFIGURATION_H_

#include <string>
#include <vector>

#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/example_parser_configuration.pb.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/example_proto_helper.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

// This is a set of helper methods that will make it possible to share
// tensorflow::Example proto Tensor conversion code inside the ExampleParserOp
// OpKernel as well as in external code.
namespace tensorflow {

// Given a graph and the node_name of a ParseExample op,
// extract the FixedLenFeature/VarLenFeature configurations.
Status ExtractExampleParserConfiguration(
    const tensorflow::GraphDef& graph, const string& node_name,
    tensorflow::Session* session,
    std::vector<FixedLenFeature>* fixed_len_features,
    std::vector<VarLenFeature>* var_len_features);

// Given a config proto, ostensibly extracted via python,
// fill a vector of C++ structs suitable for calling
// the tensorflow.Example -> Tensor conversion code.
Status ExampleParserConfigurationProtoToFeatureVectors(
    const ExampleParserConfiguration& config_proto,
    std::vector<FixedLenFeature>* fixed_len_features,
    std::vector<VarLenFeature>* var_len_features);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_EXAMPLE_EXAMPLE_PARSER_CONFIGURATION_H_
