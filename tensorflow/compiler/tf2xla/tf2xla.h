/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_TF2XLA_TF2XLA_H_
#define TENSORFLOW_COMPILER_TF2XLA_TF2XLA_H_

#include "tensorflow/compiler/tf2xla/tf2xla.pb.h"
#include "tensorflow/compiler/xla/client/client.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/core/framework/graph.pb.h"

namespace tensorflow {

// Converts a tensorflow::GraphDef into an xla::XlaComputation. The given
// `config` specifies the portion of the graph to convert, via feeds and
// fetches. Each feed is a positional input argument for the generated
// computation, while each fetch is a positional output argument.
//
// The computation is built in the context of the given `client`, which may
// subsequently be used to compile or execute the computation.
Status ConvertGraphDefToXla(GraphDef graph_def, const tf2xla::Config& config,
                            xla::Client* client,
                            xla::XlaComputation* computation);

// Similar to ConvertGraphDefToXla, but uses MLIR.
Status ConvertGraphDefToXlaViaMlir(const GraphDef& graph_def,
                                   const tf2xla::Config& config,
                                   xla::XlaComputation* computation);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_TF2XLA_H_
