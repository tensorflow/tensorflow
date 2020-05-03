/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_TOOLS_GRAPH_TRANSFORMS_TRANSFORM_GRAPH_H_
#define TENSORFLOW_TOOLS_GRAPH_TRANSFORMS_TRANSFORM_GRAPH_H_

#include <vector>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

// Convenience function to handle argument parsing for the command line tool.
// If init_main is false, we're testing so don't call core initialization.
int ParseFlagsAndTransformGraph(int argc, char* argv[], bool init_main);

// Handles converting the transforms string into transform names and their
// arguments.
typedef std::vector<std::pair<string, TransformFuncParameters>>
    TransformParameters;
Status ParseTransformParameters(const string& transforms_string,
                                TransformParameters* params_list);

// Applies a series of transformations to the GraphDef. These transforms are
// defined by modules that call REGISTER_GRAPH_TRANSFORM() to associate a
// function with a name string.
Status TransformGraph(const std::vector<string>& inputs,
                      const std::vector<string>& outputs,
                      const TransformParameters& transform_params,
                      GraphDef* graph_def);

}  // namespace graph_transforms
}  // namespace tensorflow

#endif  // TENSORFLOW_TOOLS_GRAPH_TRANSFORMS_TRANSFORM_GRAPH_H_
