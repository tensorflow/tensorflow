/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_GRAPH_TEST_UTILS_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_GRAPH_TEST_UTILS_H_

#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/node_def.pb.h"

namespace tensorflow {
namespace grappler {
namespace graph_tests_utils {

NodeDef MakeMapNode(absl::string_view name, absl::string_view input_node_name,
                    absl::string_view function_name = "XTimesTwo");

NodeDef MakeFilterNode(absl::string_view name,
                       absl::string_view input_node_name,
                       absl::string_view function_name = "IsZero");

NodeDef MakeMapAndBatchNode(absl::string_view name,
                            absl::string_view input_node_name,
                            absl::string_view batch_size_node_name,
                            absl::string_view num_parallel_calls_node_name,
                            absl::string_view drop_remainder_node_name,
                            absl::string_view function_name = "XTimesTwo");

}  // end namespace graph_tests_utils
}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_GRAPH_TEST_UTILS_H_
