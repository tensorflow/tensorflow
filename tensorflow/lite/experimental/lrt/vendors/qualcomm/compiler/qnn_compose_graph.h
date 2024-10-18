// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LRT_VENDORS_QUALCOMM_COMPILER_QNN_COMPOSE_GRAPH_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LRT_VENDORS_QUALCOMM_COMPILER_QNN_COMPOSE_GRAPH_H_

#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_model.h"
#include "tensorflow/lite/experimental/lrt/vendors/qualcomm/qnn_manager.h"

namespace lrt::qnn {

// Composes a new QNN Graph from given Lrt Graph. Qnn Graph is written to
// context behind "qnn". Uses given graph_name to name entry point.
LrtStatus ComposeGraph(QnnManager& qnn, LrtSubgraph subgraph,
                       absl::string_view qnn_graph_name);

}  // namespace lrt::qnn

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LRT_VENDORS_QUALCOMM_COMPILER_QNN_COMPOSE_GRAPH_H_
