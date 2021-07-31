/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CC_SAVED_MODEL_METRICS_H_
#define TENSORFLOW_CC_SAVED_MODEL_METRICS_H_
#include <string>

#include "tensorflow/core/lib/monitoring/counter.h"

namespace tensorflow {
namespace metrics {

// Returns "/tensorflow/core/saved_model/write/count" cell. This metric
// has 1 field "write_version", which is equal to the
// `tensorflow::libexport::GetWriteVersion` of the protobuf and should be
// incremented when a SavedModel has been successfully written.
monitoring::CounterCell& SavedModelWrite(const std::string& write_version);

// Returns "/tensorflow/core/saved_model/read/count" cell. This metric
// has 1 field "write_version", which is equal to the
// `tensorflow::libexport::GetWriteVersion` of the protobuf, and should be
// incremented when a SavedModel has been successfully read.
monitoring::CounterCell& SavedModelRead(const std::string& write_version);

// Returns "/tensorflow/core/saved_model/write/api" cell. This metric has 1
// field "api_label" which corresponds to a SavedModel write API. The cell for
// `foo` should be incremented when the write API `foo` is called.
monitoring::CounterCell& SavedModelWriteApi(const std::string& api_label);

// Returns "/tensorflow/core/saved_model/read/api" cell. This metric has 1
// field "api_label" which corresponds to a SavedModel read API. The cell for
// `foo` should be incremented when the read API `foo` is called.
monitoring::CounterCell& SavedModelReadApi(const std::string& api_label);

}  // namespace metrics
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_SAVED_MODEL_METRICS_H_
