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
#ifndef TENSORFLOW_CC_EXPERIMENTAL_LIBEXPORT_METRICS_H_
#define TENSORFLOW_CC_EXPERIMENTAL_LIBEXPORT_METRICS_H_
#include <string>

#include "tensorflow/core/lib/monitoring/counter.h"

namespace tensorflow {
namespace libexport {
namespace metrics {

// Returns "/tensorflow/core/saved_model/write/count" cell. This metric
// has 0 fields and should be incremented when a SavedModel has been
// successfully written.
monitoring::CounterCell& Write();

// Returns "/tensorflow/core/saved_model/read/count" cell. This metric
// has 0 fields and should be incremented when a SavedModel has been
// successfully read.
monitoring::CounterCell& Read();

// Returns "/tensorflow/core/saved_model/write/api" cell belonging to fields
// (api_label, write_version). The "api_label" field corresponds to a SavedModel
// write API, and the "write_version" field is equal to
// tensorflow::libexport::GetWriteVersion of the protobuf. The cell for (foo, #)
// should be incremented when the write API `foo` is called.
monitoring::CounterCell& WriteApi(const std::string& api_label,
                                  const std::string& write_version);

// Returns "/tensorflow/core/saved_model/read/api" cell belonging to fields
// (api_label, write_version). The "api_label" field corresponds to a SavedModel
// read API, and the "write_version" field is equal to
// tensorflow::libexport::GetWriteVersion of the protobuf. The cell for (foo, #)
// should be incremented when the read API `foo` is called.
monitoring::CounterCell& ReadApi(const std::string& api_label,
                                 const std::string& write_version);

}  // namespace metrics
}  // namespace libexport
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_EXPERIMENTAL_LIBEXPORT_METRICS_H_
