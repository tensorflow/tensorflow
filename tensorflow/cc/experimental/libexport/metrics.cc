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

#include "tensorflow/cc/experimental/libexport/metrics.h"

#include <string>

#include "tensorflow/core/lib/monitoring/counter.h"

namespace tensorflow {
namespace libexport {
namespace metrics {

namespace {

// Counter that tracks total number of SavedModels written.
auto* saved_model_write_counter = monitoring::Counter<0>::New(
    "/tensorflow/core/saved_model/write/count",
    "The number of SavedModels successfully written.");

// Counter that tracks total number of SavedModels read.
auto* saved_model_read_counter = monitoring::Counter<0>::New(
    "/tensorflow/core/saved_model/read/count",
    "The number of SavedModels successfully loaded.");

// Counter that tracks number of calls for each SavedModel write API. Summing
// across "api_label" is not expected to equal the ".../write/count" cell value
// because programs can invoke more than one API to save a single SM and
// because the API may error out before successfully writing a SM.
auto* saved_model_write_api = monitoring::Counter<2>::New(
    "/tensorflow/core/saved_model/write/api",
    "The API used to write the SavedModel.", "api_label", "write_version");

// Counter that tracks number of calls for each SavedModel read API. Summing
// across "api_label" is not expected to equal the ".../read/count" cell value
// because programs can invoke more than one API to load a single SM and
// because the API may error out before successfully reading a SM.
auto* saved_model_read_api = monitoring::Counter<2>::New(
    "/tensorflow/core/saved_model/read/api",
    "The API used to load the SavedModel.", "api_label", "write_version");

}  // namespace

monitoring::CounterCell& Write() {
  return *saved_model_write_counter->GetCell();
}

monitoring::CounterCell& Read() { return *saved_model_read_counter->GetCell(); }

monitoring::CounterCell& WriteApi(const std::string& api_label,
                                  const std::string& write_version) {
  return *saved_model_write_api->GetCell(api_label, write_version);
}

monitoring::CounterCell& ReadApi(const std::string& api_label,
                                 const std::string& write_version) {
  return *saved_model_read_api->GetCell(api_label, write_version);
}

}  // namespace metrics
}  // namespace libexport
}  // namespace tensorflow
