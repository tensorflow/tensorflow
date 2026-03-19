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

#ifndef XLA_TSL_PLATFORM_DEFAULT_CRASH_ANALYSIS_H_
#define XLA_TSL_PLATFORM_DEFAULT_CRASH_ANALYSIS_H_

#include <string>

#include "tsl/platform/protobuf.h"

namespace tensorflow {
namespace crash_analysis {

class BufferedDataSource {};

// Reports `message` proto which will be stored in the `file_name` in case
// of a process crash.
// Default implementation is currently NOOP.
BufferedDataSource* ReportProtoDataOnCrash(
    const std::string& file_name, const tsl::protobuf::Message& message);

// Removes `data_source` from the list of data reported in case of a process
// crash.
// Default implementation is currently NOOP.
void RemoveReportData(const BufferedDataSource* data_source);

// Reports `event_data` with the associated `message` under `event_name` to the
// crash analysis system. This does not require process crash.
// Default implementation is currently NOOP.
void ReportEvent(const std::string& event_name, const std::string& message,
                 const std::string& event_data);

}  // namespace crash_analysis
}  // namespace tensorflow

#endif  // XLA_TSL_PLATFORM_DEFAULT_CRASH_ANALYSIS_H_
