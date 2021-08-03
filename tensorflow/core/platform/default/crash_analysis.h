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

#ifndef TENSORFLOW_CORE_PLATFORM_DEFAULT_CRASH_ANALYSIS_H_
#define TENSORFLOW_CORE_PLATFORM_DEFAULT_CRASH_ANALYSIS_H_

#include <string>

#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
namespace crash_analysis {

class BufferedDataSource {};

BufferedDataSource* ReportProtoDataOnCrash(const std::string& file_name,
                                           const protobuf::Message& message);

void RemoveReportData(const BufferedDataSource* data_source);

}  // namespace crash_analysis
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_DEFAULT_CRASH_ANALYSIS_H_
