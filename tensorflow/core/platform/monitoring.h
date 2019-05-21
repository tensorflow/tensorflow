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

#ifndef TENSORFLOW_CORE_PLATFORM_MONITORING_H_
#define TENSORFLOW_CORE_PLATFORM_MONITORING_H_

namespace tensorflow {
namespace monitoring {

// Starts exporting metrics through a platform-specific monitoring API (if
// provided). For builds using "tensorflow/core/platform/default", this is
// currently a no-op. This function is idempotent.
//
// The TensorFlow runtime will call this the first time a new session is created
// using the NewSession() method or an Eager Context is created.
void StartExporter();

// Manually invokes a one time metrics export through a platform-specific
// monitoring API (if provided). For builds using
// "tensorflow/core/platform/default", this is currently a no-op.
void ExportMetrics();

}  // namespace monitoring
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_MONITORING_H_
