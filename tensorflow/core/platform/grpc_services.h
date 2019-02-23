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
#ifndef TENSORFLOW_CORE_PLATFORM_GRPC_SERVICES_H_
#define TENSORFLOW_CORE_PLATFORM_GRPC_SERVICES_H_

#include "tensorflow/core/profiler/profiler_analysis.grpc.pb.h"
#include "tensorflow/core/profiler/profiler_service.grpc.pb.h"

#if !defined(PLATFORM_GOOGLE)

namespace tensorflow {
namespace grpc {

// Google internal GRPC generates services under namespace "tensorflow::grpc".
// Creating aliases here to make sure we can access services under namespace
// "tensorflow::grpc" both in google internal and open-source.
using ::tensorflow::ProfileAnalysis;
using ::tensorflow::ProfilerService;

}  // namespace grpc
}  // namespace tensorflow
#endif

#endif  // TENSORFLOW_CORE_PLATFORM_GRPC_SERVICES_H_
