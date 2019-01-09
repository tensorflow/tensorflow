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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_STREAM_EXECUTOR_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_STREAM_EXECUTOR_UTIL_H_

#include "tensorflow/compiler/xla/layout.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

// Helper functions for interacting with StreamExecutor.

namespace xla {
namespace gpu {

// Returns true if the given StreamExecutor is for a Volta or newer nvidia GPU.
bool IsVoltaOrLater(const se::StreamExecutor& stream_exec);

// Returns (input, filter, output) XLA Layout protos given the StreamExecutor
// layouts.
StatusOr<std::tuple<Layout, Layout, Layout>>
StreamExecutorConvLayoutsToXlaLayouts(const ConvolutionDimensionNumbers& dnums,
                                      se::dnn::DataLayout input,
                                      se::dnn::FilterLayout filter,
                                      se::dnn::DataLayout output);

// Returns (input, filter, output) StreamExecutor layouts given the XLA layouts.
StatusOr<
    std::tuple<se::dnn::DataLayout, se::dnn::FilterLayout, se::dnn::DataLayout>>
XlaConvLayoutsToStreamExecutorLayouts(const ConvolutionDimensionNumbers& dnums,
                                      const Layout& input, const Layout& filter,
                                      const Layout& output);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_STREAM_EXECUTOR_UTIL_H_
