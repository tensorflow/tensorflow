/* Copyright 2020 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_TPU_TPU_API_H_
#define XLA_STREAM_EXECUTOR_TPU_TPU_API_H_

#include "xla/stream_executor/tpu/libtftpu.h"
#include "xla/stream_executor/tpu/tpu_executor_api.h"
#include "xla/stream_executor/tpu/tpu_ops_c_api.h"
#include "xla/stream_executor/tpu/tpu_profiler_c_api.h"

namespace stream_executor {
namespace tpu {

TfTpu_BaseFn* InitializeApiFn();

const TfTpu_OpsApiFn* OpsApiFn();

const TfTpu_ProfilerApiFn* ProfilerApiFn();

}  // namespace tpu
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_TPU_TPU_API_H_
