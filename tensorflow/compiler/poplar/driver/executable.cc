/* Copyright 2017 Graphcore Ltd
 */

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/poplar/driver/executable.h"
#include "tensorflow/compiler/poplar/stream_executor/executor.h"

#include <stdint.h>
#include <algorithm>
#include <set>
#include <unordered_set>
#include <utility>
#include <vector>

namespace se = ::perftools::gputools;
namespace sep = ::perftools::gputools::poplarplugin;

namespace xla {
namespace poplarplugin {

static const int SEQUENCE_PROGRAM = 0;
static const int COPY_IN_PROGRAM = 1;
static const int COPY_OUT_PROGRAM = 2;

PoplarExecutable::PoplarExecutable(
        std::unique_ptr<HloModule> hlo_module,
        std::unique_ptr<HloModuleConfig> module_config,
        std::unique_ptr<poplar::Engine> engine,
        const std::vector<char*>& inputs,
        const std::vector<char*>& outputs)
    : Executable(std::move(hlo_module),
                 std::move(module_config)),
      poplar_engine_(std::move(engine)),
      input_buffers_(inputs),
      output_buffers_(outputs) {
}

PoplarExecutable::~PoplarExecutable() {
  // TODO remove this when we don't need Copy buffers
  for (auto p : input_buffers_) {
    delete p;
  }
  for (auto p : output_buffers_) {
    delete p;
  }
}

StatusOr<se::DeviceMemoryBase>
PoplarExecutable::ExecuteOnStream(
    const ServiceExecutableRunOptions* run_options,
    tensorflow::gtl::ArraySlice<se::DeviceMemoryBase> arguments,
    HloExecutionProfile* hlo_execution_profile) {
  se::Stream* stream = run_options->stream();

  LOG(INFO) << "Execute " << module().name();

  perftools::gputools::StreamExecutor* executor(stream->parent());
  sep::PoplarExecutor* poplarExecutor(
          static_cast<sep::PoplarExecutor*>(executor->implementation()));

  // Allocate output buffer(s)
  perftools::gputools::DeviceMemoryBase retbuf;
  TF_ASSIGN_OR_RETURN(retbuf,
                      poplarExecutor->AllocateOutputBuffer(result_shape()));

  // TODO replace with future poplar Engine copy interface
  for (uint64 a = 0; a < arguments.size(); a++) {
    // Copy data from cache buffer to Copy buffer
    auto arg(arguments[a]);
    poplarExecutor->CopyDataToPoplar(&arg, input_buffers_[a]);
  }

  poplar_engine_->run(COPY_IN_PROGRAM);
  poplar_engine_->run(SEQUENCE_PROGRAM);
  poplar_engine_->run(COPY_OUT_PROGRAM);

  // Copy data back from the temp buffers to cache buffers
  poplarExecutor->CopyDataFromPoplar(result_shape(), output_buffers_, &retbuf);

  return retbuf;
}

StatusOr<std::unique_ptr<ShapedBuffer>> PoplarExecutable::ExecuteOnStream(
    const ServiceExecutableRunOptions* run_options,
    tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments,
    HloExecutionProfile* hlo_execution_profile) {
  return tensorflow::errors::Unimplemented(
          "ExecuteOnStream is not yet supported on Poplar.");
}

Status PoplarExecutable::ExecuteOnStream(
    const ServiceExecutableRunOptions* run_options,
    tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments,
    ShapedBuffer* result_buffer,
    HloExecutionProfile* hlo_execution_profile) {
  return tensorflow::errors::Unimplemented(
          "ExecuteOnStream is not yet supported on Poplar.");
}

StatusOr<se::DeviceMemoryBase>
PoplarExecutable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    tensorflow::gtl::ArraySlice<se::DeviceMemoryBase> arguments) {
  return tensorflow::errors::Unimplemented(
          "ExecuteAsyncOnStream is not yet supported on Poplar.");
}

}  // namespace poplarplugin
}  // namespace xla

