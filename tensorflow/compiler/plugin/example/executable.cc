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

#include "tensorflow/compiler/plugin/example/executable.h"
#include "tensorflow/compiler/plugin/example/executor.h"

namespace se = ::perftools::gputools;
namespace sep = ::perftools::gputools::exampleplugin;

namespace xla {
namespace exampleplugin {

ExampleExecutable::ExampleExecutable(
        std::unique_ptr<HloModule> hlo_module,
        std::unique_ptr<HloModuleConfig> module_config)
    : Executable(std::move(hlo_module),
                 std::move(module_config)) {
}

ExampleExecutable::~ExampleExecutable() {}

StatusOr<se::DeviceMemoryBase>
ExampleExecutable::ExecuteOnStream(
    const ServiceExecutableRunOptions* run_options,
    tensorflow::gtl::ArraySlice<se::DeviceMemoryBase> arguments,
    HloExecutionProfile* hlo_execution_profile) {
  se::Stream* stream = run_options->stream();

  VLOG(1) << "Execute " << module().name();
  if (VLOG_IS_ON(2)) {
    for (const auto& a : arguments) {
      VLOG(2) << "-- argument " << a.opaque();
    }
  }

  uint64 start_micros = tensorflow::Env::Default()->NowMicros();

  perftools::gputools::StreamExecutor* executor(stream->parent());
  sep::ExampleExecutor* exampleExecutor(
          static_cast<sep::ExampleExecutor*>(executor->implementation()));

  perftools::gputools::DeviceMemoryBase retbuf;
  TF_ASSIGN_OR_RETURN(retbuf,
                      exampleExecutor->ExecuteGraph(result_shape(), arguments));

  uint64 end_micros = tensorflow::Env::Default()->NowMicros();

  {
    tensorflow::mutex_lock lock(mutex_);
    const double nanoseconds = (end_micros - start_micros) * 1000.0;
    execution_profile_.set_compute_time_ns(std::max(nanoseconds, 1.0));
  }

  return retbuf;
}

StatusOr<std::unique_ptr<ShapedBuffer>> ExampleExecutable::ExecuteOnStream(
    const ServiceExecutableRunOptions* run_options,
    tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments,
    HloExecutionProfile* hlo_execution_profile) {
  return tensorflow::errors::Unimplemented(
          "ExecuteOnStream is not yet supported on Example.");
}

StatusOr<se::DeviceMemoryBase>
ExampleExecutable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    tensorflow::gtl::ArraySlice<se::DeviceMemoryBase> arguments) {
  return tensorflow::errors::Unimplemented(
          "ExecuteAsyncOnStream is not yet supported on Example.");
}

}  // namespace exampleplugin
}  // namespace xla

