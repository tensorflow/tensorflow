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

#include "tensorflow/compiler/plugin/poplar/driver/executable.h"

namespace se = ::perftools::gputools;
namespace sep = ::perftools::gputools::poplarplugin;

namespace xla {
namespace poplarplugin {

PoplarExecutable::PoplarExecutable(
        std::unique_ptr<HloModule> hlo_module,
        std::shared_ptr<poplar::Engine> engine,
        const sep::OutputMap& output_map,
        const sep::ConversionList& input_convertors,
        const sep::ConversionList& output_convertors)
    : Executable(std::move(hlo_module)),
      poplar_engine_(std::move(engine)),
      output_map_(std::move(output_map)),
      input_convertors_(std::move(input_convertors)),
      output_convertors_(std::move(output_convertors)) {}

PoplarExecutable::~PoplarExecutable() {}

StatusOr<se::DeviceMemoryBase>
PoplarExecutable::ExecuteOnStream(
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
  sep::PoplarExecutor* poplarExecutor(
          static_cast<sep::PoplarExecutor*>(executor->implementation()));

  perftools::gputools::DeviceMemoryBase retbuf;
  TF_ASSIGN_OR_RETURN(retbuf,
                      poplarExecutor->ExecuteEngine(poplar_engine_,
                                                    result_shape(),
                                                    arguments,
                                                    output_map_,
                                                    input_convertors_,
                                                    output_convertors_));

  uint64 end_micros = tensorflow::Env::Default()->NowMicros();

  {
    tensorflow::mutex_lock lock(mutex_);
    const double nanoseconds = (end_micros - start_micros) * 1000.0;
    execution_profile_.set_compute_time_ns(std::max(nanoseconds, 1.0));
  }

  return retbuf;
}

StatusOr<std::unique_ptr<ShapedBuffer>> PoplarExecutable::ExecuteOnStream(
    const ServiceExecutableRunOptions* run_options,
    tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments,
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

std::unique_ptr<HloCostAnalysis> PoplarExecutable::CreateCostAnalysis()
    const {
  return MakeUnique<HloCostAnalysis>(ShapeSizeBytes);
}

/*static*/ int64 PoplarExecutable::ShapeSizeBytes(const Shape& shape) {
  if (ShapeUtil::IsOpaque(shape)) {
    return sizeof(void*);
  }
  return ShapeUtil::ByteSizeOf(shape, sizeof(void*));
}

}  // namespace poplarplugin
}  // namespace xla

