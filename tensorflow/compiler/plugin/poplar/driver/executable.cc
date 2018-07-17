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
#include "tensorflow/compiler/plugin/poplar/driver/executable.pb.h"

namespace xla {
namespace poplarplugin {

PoplarExecutable::PoplarExecutable(
    std::unique_ptr<HloModule> hlo_module,
    std::unique_ptr<HloProfilePrinterData> profile_printer,
    std::unique_ptr<HloProfileIndexMap> profile_index_map,
    std::shared_ptr<poplar::Engine> engine,
    const ::xla::poplarplugin::OutputMap& output_map,
    std::vector<std::unique_ptr<Literal>> literal_output,
    const std::vector<bool>& parameter_streamed,
    const std::vector<bool>& output_streamed)
    : Executable(std::move(hlo_module), std::move(profile_printer),
                 std::move(profile_index_map)),
      poplar_engine_(std::move(engine)),
      output_map_(std::move(output_map)),
      literal_output_(std::move(literal_output)),
      parameter_streamed_(std::move(parameter_streamed)),
      output_streamed_(std::move(output_streamed)),
      first_execution_(true) {}

PoplarExecutable::~PoplarExecutable() {}

StatusOr<ScopedShapedBuffer> PoplarExecutable::ExecuteOnStream(
    const ServiceExecutableRunOptions* run_options,
    tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments,
    HloExecutionProfile* hlo_execution_profile) {
  se::Stream* stream = run_options->stream();

  std::vector<se::DeviceMemoryBase> argument_buffers;
  for (int i = 0; i < arguments.size(); ++i) {
    argument_buffers.push_back(arguments[i]->buffer(/*index=*/{}));
  }

  VLOG(1) << "Execute " << module().name();
  if (VLOG_IS_ON(2)) {
    for (const auto& a : argument_buffers) {
      VLOG(2) << "-- argument " << a.opaque();
    }
  }

  uint64 start_micros = tensorflow::Env::Default()->NowMicros();

  perftools::gputools::StreamExecutor* executor(stream->parent());
  PoplarExecutor* poplarExecutor(
      static_cast<PoplarExecutor*>(executor->implementation()));

  DeviceMemoryAllocator* memory_allocator = run_options->allocator();

  se::DeviceMemoryBase result;
  TF_ASSIGN_OR_RETURN(
      result, poplarExecutor->ExecuteEngine(executor, *this, memory_allocator,
                                            argument_buffers));

  first_execution_ = false;

  uint64 end_micros = tensorflow::Env::Default()->NowMicros();

  {
    tensorflow::mutex_lock lock(mutex_);
    const double nanoseconds = (end_micros - start_micros) * 1000.0;
    execution_profile_.set_compute_time_ns(std::max(nanoseconds, 1.0));
    execution_profile_.set_compute_cycle_count(1);
  }

  ScopedShapedBuffer result_buffer(result_shape(), result_shape(),
                                   run_options->allocator(),
                                   stream->parent()->device_ordinal());

  // Copy DeviceMemoryBase values which contain the array(s) of the result into
  // the respective location in ShapedBuffer which is returned to the caller.

  TF_RETURN_IF_ERROR(result_buffer.buffers().ForEachMutableElementWithStatus(
      [&result, poplarExecutor](const ShapeIndex& index,
                                se::DeviceMemoryBase* device_memory) {
        se::DeviceMemoryBase buffer = result;
        for (auto i : index) {
          TF_ASSIGN_OR_RETURN(buffer,
                              poplarExecutor->GetTupleBufferByIndex(buffer, i));
        }
        CHECK(!buffer.is_null() || buffer.size() == 0);
        *device_memory = buffer;
        return Status::OK();
      }));

  return std::move(result_buffer);
}

StatusOr<ScopedShapedBuffer> PoplarExecutable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments) {
  return xla::Unimplemented(
      "ExecuteAsyncOnStream is not yet supported on Poplar.");
}

/*static*/ int64 PoplarExecutable::ShapeSizeBytes(const Shape& shape) {
  if (ShapeUtil::IsOpaque(shape)) {
    return sizeof(void*);
  }
  return ShapeUtil::ByteSizeOf(shape, sizeof(void*));
}

/*static*/ StatusOr<PoplarExecutable*> PoplarExecutable::Deserialize(
    std::unique_ptr<HloModule> hlo_module,
    std::unique_ptr<HloProfilePrinterData> hlo_profile_printer,
    std::unique_ptr<HloProfileIndexMap> hlo_profile_index_map,
    const std::string& filename) {
  VLOG(1) << "Restoring executable from " << filename;

  PoplarExecutableProto proto;

  // TODO expand this when poplar engine serialization support is ready
  return ReadBinaryProto(tensorflow::Env::Default(), filename, &proto);
}

/*static*/ Status PoplarExecutable::Serialize(
    const PoplarExecutable& executable, const std::string& filename) {
  PoplarExecutableProto proto;

  // TODO expand this when poplar engine serialization support is ready
  return WriteBinaryProto(tensorflow::Env::Default(), filename, proto);
}

}  // namespace poplarplugin
}  // namespace xla
