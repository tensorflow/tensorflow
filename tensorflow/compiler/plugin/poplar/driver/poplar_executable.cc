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

#include "tensorflow/compiler/plugin/poplar/driver/poplar_executable.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_executable.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_platform.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/xla_ipu_common.h"

#include <fstream>

namespace xla {
namespace poplarplugin {

PoplarExecutable::PoplarExecutable(
    std::unique_ptr<HloModule> hlo_module,
    std::unique_ptr<HloProfilePrinterData> profile_printer,
    std::unique_ptr<HloProfileIndexMap> profile_index_map,
    std::unique_ptr<poplar::Engine> engine,
    const InputOutputAliasingMap& input_output_aliasing_map,
    const bool is_constant_graph,
    std::vector<std::vector<Literal>> literal_output, const bool is_remap_graph,
    std::vector<uint64> remaped_output, uint32 replication_factor,
    const InfeedInfos& infeed_infos, const OutfeedInfos& outfeed_infos)
    : Executable(std::move(hlo_module), std::move(profile_printer),
                 std::move(profile_index_map)),
      poplar_engine_(std::move(engine)),
      input_output_aliasing_map_(std::move(input_output_aliasing_map)),
      literal_output_(std::move(literal_output)),
      is_constant_graph_(is_constant_graph),
      remaped_output_(std::move(remaped_output)),
      is_remap_graph_(is_remap_graph),
      execution_count_(0),
      replication_factor_(replication_factor),
      infeed_infos_(std::move(infeed_infos)),
      outfeed_infos_(std::move(outfeed_infos)),
      loaded_from_cache_(false) {}

PoplarExecutable::~PoplarExecutable() {
  if (poplar_engine_.get() != nullptr) {
    auto platform =
        se::MultiPlatformManager::PlatformWithName(tensorflow::PLATFORM_NAME);
    if (platform.ok()) {
      auto* p = static_cast<PoplarPlatform*>(platform.ValueOrDie());
      p->AboutToFreeEngine(poplar_engine_.get());
    }
  }
}

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
  PoplarExecutor::AsPoplarStream(stream)->BlockUntilDone();
  TF_ASSIGN_OR_RETURN(
      result, poplarExecutor->ExecuteEngine(executor, *this, memory_allocator,
                                            argument_buffers));

  execution_count_++;
  if (poplarExecutor->ReportEventNthExecution() > 0 &&
      execution_count_ >= poplarExecutor->ReportEventNthExecution()) {
    execution_count_ = 0;
  }

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
        if (VLOG_IS_ON(2)) {
          VLOG(2) << "-- return " << buffer.opaque();
        }
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
  if (shape.IsOpaque()) {
    return sizeof(void*);
  }
  return ShapeUtil::ByteSizeOf(shape, sizeof(void*));
}

/*static*/ StatusOr<PoplarExecutable*> PoplarExecutable::Deserialize(
    std::unique_ptr<HloModule> hlo_module,
    std::unique_ptr<HloProfilePrinterData> profile_printer,
    std::unique_ptr<HloProfileIndexMap> profile_index_map,
    const std::string& filename) {
  PoplarExecutableProto proto;

  TF_RETURN_IF_ERROR(
      ReadBinaryProto(tensorflow::Env::Default(), filename, &proto));

  // Load metadata
  int replication_factor = proto.replication_factor();

  InfeedInfos infeeds;
  for (const auto infeed : proto.infeeds()) {
    FeedInfo info;
    info.stream_prefix = infeed.stream_prefix();
    info.config = infeed.config();
    info.shape = Shape(infeed.shape());

    infeeds.push_back(info);
  }

  OutfeedInfos outfeeds;
  for (const auto infeed : proto.outfeeds()) {
    FeedInfo info;
    info.stream_prefix = infeed.stream_prefix();
    info.config = infeed.config();
    info.shape = Shape(infeed.shape());

    outfeeds.push_back(info);
  }

  // Load the poplar compilation options from the serialized executable
  poplar::OptionFlags opts;
  for (const auto flag : proto.option_flags()) {
    opts.set(flag.key(), flag.value());
  }

  // Load the executable
  std::string poplar_executable_filename = proto.engine();
  std::unique_ptr<poplar::Engine> engine;
  try {
    std::ifstream file(poplar_executable_filename, std::ios::binary);
    auto poplar_executable = poplar::Executable::deserialize(file);
    engine.reset(new poplar::Engine(std::move(poplar_executable), opts));
  } catch (const std::exception& e) {
    return PoplarExceptionToTensorflowStatus("[Deserialize] ", e);
  }

  auto iomap = InputOutputAliasingMap(hlo_module.get());

  auto executable = new PoplarExecutable(
      std::move(hlo_module), std::move(profile_printer),
      std::move(profile_index_map), std::move(engine), std::move(iomap), false,
      {}, false, {}, replication_factor, std::move(infeeds),
      std::move(outfeeds));

  executable->loaded_from_cache_ = true;

  return executable;
}

/*static*/ Status PoplarExecutable::Serialize(
    const std::string& filename, const poplar::Executable& executable,
    const InfeedInfos& infeeds, const OutfeedInfos& outfeeds,
    uint32 replication_count, const poplar::OptionFlags& opts) {
  PoplarExecutableProto proto;

  // Write poplar executable to a file
  std::string poplar_executable_filename = filename + ".poplar_exec";
  try {
    auto file = std::ofstream(poplar_executable_filename, std::ios::binary);
    executable.serialize(file);
  } catch (const std::exception& e) {
    return PoplarExceptionToTensorflowStatus("[Serialize] ", e);
  }

  proto.set_engine(poplar_executable_filename);

  proto.set_replication_factor(replication_count);

  for (const auto& infeed : infeeds) {
    auto* feed = proto.add_infeeds();
    feed->set_stream_prefix(infeed.stream_prefix);
    *(feed->mutable_config()) = infeed.config;
    *(feed->mutable_shape()) = infeed.shape.ToProto();
  }

  for (const auto& outfeed : outfeeds) {
    auto* feed = proto.add_infeeds();
    feed->set_stream_prefix(outfeed.stream_prefix);
    *(feed->mutable_config()) = outfeed.config;
    *(feed->mutable_shape()) = outfeed.shape.ToProto();
  }

  // write the compilation options into the serialized executable
  for (const auto flag : opts) {
    auto* poplar_opt = proto.add_option_flags();
    poplar_opt->set_key(flag.first);
    poplar_opt->set_value(flag.second);
  }

  return WriteBinaryProto(tensorflow::Env::Default(), filename, proto);
}

}  // namespace poplarplugin
}  // namespace xla
