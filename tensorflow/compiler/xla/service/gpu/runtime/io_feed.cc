/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/runtime/io_feed.h"

#include <memory>
#include <string_view>

#include "tensorflow/compiler/xla/runtime/custom_call.h"
#include "tensorflow/compiler/xla/runtime/executable.h"
#include "tensorflow/compiler/xla/service/gpu/infeed_manager.h"
#include "tensorflow/compiler/xla/service/gpu/outfeed_manager.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/support.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"

namespace xla {
namespace gpu {

using runtime::CustomCall;

static absl::Status InfeedImpl(const ServiceExecutableRunOptions* run_options,
                               CustomCall::RemainingArgs args,
                               std::string_view config) {
  VLOG(3) << "Infeeding to GPU";

  se::Stream* stream = run_options->stream();
  ShapeTree<se::ScopedDeviceMemory<uint8_t>> source_buffers =
      GetOrCreateInfeedManager(stream->parent())->BlockingGetNextDestination();

  // Check that we have correct number of arguments.
  if (args.size() != source_buffers.leaf_count())
    return absl::InvalidArgumentError("Incorrect number of arguments");

  size_t index = 0;
  for (auto& source : source_buffers.leaves()) {
    // Get the destination buffer.
    auto dest = args.get<runtime::StridedMemrefView>(index);
    if (failed(dest))
      return absl::InternalError("Failed to get the destination buffer");

    // Get the source buffer shape.
    const Shape& source_shape =
        ShapeUtil::GetSubshape(source_buffers.shape(), source.first);

    // Check that destination shape matches the source shape.
    Shape dest_shape = ToShape(*dest);
    if (!ShapeUtil::ReshapeIsBitcast(dest_shape, source_shape)) {
      return absl::InvalidArgumentError(
          "The destination shape does not match the source shape");
    }

    se::DeviceMemoryBase dest_address = GetDeviceAddress(*dest);
    se::ScopedDeviceMemory<uint8_t>& buffer = source.second;
    stream->ThenMemcpy(&dest_address, *buffer.ptr(), buffer.ptr()->size());

    ++index;
  }

  // TODO(ezhulenev): Make this function async?
  Status block_status = stream->BlockHostUntilDone();
  if (!block_status.ok()) return ToAbslStatus(block_status);

  VLOG(3) << "Infeeding to GPU complete";

  return absl::OkStatus();
}

static absl::Status OutfeedImpl(const ServiceExecutableRunOptions* run_options,
                                CustomCall::RemainingArgs args,
                                std::string_view config) {
  VLOG(3) << "Outfeeding from GPU";

  se::Stream* stream = run_options->stream();
  OutfeedManager* outfeed_manager = GetOrCreateOutfeedManager(stream->parent());
  ShapeTree<std::unique_ptr<OutfeedBuffer>>* dest_buffers =
      outfeed_manager->BlockingGetNextDestination();

  // Nothing to be done for an outfeed with no inputs.
  // Note: Must do this after `BlockingGetNextDestination` above to dequeue an
  // entry from the outfeed manager.
  if (args.empty()) return absl::OkStatus();

  // Check that we have correct number of arguments.
  if (args.size() != dest_buffers->leaf_count())
    return absl::InvalidArgumentError("Incorrect number of arguments");

  int64_t leaf_count = dest_buffers->leaf_count();
  auto dest_leaf_it = dest_buffers->leaf_begin();

  for (int64_t index = 0; index < leaf_count; ++index) {
    const ShapeIndex& shape_index = dest_leaf_it->first;
    std::unique_ptr<OutfeedBuffer>& buffer = dest_leaf_it->second;

    // NOTE: This code needs deal with the `dest_buffers` object getting
    // deleted when it is executing. Specifically, objects in the outfeed queue
    // are pointers to instances of stack-allocated objects in
    // `GpuTransferManager::TransferLiteralFromOutfeed`. When all leaf node
    // buffers are notified via "buffer->Done()" below in the stream host
    // callback, `TransferLiteralFromOutfeed` deletes this stack-allocated
    // object when it returns. This means that it is possible that during the
    // last iteration, after the call to "buffer->Done()" is scheduled onto the
    // stream, the `dest_buffers` object might get deleted, so we should avoid
    // accessing the object after that.
    //
    // To achieve that, increment the leaf iterator here before the last "Done"
    // is enqueued, instead of in the loop increment, which would be after the
    // "Done" is scheduled.
    ++dest_leaf_it;

    // Get the source buffer.
    auto source = args.get<runtime::StridedMemrefView>(index);
    if (failed(source))
      return absl::InternalError("Failed to get the source buffer");

    // Get the source buffer shape.
    const Shape& dest_shape =
        ShapeUtil::GetSubshape(dest_buffers->shape(), shape_index);

    // Check that destination shape matches the source shape.
    Shape source_shape = ToShape(*source);
    if (!ShapeUtil::ReshapeIsBitcast(dest_shape, source_shape)) {
      return absl::InvalidArgumentError(
          "The destination shape does not match the source shape");
    }

    se::DeviceMemoryBase source_address = GetDeviceAddress(*source);

    // Schedule the memory transfer.
    auto* dest_address = buffer->destination()->untyped_data();
    stream->ThenMemcpy(dest_address, source_address, buffer->length())
        .ThenDoHostCallback([&buffer]() { buffer->Done(); });
  }

  Status block_status = stream->BlockHostUntilDone();
  if (!block_status.ok()) return ToAbslStatus(block_status);

  VLOG(3) << "Outfeeding from GPU complete";

  return absl::OkStatus();
}

//===----------------------------------------------------------------------===//
// Define Xla runtime bindings for the custom calls.
//===----------------------------------------------------------------------===//

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    Infeed, FunctionWrapper<InfeedImpl>(), checks,
    CustomCall::Bind("xla.gpu.infeed")
        .UserData<const ServiceExecutableRunOptions*>()
        .Arg<CustomCall::RemainingArgs>()  // args
        .Attr<std::string_view>("config"));

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    Outfeed, FunctionWrapper<OutfeedImpl>(), checks,
    CustomCall::Bind("xla.gpu.outfeed")
        .UserData<const ServiceExecutableRunOptions*>()
        .Arg<CustomCall::RemainingArgs>()  // args
        .Attr<std::string_view>("config"));

//===----------------------------------------------------------------------===//

void RegisterIoFeedCustomCalls(runtime::DirectCustomCallRegistry& registry) {
  registry.Register("xla.gpu.infeed", Infeed);
  registry.Register("xla.gpu.outfeed", Outfeed);
}

}  // namespace gpu
}  // namespace xla
