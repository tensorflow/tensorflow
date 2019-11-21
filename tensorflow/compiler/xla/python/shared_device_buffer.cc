/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/python/shared_device_buffer.h"

#include <memory>

#include "tensorflow/stream_executor/device_memory_allocator.h"

namespace xla {

void BufferDefinitionEvent::SetDefinitionEvent(EventPool::Handle event,
                                               se::Stream* stream) {
  absl::MutexLock lock(&mu_);
  CHECK(!event_.event());
  event_ = std::move(event);
  CHECK(streams_defined_on_.empty());
  streams_defined_on_.push_back(stream);
}

bool BufferDefinitionEvent::EventHasBeenRecorded() {
  return event_.event() != nullptr;
}

void BufferDefinitionEvent::WaitForEventOnStream(se::Stream* stream) {
  absl::MutexLock lock(&mu_);

  // We cannot wait for an event until ThenRecordEvent has been called; on GPU
  // newly created events are deemed to have already happened past.
  mu_.Await(
      absl::Condition(this, &BufferDefinitionEvent::EventHasBeenRecorded));

  // The set of defined streams is expected to be very small indeed (usually
  // 1-2), so a simple linear scan should be fast enough.
  if (std::find(streams_defined_on_.begin(), streams_defined_on_.end(),
                stream) != streams_defined_on_.end()) {
    // stream is in streams_defined_on_; it doesn't need to be waited on.
    return;
  }

  stream->ThenWaitFor(event_.event());
  streams_defined_on_.push_back(stream);
}

static std::shared_ptr<SharedDeviceBuffer> BufferFromScopedShapedBufferIterator(
    const Shape& on_device_shape, int device_ordinal,
    se::DeviceMemoryAllocator* allocator,
    ShapeTree<se::DeviceMemoryBase>::iterator* iterator,
    const ShapeTree<se::DeviceMemoryBase>::iterator& end,
    const std::shared_ptr<BufferDefinitionEvent>& definition_event) {
  CHECK(*iterator != end);

  se::OwningDeviceMemory device_memory((*iterator)->second, device_ordinal,
                                       allocator);
  (*iterator)->second = se::DeviceMemoryBase();
  ++*iterator;

  std::vector<std::shared_ptr<SharedDeviceBuffer>> children;
  if (on_device_shape.IsTuple()) {
    int num_children = ShapeUtil::TupleElementCount(on_device_shape);
    children.reserve(num_children);
    for (int i = 0; i < num_children; ++i) {
      children.push_back(BufferFromScopedShapedBufferIterator(
          on_device_shape.tuple_shapes(i), device_ordinal, allocator, iterator,
          end, definition_event));
    }
  }
  return std::make_shared<SharedDeviceBuffer>(
      on_device_shape, std::move(device_memory), children, definition_event);
}

/* static */ std::shared_ptr<SharedDeviceBuffer>
SharedDeviceBuffer::FromScopedShapedBuffer(
    ScopedShapedBuffer shaped_buffer,
    const std::shared_ptr<BufferDefinitionEvent>& definition_event) {
  ShapeTree<se::DeviceMemoryBase>::iterator iterator =
      shaped_buffer.buffers().begin();
  std::shared_ptr<SharedDeviceBuffer> output =
      BufferFromScopedShapedBufferIterator(
          shaped_buffer.on_device_shape(), shaped_buffer.device_ordinal(),
          shaped_buffer.memory_allocator(), &iterator,
          shaped_buffer.buffers().end(), definition_event);
  CHECK(iterator == shaped_buffer.buffers().end());
  return output;
}

/* static */ StatusOr<std::shared_ptr<SharedDeviceBuffer>>
SharedDeviceBuffer::MakeTuple(
    std::vector<std::shared_ptr<SharedDeviceBuffer>> children,
    TransferManager* transfer_manager, se::DeviceMemoryAllocator* allocator,
    int device_ordinal,
    std::shared_ptr<BufferDefinitionEvent> definition_event) {
  std::vector<Shape> child_shapes;
  child_shapes.reserve(children.size());
  for (const auto& child : children) {
    TF_RET_CHECK(child->device_memory().device_ordinal() == device_ordinal);
    child_shapes.push_back(child->on_device_shape());
  }

  Shape shape = ShapeUtil::MakeTupleShape(child_shapes);
  TF_ASSIGN_OR_RETURN(
      se::OwningDeviceMemory device_memory,
      allocator->Allocate(device_ordinal,
                          transfer_manager->GetByteSizeRequirement(shape)));
  return std::make_shared<SharedDeviceBuffer>(
      std::move(shape), std::move(device_memory), std::move(children),
      std::move(definition_event));
}

/* static */ StatusOr<std::shared_ptr<SharedDeviceBuffer>>
SharedDeviceBuffer::MakeArray(
    Shape on_device_shape, TransferManager* transfer_manager,
    se::DeviceMemoryAllocator* allocator, int device_ordinal,
    std::shared_ptr<BufferDefinitionEvent> definition_event) {
  TF_ASSIGN_OR_RETURN(
      se::OwningDeviceMemory device_memory,
      allocator->Allocate(
          device_ordinal,
          transfer_manager->GetByteSizeRequirement(on_device_shape)));
  return std::make_shared<SharedDeviceBuffer>(
      std::move(on_device_shape), std::move(device_memory),
      /*children=*/std::vector<std::shared_ptr<SharedDeviceBuffer>>{},
      std::move(definition_event));
}

// Populates a buffer tree from a ShapeTree iterator.
static void PopulateShapedBufferFromBuffer(
    const SharedDeviceBuffer& buffer,
    ShapeTree<se::DeviceMemoryBase>::iterator* iterator,
    const ShapeTree<se::DeviceMemoryBase>::iterator& end) {
  CHECK(*iterator != end);
  (*iterator)->second = *buffer.device_memory();
  ++*iterator;
  for (const auto& child : buffer.children()) {
    PopulateShapedBufferFromBuffer(*child, iterator, end);
  }
}

ShapedBuffer SharedDeviceBuffer::AsShapedBuffer(
    const Shape& on_host_shape) const {
  ShapedBuffer shaped_buffer(on_host_shape, on_device_shape_,
                             device_memory_.allocator()->platform(),
                             device_memory_.device_ordinal());
  ShapeTree<se::DeviceMemoryBase>::iterator iterator =
      shaped_buffer.buffers().begin();
  PopulateShapedBufferFromBuffer(*this, &iterator,
                                 shaped_buffer.buffers().end());
  CHECK(iterator == shaped_buffer.buffers().end());
  return shaped_buffer;
}

SharedDeviceBuffer::SharedDeviceBuffer(
    Shape on_device_shape, se::OwningDeviceMemory device_memory,
    std::vector<std::shared_ptr<SharedDeviceBuffer>> children,
    std::shared_ptr<BufferDefinitionEvent> definition_event)
    : on_device_shape_(std::move(on_device_shape)),
      device_memory_(std::move(device_memory)),
      children_(std::move(children)),
      definition_event_(std::move(definition_event)) {}

void GetDeviceBufferDefinitionEvents(
    const SharedDeviceBuffer& buffer,
    absl::flat_hash_set<BufferDefinitionEvent*>* events) {
  if (buffer.definition_event()) {
    events->insert(buffer.definition_event().get());
  }
  for (const auto& child : buffer.children()) {
    GetDeviceBufferDefinitionEvents(*child, events);
  }
}

void WaitForBufferDefinitionEventsOnStream(const SharedDeviceBuffer& buffer,
                                           se::Stream* stream) {
  absl::flat_hash_set<BufferDefinitionEvent*> events;
  GetDeviceBufferDefinitionEvents(buffer, &events);
  for (BufferDefinitionEvent* event : events) {
    event->WaitForEventOnStream(stream);
  }
}

}  // namespace xla
