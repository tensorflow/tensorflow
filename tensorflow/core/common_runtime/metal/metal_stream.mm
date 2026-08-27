/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/metal/metal_stream.h"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "absl/log/log.h"
#include "absl/synchronization/mutex.h"

SP_Stream_st::SP_Stream_st(id<MTLCommandQueue> command_queue,
                           id<MTLSharedEvent> event)
    : queue([command_queue retain]), order_event([event retain]) {}

SP_Stream_st::~SP_Stream_st() {
  [queue release];
  [order_event release];
}

SP_Event_st::SP_Event_st(id<MTLSharedEvent> e) : event([e retain]) {}

SP_Event_st::~SP_Event_st() { [event release]; }

namespace tensorflow {
namespace metal {

namespace {

// Records the first command buffer failure on the stream, if any.
void NoteFailure(SP_Stream stream, id<MTLCommandBuffer> completed) {
  if (completed.error == nil) return;
  absl::MutexLock lock(&stream->mu);
  if (stream->failed) return;  // Keep the first failure, it is the useful one.
  stream->failed = true;
  const char* message = completed.error.localizedDescription.UTF8String;
  stream->failure_message =
      message != nullptr ? message : "unknown Metal command buffer error";
}

}  // namespace

MetalDeviceState::MetalDeviceState(id<MTLDevice> mtl_device)
    : device([mtl_device retain]) {}

MetalDeviceState::~MetalDeviceState() { [device release]; }

void MetalDeviceState::AddStream(SP_Stream stream) {
  absl::MutexLock lock(&mu);
  streams.push_back(stream);
}

void MetalDeviceState::RemoveStream(SP_Stream stream) {
  absl::MutexLock lock(&mu);
  streams.erase(std::remove(streams.begin(), streams.end(), stream),
                streams.end());
}

void MetalDeviceState::BlockUntilIdle() {
  // Snapshot under the lock, then wait outside it: waiting can take
  // arbitrarily long and must not block stream creation or teardown.
  std::vector<SP_Stream> snapshot;
  {
    absl::MutexLock lock(&mu);
    snapshot = streams;
  }
  for (SP_Stream stream : snapshot) {
    uint64_t target = 0;
    {
      absl::MutexLock lock(&stream->mu);
      target = stream->last_enqueued;
    }
    if (target == 0) continue;  // Nothing was ever enqueued on this stream.
    [stream->order_event waitUntilSignaledValue:target timeoutMS:UINT64_MAX];
  }
}

MetalDeviceState* StateOf(const SP_Device* device) {
  return static_cast<MetalDeviceState*>(device->ext);
}

ScopedAutoreleasePool::ScopedAutoreleasePool()
    : pool_([[NSAutoreleasePool alloc] init]) {}

ScopedAutoreleasePool::~ScopedAutoreleasePool() {
  [static_cast<NSAutoreleasePool*>(pool_) drain];
}

OrderedCommandBuffer::OrderedCommandBuffer(SP_Stream stream)
    : stream_(stream) {
  uint64_t wait_value = 0;
  {
    absl::MutexLock lock(&stream_->mu);
    wait_value = stream_->last_enqueued;
    signal_value_ = wait_value + 1;
    stream_->last_enqueued = signal_value_;
  }

  buffer_ = [[stream_->queue commandBuffer] retain];
  if (buffer_ == nil) {
    // Give the sequence number back so the stream is not permanently stuck
    // waiting for a value that will never be signalled.
    absl::MutexLock lock(&stream_->mu);
    if (stream_->last_enqueued == signal_value_) {
      stream_->last_enqueued = wait_value;
    }
    LOG(ERROR) << "Metal: MTLCommandQueue returned no command buffer.";
    return;
  }

  // Value 0 is the event's initial state, so a wait on it is trivially
  // satisfied; skipping it avoids an encode on the very first buffer.
  if (wait_value > 0) {
    [buffer_ encodeWaitForEvent:stream_->order_event value:wait_value];
  }
}

OrderedCommandBuffer::~OrderedCommandBuffer() {
  if (buffer_ == nil) return;
  if (!committed_) {
    // Nothing was committed, so the GPU will never signal our value. Commit an
    // otherwise empty buffer purely to advance the sequence, rather than
    // leaving every later buffer on this stream blocked forever.
    LOG(WARNING) << "Metal: command buffer dropped without commit; committing "
                    "an empty buffer to keep the stream ordered.";
    Commit();
  }
  [buffer_ release];
}

void OrderedCommandBuffer::EncodeSignal() {
  [buffer_ encodeSignalEvent:stream_->order_event value:signal_value_];
}

void OrderedCommandBuffer::Commit() {
  if (buffer_ == nil || committed_) return;
  committed_ = true;
  EncodeSignal();

  // Captured by value; the block outlives this object.
  SP_Stream stream = stream_;
  [buffer_ addCompletedHandler:^(id<MTLCommandBuffer> completed) {
    NoteFailure(stream, completed);
  }];

  [buffer_ commit];
}

OrderedCommandBuffer::ExternalCommit
OrderedCommandBuffer::ReleaseForExternalCommit() {
  // Marked committed so the destructor neither commits nor warns; the caller
  // owns the signal from here.
  committed_ = true;
  return ExternalCommit{stream_, signal_value_};
}

void OrderedCommandBuffer::CommitWithHostCompletion(void (^on_complete)()) {
  if (buffer_ == nil || committed_) return;
  committed_ = true;

  SP_Stream stream = stream_;
  const uint64_t signal_value = signal_value_;
  // The block outlives this scope, so it has to be moved off the stack.
  void (^work)() = [on_complete copy];

  [buffer_ addCompletedHandler:^(id<MTLCommandBuffer> completed) {
    NoteFailure(stream, completed);
    if (completed.error == nil && work != nil) work();
    // Signalled last, and unconditionally: the next command buffer on this
    // stream is waiting on this value and must not start before the host work
    // above has finished, nor stall forever if that work was skipped.
    stream->order_event.signaledValue = signal_value;
    [work release];
  }];

  [buffer_ commit];
}

void OrderedCommandBuffer::CommitAndWait() {
  if (buffer_ == nil) return;
  Commit();
  [buffer_ waitUntilCompleted];
}

}  // namespace metal
}  // namespace tensorflow
