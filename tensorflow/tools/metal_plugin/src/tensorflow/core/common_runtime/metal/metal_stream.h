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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_METAL_METAL_STREAM_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_METAL_METAL_STREAM_H_

// This header is Objective-C++ only and can only be included from a .mm
// translation unit.

#import <Metal/Metal.h>

#include <cstdint>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/c/experimental/stream_executor/stream_executor.h"

// StreamExecutor's stream is a strictly ordered queue of work: everything
// enqueued on a stream runs after everything enqueued before it. Metal does
// not give that for free. An MTLCommandQueue guarantees command buffers are
// *scheduled* in commit order, but their execution may overlap, so committing
// each operation as its own command buffer would let two supposedly ordered
// kernels run concurrently.
//
// We restore the contract with a per-stream MTLSharedEvent used as a sequence
// counter: command buffer N waits for the event to reach N-1 and signals N on
// completion. That serialises the stream without serialising the device, so
// independent streams still overlap.
struct SP_Stream_st {
  SP_Stream_st(id<MTLCommandQueue> command_queue, id<MTLSharedEvent> event);
  ~SP_Stream_st();

  SP_Stream_st(const SP_Stream_st&) = delete;
  SP_Stream_st& operator=(const SP_Stream_st&) = delete;

  id<MTLCommandQueue> queue;
  id<MTLSharedEvent> order_event;

  // Held from the moment a command buffer takes its sequence number until it
  // is committed, so that commit order matches the order the numbers were
  // handed out in.
  //
  // A Metal queue runs command buffers in commit order, and each buffer waits
  // for its predecessor's number before doing anything. TensorFlow runs
  // kernels on several threads, so without this two buffers could be numbered
  // in one order and committed in the other, and the one committed first would
  // wait for a number only the one behind it will ever signal. The queue then
  // stops for good, which is what a GPU timeout partway through a training
  // step looks like. This serialises submission, not execution: the GPU still
  // overlaps the work itself.
  mutable absl::Mutex submit;

  mutable absl::Mutex mu;
  // Sequence number of the last command buffer handed out for this stream.
  uint64_t last_enqueued ABSL_GUARDED_BY(mu) = 0;
  // Sticky: set by a command buffer that failed, reported by
  // SP_StreamExecutor::get_stream_status until the stream is destroyed.
  bool failed ABSL_GUARDED_BY(mu) = false;
  std::string failure_message ABSL_GUARDED_BY(mu);
};

// An event is a point in a stream's sequence. Recording it captures the
// sequence number the stream had reached; waiting on it blocks until the
// stream's event counter passes that number.
struct SP_Event_st {
  explicit SP_Event_st(id<MTLSharedEvent> event);
  ~SP_Event_st();

  SP_Event_st(const SP_Event_st&) = delete;
  SP_Event_st& operator=(const SP_Event_st&) = delete;

  // Owned by this event, distinct from any stream's ordering event so that
  // recording the same event on two streams is well defined.
  id<MTLSharedEvent> event;

  // Held from the moment a command buffer takes its sequence number until it
  // is committed, so that commit order matches the order the numbers were
  // handed out in.
  //
  // A Metal queue runs command buffers in commit order, and each buffer waits
  // for its predecessor's number before doing anything. TensorFlow runs
  // kernels on several threads, so without this two buffers could be numbered
  // in one order and committed in the other, and the one committed first would
  // wait for a number only the one behind it will ever signal. The queue then
  // stops for good, which is what a GPU timeout partway through a training
  // step looks like. This serialises submission, not execution: the GPU still
  // overlaps the work itself.
  mutable absl::Mutex submit;

  mutable absl::Mutex mu;
  // 0 means "never recorded", which core treats as already complete.
  uint64_t target ABSL_GUARDED_BY(mu) = 0;
};

// Metal reports GPU timings on the command buffer, not on a timer object, so
// a timer is just the pair of timestamps harvested from the empty command
// buffers that start_timer and stop_timer enqueue.
struct SP_Timer_st {
  // Held from the moment a command buffer takes its sequence number until it
  // is committed, so that commit order matches the order the numbers were
  // handed out in.
  //
  // A Metal queue runs command buffers in commit order, and each buffer waits
  // for its predecessor's number before doing anything. TensorFlow runs
  // kernels on several threads, so without this two buffers could be numbered
  // in one order and committed in the other, and the one committed first would
  // wait for a number only the one behind it will ever signal. The queue then
  // stops for good, which is what a GPU timeout partway through a training
  // step looks like. This serialises submission, not execution: the GPU still
  // overlaps the work itself.
  mutable absl::Mutex submit;

  mutable absl::Mutex mu;
  double start_seconds ABSL_GUARDED_BY(mu) = 0.0;
  double end_seconds ABSL_GUARDED_BY(mu) = 0.0;
  bool started ABSL_GUARDED_BY(mu) = false;
  bool stopped ABSL_GUARDED_BY(mu) = false;
};

namespace tensorflow {
namespace metal {

// Per-device state hanging off SP_Device::ext.
//
// SP_StreamExecutor's synchronous memcpy and synchronize_all_activity
// callbacks take a device but no stream, so the device has to know which
// streams are live in order to drain them.
struct MetalDeviceState {
  explicit MetalDeviceState(id<MTLDevice> mtl_device);
  ~MetalDeviceState();

  MetalDeviceState(const MetalDeviceState&) = delete;
  MetalDeviceState& operator=(const MetalDeviceState&) = delete;

  void AddStream(SP_Stream stream);
  void RemoveStream(SP_Stream stream);
  // Blocks until every live stream has drained.
  void BlockUntilIdle();

  id<MTLDevice> device;

  // Held from the moment a command buffer takes its sequence number until it
  // is committed, so that commit order matches the order the numbers were
  // handed out in.
  //
  // A Metal queue runs command buffers in commit order, and each buffer waits
  // for its predecessor's number before doing anything. TensorFlow runs
  // kernels on several threads, so without this two buffers could be numbered
  // in one order and committed in the other, and the one committed first would
  // wait for a number only the one behind it will ever signal. The queue then
  // stops for good, which is what a GPU timeout partway through a training
  // step looks like. This serialises submission, not execution: the GPU still
  // overlaps the work itself.
  mutable absl::Mutex submit;

  mutable absl::Mutex mu;
  std::vector<SP_Stream> streams ABSL_GUARDED_BY(mu);
};

// Recovers the state attached to an SP_Device. Never null for a device the
// platform created.
MetalDeviceState* StateOf(const SP_Device* device);

// Drains autoreleased Objective-C objects at scope exit.
//
// TensorFlow executor threads are plain pthreads with no top-level autorelease
// pool. Metal hands back autoreleased objects from routine calls
// (-commandBuffer, -blitCommandEncoder, -computeCommandEncoder, and everything
// MPS returns), so without a pool of our own those objects leak on every op.
// Every entry point into this backend that touches Metal declares one of
// these as its first statement.
class ScopedAutoreleasePool {
 public:
  ScopedAutoreleasePool();
  ~ScopedAutoreleasePool();

  ScopedAutoreleasePool(const ScopedAutoreleasePool&) = delete;
  ScopedAutoreleasePool& operator=(const ScopedAutoreleasePool&) = delete;

 private:
  void* pool_;
};

// A command buffer that participates in its stream's ordering.
//
// Construction reserves the next sequence number and encodes the wait for the
// previous one; Commit() encodes the matching signal. Callers encode their own
// work in between. Destroying one without committing cancels the reservation,
// so a failed encode does not wedge the stream forever.
// Records a finished command buffer's failure, if it had one, and its GPU
// timing when a profiler is collecting.
//
// Exposed because not every command buffer this backend opens is committed by
// OrderedCommandBuffer::Commit. MPSGraph may call commitAndContinue and commit
// a buffer of its own, and that path has to report failures and timings the
// same way or an MPSGraph error is silently dropped and every MPSGraph op is
// missing from the profile.
void NoteCommandBufferCompletion(SP_Stream stream,
                                 id<MTLCommandBuffer> completed);

class OrderedCommandBuffer {
 public:
  explicit OrderedCommandBuffer(SP_Stream stream);
  ~OrderedCommandBuffer();

  OrderedCommandBuffer(const OrderedCommandBuffer&) = delete;
  OrderedCommandBuffer& operator=(const OrderedCommandBuffer&) = delete;

  // Nil if the command queue could not produce a buffer.
  id<MTLCommandBuffer> get() const { return buffer_; }
  bool ok() const { return buffer_ != nil; }

  // Sequence number this buffer signals once complete.
  uint64_t signal_value() const { return signal_value_; }

  // Encodes the ordering signal and commits. Further work must not be encoded
  // afterwards.
  void Commit();
  // Commit() followed by waiting for the GPU to finish this buffer.
  void CommitAndWait();

  // Hands responsibility for committing to the caller.
  //
  // Needed by the MPSGraph path. MPSGraph may call commitAndContinue, which
  // commits the buffer we started with and carries on with a fresh one, so the
  // ordering signal has to be encoded on whichever buffer is live at the end
  // rather than the one this object was constructed with. The caller must
  // encode a signal of `signal_value` on the stream's order_event and commit,
  // exactly once, or the stream stalls forever.
  struct ExternalCommit {
    SP_Stream stream;
    uint64_t signal_value;
  };
  ExternalCommit ReleaseForExternalCommit();

  // Commits without encoding the GPU-side ordering signal. Instead, once the
  // GPU work finishes, `on_complete` runs on a Metal callback thread and the
  // sequence is signalled from the host afterwards.
  //
  // This is what lets host-side copies take part in stream order. On a unified
  // memory architecture the fastest host/device copy is a plain memcpy, but a
  // memcpy issued from an ordinary completion handler would race the next
  // command buffer: the GPU signal would already have released it. Signalling
  // from the host after the copy closes that window.
  //
  // The sequence is signalled even if the command buffer failed, since leaving
  // it unsignalled would wedge every later buffer on the stream.
  void CommitWithHostCompletion(void (^on_complete)());

 private:
  void EncodeSignal();
  // Releases the stream's submission lock, once, whichever way this buffer
  // was committed.
  void ReleaseSubmission();

  SP_Stream stream_;
  id<MTLCommandBuffer> buffer_ = nil;
  uint64_t signal_value_ = 0;
  bool committed_ = false;
  bool holds_submission_ = false;
};

}  // namespace metal
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_METAL_METAL_STREAM_H_
