/* Copyright 2025 The OpenXLA Authors.

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
#ifndef XLA_PYTHON_TRANSFER_SOCKET_BULK_TRANSPORT_H_
#define XLA_PYTHON_TRANSFER_SOCKET_BULK_TRANSPORT_H_

#include <netinet/in.h>
#include <sys/socket.h>

#include <atomic>
#include <deque>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/python/transfer/event_loop.h"
#include "xla/python/transfer/streaming.h"
#include "tsl/platform/env.h"

namespace aux {

// Send sending zero-copy tcp messages, we must pool together multiple acks
// (if the message is broken up) from the kernel to call the on_done message
// for the overall user message. This table manages this book-keeping.
//
// while (true) {
//   send(...)
//   if (is_last) {
//     table.Seal(std::move(on_done));
//     break;
//   } else { table.Send(); }
// }
class ZeroCopySendAckTable {
 public:
  ZeroCopySendAckTable();

  // Called when this is not the last message of the batch of sends.
  void Send();

  // Called when this is the last message of the batch.
  uint32_t Seal(absl::AnyInvocable<void() &&> on_done);

  // acks ids must be tracked separately, but start at zero and must
  // match calls to Send() or Seal(). May ack out of order.
  void HandleAck(uint32_t v);

  // For testing of rollover behavior.
  void PretendCloseToRolloverForTests(uint32_t bump);

  // Should be [0, 0] if there is no outstanding work.
  std::pair<size_t, size_t> GetTableSizes();

  absl::Status HandleSocketErrors(int fd);

  void ClearAll();

 private:
  void GCTables();
  absl::Mutex mu_;
  size_t n_acks_in_batch_ = 0;
  uint32_t ack_ids_start_ = 0;
  uint32_t acks_start_ = 0;
  struct AckState {
    ssize_t acks_count = -1;
    absl::AnyInvocable<void() &&> on_done;
  };
  std::deque<std::optional<uint32_t>> ack_ids_;
  std::deque<AckState> acks_;
};

// Opaque state for coordinating between a SharedSendWorkQueue and a
// SharedSendMsgQueue.
class SendConnectionHandler;

// A single work queue which handles any subconnections which are
// scheduled on it.
class SharedSendWorkQueue {
 public:
  // Starts the worker thread.
  static std::shared_ptr<SharedSendWorkQueue> Start();

 private:
  void Run();

  void ScheduleWork(SendConnectionHandler* handler,
                    aux::BulkTransportInterface::SendMessage msg);

  friend class SendConnectionHandler;

  struct SendWorkItem {
    SendConnectionHandler* handler;
    aux::BulkTransportInterface::SendMessage msg;
  };
  absl::Mutex mu_;
  bool shutdown_ = false;
  std::deque<SendWorkItem> work_items_;
  std::unique_ptr<tsl::Thread> thread_;
};

// Send queue for implementing BulkTransportInterface.
class SharedSendMsgQueue {
 public:
  // Schedules a SendMessage on one of the attached worker threads.
  void ScheduleSendWork(aux::BulkTransportInterface::SendMessage msg);

  // Report to the queue that there are no more messages to send.
  void NoMoreMessages();

  // Starts a sender for 1 part of the thread.
  static void StartSubConnectionSender(
      int fd, int bond_id, std::shared_ptr<SharedSendMsgQueue> msg_queue,
      std::shared_ptr<SharedSendWorkQueue> work_queue,
      size_t artificial_send_limiti = std::numeric_limits<size_t>::max());

 private:
  friend class SendConnectionHandler;

  void ReportReadyToSend(SendConnectionHandler* handler);

  absl::Mutex mu_;
  bool shutdown_ = false;
  std::deque<SendConnectionHandler*> handlers_;
  std::deque<aux::BulkTransportInterface::SendMessage> work_items_;
};

// Recv thread for scheduling work on a BulkTransportInterface.
class RecvThreadState {
 public:
  // Schedules recv() syscall on a particular fd.
  void ScheduleRecvWork(
      size_t recv_size, int fd,
      absl::AnyInvocable<
          void(absl::StatusOr<aux::BulkTransportInterface::Message> msg) &&>
          on_recv);

  // Starts the worker thread.
  static std::shared_ptr<RecvThreadState> Create(
      std::optional<SlabAllocator> allocator, SlabAllocator uallocator);

 private:
  RecvThreadState(std::optional<SlabAllocator> allocator,
                  SlabAllocator uallocator);
  ~RecvThreadState() = default;

  void DoRecvWork();

  struct recv_work_item {
    size_t recv_size;
    int fd;
    absl::AnyInvocable<
        void(absl::StatusOr<aux::BulkTransportInterface::Message> msg) &&>
        on_recv;
  };

  absl::Status HandleRecvItem(recv_work_item& work, size_t& zc_send_count,
                              size_t& non_zc_send_count);

  std::optional<SlabAllocator> allocator_;
  SlabAllocator uallocator_;
  absl::Mutex recv_mu_;
  bool recv_shutdown_ = false;
  std::deque<recv_work_item> recv_work_items_;
  std::unique_ptr<tsl::Thread> recv_thread_;
};

// Create a socket transport factory that allocates out of allocator and
// unpinned_allocator and communicates over all addrs in parallel.
absl::StatusOr<std::shared_ptr<BulkTransportFactory>>
CreateSocketBulkTransportFactory(std::vector<SocketAddress> addrs,
                                 std::optional<SlabAllocator> allocator,
                                 SlabAllocator unpinned_allocator);

}  // namespace aux

#endif  // XLA_PYTHON_TRANSFER_SOCKET_BULK_TRANSPORT_H_
