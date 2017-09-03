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

#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_VERBS_RDMA_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_VERBS_RDMA_H_

#ifdef TENSORFLOW_USE_VERBS

#include <infiniband/verbs.h>
#include <cstring>  // for memset
#include <functional>
#include <memory>  // for shared_ptr
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {

// structure to save the address of remote channels.
struct RdmaAddress {
  uint32_t lid;
  uint32_t qpn;
  uint32_t psn;
  uint64_t snp;
  uint64_t iid;
};
// structure to save information for remote memory regions.
struct RemoteMR {
  uint64_t remote_addr;
  uint32_t rkey;
};
enum BufferStatus { none, idle, busy };
enum Location { local, remote };
enum BufferType { ACK, MESSAGE, TENSOR };
enum RdmaMessageType {
  RDMA_MESSAGE_ACK,
  RDMA_MESSAGE_BUFFER_IDLE,
  RDMA_MESSAGE_BUFFER_REQUEST,
  RDMA_MESSAGE_BUFFER_RESPONSE,
  RDMA_MESSAGE_TENSOR_REQUEST,
  RDMA_MESSAGE_TENSOR_WRITE
};
class RdmaBuffer;
// Class that represents the Rdma Adapter.
// Responsible for creation of the completion queue, and handling
// of work completions.
class RdmaAdapter {
  friend class RdmaChannel;
  friend class RdmaBuffer;
  friend class RdmaAckBuffer;
  friend class RdmaMessageBuffer;
  friend class RdmaTensorBuffer;
  friend class RdmaMgr;
  friend class RdmaRemoteRendezvous;

 public:
  RdmaAdapter(const WorkerEnv* worker_env);
  ~RdmaAdapter();
  // Adapter name, e.g. mlx5_0.
  string name() const;
  void Process_CQ();

 protected:
  static const int MAX_CONCURRENT_WRITES = 1000;
  ibv_context* context_;
  // ibverbs protection domain
  ibv_pd* pd_;
  // Completion event channel, to wait for work completions
  ibv_comp_channel* event_channel_;
  // Completion queue, to poll on work completions
  ibv_cq* cq_;
  // Pre-allocated work completions array used for polling
  ibv_wc wc_[MAX_CONCURRENT_WRITES * 2];
  // worker env for thread
  const WorkerEnv* worker_env_;
  // thread for cq.
  std::unique_ptr<Thread> polling_thread_;
};

// Class that represents a connection to a remote Rdma peer.
// Responsible for connecting queue pairs.
class RdmaChannel {
  friend class RdmaAdapter;
  friend class RdmaBuffer;
  friend class RdmaAckBuffer;
  friend class RdmaMessageBuffer;
  friend class RdmaTensorBuffer;
  friend class RdmaMgr;
  friend class RdmaRemoteRendezvous;

 public:
  explicit RdmaChannel(const RdmaAdapter* adapter, const string local_name,
                       const string remote_name_);
  ~RdmaChannel();
  inline const RdmaAddress& self() { return self_; }
  RdmaAddress address() const;
  inline const std::vector<RdmaBuffer*>& message_buffers() const {
    return message_buffers_;
  }
  void Connect(const RdmaAddress& remoteAddr);
  void Connect();
  void Recv();
  RdmaBuffer* FindBuffer(const uint32_t index);
  RdmaBuffer* FindBuffer(const string& name);
  RdmaBuffer* FindOrCreateBuffer(const string& name,
                                 BufferType buffer_type = TENSOR);
  uint32_t LookupBufferIndex(const string& buffer_name);
  void SetRemoteAddress(const RdmaAddress& ra, bool override);
  void InsertRecvCallback(const string& key, std::function<void()> recv_done);
  void RemoveRecvCallback(const string& key);
  void RunRecvCallback(const string& key);
  static const int kNumMessageBuffers = 4;

 protected:
  const RdmaAdapter* adapter_;
  RdmaAddress self_;
  string local_name_;
  string remote_name_;
  ibv_qp* qp_;
  mutex mu_;
  bool connected_ GUARDED_BY(bt_mu_) = false;
  RdmaAddress remote_ GUARDED_BY(bt_mu_);
  bool remote_set_ GUARDED_BY(bt_mu_) = false;
  mutex ct_mu_;
  typedef std::unordered_map<string, std::function<void()> > CallbackTable;
  CallbackTable callback_table_ GUARDED_BY(ct_mu_);
  mutex bt_mu_;
  typedef std::unordered_map<unsigned int, RdmaBuffer*> BufferTable;
  BufferTable buffer_table_ GUARDED_BY(bt_mu_);
  typedef std::unordered_map<uint32_t, string> BufferIndexNameTable;
  BufferIndexNameTable buffer_index_name_table_ GUARDED_BY(bt_mu_);
  typedef std::unordered_map<string, uint32_t> BufferNameIndexTable;
  BufferNameIndexTable buffer_name_index_table_ GUARDED_BY(bt_mu_);
  RdmaBuffer* tx_message_buffer_;
  RdmaBuffer* rx_message_buffer_;
  RdmaBuffer* tx_ack_buffer_;
  RdmaBuffer* rx_ack_buffer_;
  std::vector<RdmaBuffer*> message_buffers_;
};

// Class that represents a buffer for Rdma writes and reads.
class RdmaBuffer {
  friend class RdmaChannel;
  friend class RdmaAdapter;
  friend class RdmaMgr;
  friend class RdmaRemoteRendezvous;

 public:
  explicit RdmaBuffer(RdmaChannel* channel, string name);
  virtual ~RdmaBuffer();

  inline void* buffer() const { return buffer_; }
  inline ibv_mr* self() const { return self_; }
  inline void SetBufferStatus(Location loc, BufferStatus status) {
    mu_.lock();
    if (loc == local) {
      local_status_ = status;
    } else {
      remote_status_ = status;
    }
    mu_.unlock();
  }
  void FreeBuffer();
  void EnqueueItem(string Item);
  virtual void SendNextItem(){};
  void CreateCPUBuffer(size_t size, bool lock = true);
  void SetRemoteMR(RemoteMR rmi, bool override);
  uint32_t LookupBufferIndex(const string& buffer_name) {
    return const_cast<RdmaChannel*>(channel_)->LookupBufferIndex(buffer_name);
  }
  void Write(uint32_t imm_data, size_t buffer_size);

 protected:
  const RdmaChannel* channel_;
  void* buffer_ = nullptr;
  bool buffer_on_host_ = true;
  size_t size_ = 0;
  const string name_;
  ibv_mr* self_ = nullptr;
  mutex mu_;
  RemoteMR remote_;
  std::queue<string> queue_ GUARDED_BY(mu_);
  BufferStatus local_status_ GUARDED_BY(mu_) = none;
  BufferStatus remote_status_ GUARDED_BY(mu_) = none;
};

class RdmaAckBuffer : public RdmaBuffer {
 public:
  explicit RdmaAckBuffer(RdmaChannel* channel, string name);
  virtual ~RdmaAckBuffer() override {}
  void SendNextItem() override;
};

class RdmaMessageBuffer : public RdmaBuffer {
  friend class RdmaChannel;
  friend class RdmaAapater;

 public:
  explicit RdmaMessageBuffer(RdmaChannel* channel, string name);
  virtual ~RdmaMessageBuffer() override {}
  void SendNextItem() override;
};

class RdmaTensorBuffer : public RdmaBuffer {
 public:
  explicit RdmaTensorBuffer(RdmaChannel* channel, string name);
  virtual ~RdmaTensorBuffer() override;
  void SendNextItem() override;
  void PostCopyOperations(bool can_memcpy, size_t buffer_size,
                          size_t tensor_bytes, const string& key,
                          const Tensor& in, int64 step_id, bool is_dead,
                          const string& key_with_step_id, const Tensor* copy,
                          const TensorProto* proto, const StringPiece* copy_buf,
                          const Rendezvous::Args& send_args,
                          const Rendezvous::Args& recv_args);

  void ReSendNextItem();

 private:
  Rendezvous::DoneCallback getRecvTensorCallback(
      const string& key_with_step_id, const string& key, int64 step_id,
      const Rendezvous::ParsedKey& parsed);

  struct ReItem {
    Rendezvous::Args send_args;
    Rendezvous::Args recv_args;
    Tensor in;
    bool is_dead;

    ReItem(const Rendezvous::Args& send_args_,
           const Rendezvous::Args& recv_args_, const Tensor& in_, bool is_dead_)
        : send_args(send_args_),
          recv_args(recv_args_),
          in(in_),
          is_dead(is_dead_) {
      if (send_args.device_context) {
        send_args.device_context->Ref();
      }
      if (recv_args.device_context) {
        recv_args.device_context->Ref();
      }
    }

    ~ReItem() {
      if (send_args.device_context) {
        send_args.device_context->Unref();
      }
      if (recv_args.device_context) {
        recv_args.device_context->Unref();
      }
    }
  };
  typedef std::map<string, ReItem*> Table;
  typedef Table::iterator Itable;

  std::queue<string> requeue GUARDED_BY(mu_);
  Table retable GUARDED_BY(mu_);
};

struct RdmaMessage {
  RdmaMessageType type_;
  uint16_t name_size_;
  string name_;
  int64 step_id_;
  uint64_t buffer_size_;
  uint64_t remote_addr_;
  uint32_t rkey_;
  bool is_dead_;
  DataType data_type_;
  TensorShape tensor_shape_;
  size_t tensor_bytes_;

  // type|name_size|name|step_id|buffer_size|remote_addr|rkey|is_dead|...
  //   1B|    2B   | 512|  8B   |    8B     |       8B  | 4B |    1B |...
  // ...|data_type|tensor_shape|tensor_bytes|tensor_buffer
  // ...|   XB    |    XB      |    8B      |...
  //
  static const size_t kNameCapacity = 512;
  static const size_t kTypeStartIndex = 0;
  static const size_t kNameSizeStartIndex = kTypeStartIndex + sizeof(type_);
  static const size_t kNameStartIndex =
      kNameSizeStartIndex + sizeof(name_size_);
  static const size_t kStepIdStartIndex = kNameStartIndex + kNameCapacity;
  static const size_t kBufferSizeStartIndex =
      kStepIdStartIndex + sizeof(step_id_);
  static const size_t kRemoteAddrStartIndex =
      kBufferSizeStartIndex + sizeof(buffer_size_);
  static const size_t kRkeyStartIndex =
      kRemoteAddrStartIndex + sizeof(remote_addr_);
  static const size_t kIsDeadStartIndex = kRkeyStartIndex + sizeof(rkey_);
  static const size_t kDataTypeStartIndex =
      kIsDeadStartIndex + sizeof(is_dead_);
  static const size_t kTensorShapeStartIndex =
      kDataTypeStartIndex + sizeof(data_type_);
  static const size_t kTensorBytesStartIndex =
      kTensorShapeStartIndex + sizeof(TensorShape);
  static const size_t kTensorBufferStartIndex =
      kTensorBytesStartIndex + sizeof(tensor_bytes_);
  static const size_t kMessageTotalBytes = kTensorBufferStartIndex;
  static const size_t kRdmaMessageBufferSize = kMessageTotalBytes;
  static const size_t kRdmaAckBufferSize = kMessageTotalBytes;
  static string CreateMessage(const RdmaMessage& rm);
  static void ParseMessage(RdmaMessage& rm, void* buffer);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_USE_VERBS
#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_VERBS_RDMA_H_
