#ifndef TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_CLIENT_TAG_H_
#define TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_CLIENT_TAG_H_

#include <functional>
#include "core/channel.hh"
#include "core/packet_queue.hh"
#include "core/temporary_buffer.hh"

#include "tensorflow/contrib/seastar/seastar_tensor_coding.h"
#include "tensorflow/contrib/seastar/seastar_worker_service_method.h"
#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/worker.pb.h"

namespace tensorflow {
typedef std::function<void(const Status&)> StatusCallback;
typedef std::function<Status()> ParseMessageCallback;

class SeastarWorkerService;
class SeastarTensorResponse;
class SeastarClientTag;
struct WorkerEnv;

void InitSeastarClientTag(protobuf::Message* request,
                          protobuf::Message* response, StatusCallback done,
                          SeastarClientTag* tag, CallOptions* call_opts);

void InitSeastarClientTag(protobuf::Message* request,
                          SeastarTensorResponse* response, StatusCallback done,
                          SeastarClientTag* tag, CallOptions* call_opts);

class SeastarClientTag {
 public:
  // Client Header 32B:
  // |ID:8B|tag:8B|method:4B|reserve:4B|body_len:8B|
  static const uint64_t HEADER_SIZE = 32;
  SeastarClientTag(tensorflow::SeastarWorkerServiceMethod method,
                   WorkerEnv* env);
  virtual ~SeastarClientTag();

  // Called by seatar remote worker, notify seastar engine to send request.
  void StartReq(seastar::channel* seastar_channel);

  bool IsRecvTensor();
  Status ParseMessage();

  // Called by seastar engine, handle the upper layer callback, ex. callback of
  // 'RecvOp'.
  void RecvRespDone(Status s);

  uint64_t GetResponseBodySize();
  char* GetResponseBodyBuffer();

  uint64_t GetResponseMessageSize();
  char* GetResponseMessageBuffer();

  uint64_t GetResponseTensorSize();
  char* GetResponseTensorBuffer();

 private:
  friend class SeastarTagFactory;
  seastar::user_packet* ToUserPacket();
  void Schedule(std::function<void()> f);

 public:
  // Used to handle the upper layer call back when resp recevied.
  StatusCallback done_;
  SeastarWorkerServiceMethod method_;
  WorkerEnv* env_;
  int16_t status_;
  uint16_t resp_err_msg_len_;
  SeastarBuf req_header_buf_;
  SeastarBuf req_body_buf_;
  SeastarBuf resp_body_buf_;
  SeastarBuf resp_message_buf_;
  SeastarBuf resp_tensor_buf_;
  ParseMessageCallback parse_message_;
  CallOptions* call_opts_;
  int timeout_in_ms_;
};

}  // end of namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_CLIENT_TAG_H_
