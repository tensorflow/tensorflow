#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/logging.h"

#include "tensorflow/contrib/seastar/seastar_client_tag.h"
#include "tensorflow/contrib/seastar/seastar_message.h"
#include "tensorflow/contrib/seastar/seastar_tensor_coding.h"

namespace tensorflow {
namespace {
void ProcessCallOptions(SeastarClientTag* tag) {
  if (tag->call_opts_ != nullptr) {
    if (tag->call_opts_->GetTimeout() > 0) {
      tag->timeout_in_ms_ = tag->call_opts_->GetTimeout();
    }
  }
}
}  // namespace

void InitSeastarClientTag(protobuf::Message* request,
                          protobuf::Message* response, StatusCallback done,
                          SeastarClientTag* tag, CallOptions* call_opts) {
  tag->req_body_buf_.len_ = request->ByteSize();
  tag->req_body_buf_.data_ = new char[tag->req_body_buf_.len_]();
  request->SerializeToArray(tag->req_body_buf_.data_, tag->req_body_buf_.len_);

  tag->req_header_buf_.len_ = SeastarClientTag::HEADER_SIZE;
  tag->req_header_buf_.data_ = new char[SeastarClientTag::HEADER_SIZE];

  memcpy(tag->req_header_buf_.data_, "DEADBEEF", 8);
  memcpy(tag->req_header_buf_.data_ + 8, &tag, 8);
  memcpy(tag->req_header_buf_.data_ + 16, &tag->method_, 4);
  memcpy(tag->req_header_buf_.data_ + 24, &tag->req_body_buf_.len_, 8);

  StatusCallback wrapper_done = std::bind(
      [response, tag](StatusCallback done, const Status& s) {
        if (!s.ok()) {
          if (tag->method_ == SeastarWorkerServiceMethod::kLogging ||
              tag->method_ == SeastarWorkerServiceMethod::kTracing) {
            // Logging & Tracing in worker.cc is UNIMPLEMENTED, ignore the error
          } else {
            // Debugging info
            LOG(INFO) << "RPC's status is not ok. status code=" << s.code()
                      << ", err msg=" << s.error_message().c_str();
          }
        } else {
          response->ParseFromArray(tag->resp_body_buf_.data_,
                                   tag->resp_body_buf_.len_);
        }
        done(s);
        delete tag;
      },
      std::move(done), std::placeholders::_1);

  tag->done_ = std::move(wrapper_done);
  tag->call_opts_ = call_opts;
  ProcessCallOptions(tag);
}

void InitSeastarClientTag(protobuf::Message* request,
                          SeastarTensorResponse* response, StatusCallback done,
                          SeastarClientTag* tag, CallOptions* call_opts) {
  tag->req_body_buf_.len_ = request->ByteSize();
  tag->req_body_buf_.data_ = new char[tag->req_body_buf_.len_]();
  request->SerializeToArray(tag->req_body_buf_.data_, tag->req_body_buf_.len_);

  tag->req_header_buf_.len_ = SeastarClientTag::HEADER_SIZE;
  tag->req_header_buf_.data_ = new char[SeastarClientTag::HEADER_SIZE];

  memcpy(tag->req_header_buf_.data_, "DEADBEEF", 8);
  memcpy(tag->req_header_buf_.data_ + 8, &tag, 8);
  memcpy(tag->req_header_buf_.data_ + 16, &tag->method_, 4);
  // Ignore the status segment in request
  // memcpy(tag->req_header_buf_.data_ + 20, &tag->status_, 2);
  memcpy(tag->req_header_buf_.data_ + 24, &tag->req_body_buf_.len_, 8);

  ParseMessageCallback wrapper_parse_message = [request, response, tag]() {
    SeastarMessage sm;
    SeastarMessage::DeserializeMessage(&sm, tag->resp_message_buf_.data_);

    response->SetIsDead(sm.is_dead_);
    response->SetDataType(sm.data_type_);
    bool can_memcpy = DataTypeCanUseMemcpy(sm.data_type_);

    if (can_memcpy) {
      if (response->GetDevice()->tensorflow_gpu_device_info() &&
          (!response->GetOnHost())) {
        AllocatorAttributes alloc_attrs;
        alloc_attrs.set_gpu_compatible(true);
        alloc_attrs.set_on_host(true);
        Allocator* alloc = response->GetDevice()->GetAllocator(alloc_attrs);
        Tensor cpu_copy(alloc, sm.data_type_, sm.tensor_shape_);

        tag->resp_tensor_buf_.data_ =
            reinterpret_cast<char*>(DMAHelper::base(&cpu_copy));
        tag->resp_tensor_buf_.len_ = sm.tensor_bytes_;
        tag->resp_tensor_buf_.owned_ = false;

        response->SetTensor(cpu_copy);

      } else {
        Tensor val(response->GetAlloc(), sm.data_type_, sm.tensor_shape_);
        tag->resp_tensor_buf_.data_ =
            reinterpret_cast<char*>(DMAHelper::base(&val));
        tag->resp_tensor_buf_.len_ = sm.tensor_bytes_;
        tag->resp_tensor_buf_.owned_ = false;

        response->SetTensor(val);
      }
    } else {
      tag->resp_tensor_buf_.len_ = sm.tensor_bytes_;
      tag->resp_tensor_buf_.data_ = new char[tag->resp_tensor_buf_.len_]();
    }

    return Status();
  };
  tag->parse_message_ = std::move(wrapper_parse_message);

  StatusCallback wrapper_done = std::bind(
      [response, tag](StatusCallback done, const Status& s) {
        if (!s.ok()) {
          LOG(ERROR) << "wrapper_done, status not ok. status code=" << s.code()
                     << ", err msg=" << s.error_message().c_str();
          done(s);
          delete tag;
          return;
        }

        bool can_memcpy = DataTypeCanUseMemcpy(response->GetDataType());
        if (can_memcpy) {
          if (response->GetDevice()->tensorflow_gpu_device_info() &&
              (!response->GetOnHost())) {
            Tensor* gpu_copy =
                new Tensor(response->GetAlloc(), response->GetTensor().dtype(),
                           response->GetTensor().shape());
            DeviceContext* recv_dev_context = response->GetDevice()
                                                  ->tensorflow_gpu_device_info()
                                                  ->default_context;
            recv_dev_context->CopyCPUTensorToDevice(
                &response->GetTensor(), response->GetDevice(), gpu_copy,
                [gpu_copy, response, done, tag](const Status& s) {
                  CHECK(s.ok()) << "copy tensor to gpu sync";
                  response->SetTensor(*gpu_copy);
                  done(s);
                  delete gpu_copy;
                  delete tag;
                });
          } else {
            done(s);
            delete tag;
          }
        } else {
          // could not memcopy
          ParseProtoUnlimited(&response->GetTensorProto(),
                              tag->resp_tensor_buf_.data_,
                              tag->resp_tensor_buf_.len_);
          Tensor val;
          Status status = response->GetDevice()->MakeTensorFromProto(
              response->GetTensorProto(), response->GetAllocAttributes(), &val);
          CHECK(status.ok()) << "make cpu tensor from proto.";
          response->SetTensor(val);
          done(status);
          delete tag;
        }
      },
      std::move(done), std::placeholders::_1);

  tag->done_ = std::move(wrapper_done);
  tag->call_opts_ = call_opts;
  ProcessCallOptions(tag);
}

SeastarClientTag::SeastarClientTag(
    tensorflow::SeastarWorkerServiceMethod method, WorkerEnv* env)
    : method_(method), env_(env), resp_err_msg_len_(0), timeout_in_ms_(0) {
  resp_message_buf_.len_ = SeastarMessage::kMessageTotalBytes;
  resp_message_buf_.data_ = new char[resp_message_buf_.len_];
}

SeastarClientTag::~SeastarClientTag() {
  delete[] req_header_buf_.data_;
  delete[] req_body_buf_.data_;
  delete[] resp_body_buf_.data_;

  delete[] resp_message_buf_.data_;
  if (resp_tensor_buf_.owned_) {
    delete[] resp_tensor_buf_.data_;
  }
}

void SeastarClientTag::StartReq(seastar::channel* seastar_channel) {
  seastar_channel->put(ToUserPacket());
}

bool SeastarClientTag::IsRecvTensor() {
  return method_ == SeastarWorkerServiceMethod::kRecvTensor;
}

Status SeastarClientTag::ParseMessage() { return parse_message_(); }

void SeastarClientTag::Schedule(std::function<void()> f) {
  env_->compute_pool->Schedule(std::move(f));
}

void SeastarClientTag::RecvRespDone(Status s) {
  Schedule([this, s]() { done_(s); });
}

uint64_t SeastarClientTag::GetResponseBodySize() { return resp_body_buf_.len_; }

char* SeastarClientTag::GetResponseBodyBuffer() { return resp_body_buf_.data_; }

uint64_t SeastarClientTag::GetResponseMessageSize() {
  return resp_message_buf_.len_;
}

char* SeastarClientTag::GetResponseMessageBuffer() {
  return resp_message_buf_.data_;
}

uint64_t SeastarClientTag::GetResponseTensorSize() {
  return resp_tensor_buf_.len_;
}

char* SeastarClientTag::GetResponseTensorBuffer() {
  return resp_tensor_buf_.data_;
}

seastar::user_packet* SeastarClientTag::ToUserPacket() {
  seastar::net::fragment reqHeader{req_header_buf_.data_, req_header_buf_.len_};
  seastar::net::fragment reqBody{req_body_buf_.data_, req_body_buf_.len_};

  std::vector<seastar::net::fragment> frags = {reqHeader, reqBody};
  seastar::user_packet* up = new seastar::user_packet;
  up->_fragments = frags;
  up->_done = [] {};
  return up;
}

}  // namespace tensorflow
