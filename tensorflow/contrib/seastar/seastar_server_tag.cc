#include "tensorflow/contrib/seastar/seastar_server_tag.h"

#include "tensorflow/contrib/seastar/seastar_message.h"
#include "tensorflow/contrib/seastar/seastar_worker_service.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

namespace {

void SerializeTensorMessage(const Tensor& in, const TensorProto& inp,
                            bool is_dead, SeastarBuf* message_buf,
                            SeastarBuf* tensor_buf) {
  SeastarMessage sm;
  sm.tensor_shape_ = in.shape();
  sm.data_type_ = in.dtype();
  sm.is_dead_ = is_dead;

  bool can_memcpy = DataTypeCanUseMemcpy(sm.data_type_);
  if (can_memcpy) {
    sm.tensor_bytes_ = in.TotalBytes();
    tensor_buf->len_ = sm.tensor_bytes_;
    tensor_buf->data_ = const_cast<char*>(in.tensor_data().data());
    tensor_buf->owned_ = false;

  } else {
    sm.tensor_bytes_ = inp.ByteSize();
    tensor_buf->len_ = sm.tensor_bytes_;
    tensor_buf->data_ = new char[tensor_buf->len_]();
    inp.SerializeToArray(tensor_buf->data_, tensor_buf->len_);
  }

  message_buf->len_ = SeastarMessage::kMessageTotalBytes;
  message_buf->data_ = new char[message_buf->len_];
  SeastarMessage::SerializeMessage(sm, message_buf->data_);
}

}  // namespace

void InitSeastarServerTag(protobuf::Message* request,
                          protobuf::Message* response, SeastarServerTag* tag) {
  request->ParseFromArray(tag->req_body_buf_.data_, tag->req_body_buf_.len_);

  StatusCallback done = [response, tag](const Status& s) {
    tag->resp_header_buf_.len_ = SeastarServerTag::HEADER_SIZE;
    if (!s.ok()) {
      tag->resp_header_buf_.len_ += s.error_message().length();
    }
    tag->resp_header_buf_.data_ = new char[tag->resp_header_buf_.len_]();

    memcpy(tag->resp_header_buf_.data_, "CAFEBABE", 8);
    memcpy(tag->resp_header_buf_.data_ + 8, &tag->client_tag_id_, 8);
    memcpy(tag->resp_header_buf_.data_ + 16, &tag->method_, 4);
    int16_t code = static_cast<int16_t>(s.code());
    memcpy(tag->resp_header_buf_.data_ + 20, &code, 2);

    if (s.ok()) {
      uint16_t err_len = 0;
      memcpy(tag->resp_header_buf_.data_ + 22, &err_len, 2);
      tag->resp_body_buf_.len_ = response->ByteSize();
      tag->resp_body_buf_.data_ = new char[tag->resp_body_buf_.len_]();
      response->SerializeToArray(tag->resp_body_buf_.data_,
                                 tag->resp_body_buf_.len_);

      memcpy(tag->resp_header_buf_.data_ + 24, &tag->resp_body_buf_.len_, 8);
    } else {
      // TODO: RemoteWorker::LoggingRequest doesn't need to response.
      //      can be more elegant.
      uint16_t err_len =
          std::min(UINT16_MAX, (int)(s.error_message().length()));
      memcpy(tag->resp_header_buf_.data_ + 22, &err_len, 2);
      memcpy(tag->resp_header_buf_.data_ + 32, s.error_message().c_str(),
             err_len);
    }

    tag->StartResp();
  };

  tag->send_resp_ = std::move(done);
  tag->clear_ = [](const Status& s) {};
}

void InitSeastarServerTag(protobuf::Message* request,
                          SeastarTensorResponse* response,
                          SeastarServerTag* tag, StatusCallback clear) {
  request->ParseFromArray(tag->req_body_buf_.data_, tag->req_body_buf_.len_);

  StatusCallback done = [request, response, tag](const Status& s) {
    tag->resp_header_buf_.len_ = SeastarServerTag::HEADER_SIZE;
    if (!s.ok()) {
      tag->resp_header_buf_.len_ += s.error_message().length();
    }
    tag->resp_header_buf_.data_ = new char[tag->resp_header_buf_.len_]();

    memcpy(tag->resp_header_buf_.data_, "CAFEBABE", 8);
    memcpy(tag->resp_header_buf_.data_ + 8, &tag->client_tag_id_, 8);
    memcpy(tag->resp_header_buf_.data_ + 16, &tag->method_, 4);
    int16_t code = static_cast<int16_t>(s.code());
    memcpy(tag->resp_header_buf_.data_ + 20, &code, 2);

    if (s.ok()) {
      uint16_t err_len = 0;
      memcpy(tag->resp_header_buf_.data_ + 22, &err_len, 2);
      SerializeTensorMessage(response->GetTensor(), response->GetTensorProto(),
                             response->GetIsDead(), &tag->resp_message_buf_,
                             &tag->resp_tensor_buf_);
    } else {
      uint16_t err_len =
          std::min(UINT16_MAX, (int)(s.error_message().length()));
      memcpy(tag->resp_header_buf_.data_ + 22, &err_len, 2);
      memcpy(tag->resp_header_buf_.data_ + 32, s.error_message().c_str(),
             err_len);
    }

    tag->StartRespWithTensor();
  };

  // used for zero copy sending tensor
  tag->clear_ = std::move(clear);
  tag->send_resp_ = std::move(done);
}

SeastarServerTag::SeastarServerTag(seastar::channel* seastar_channel,
                                   SeastarWorkerService* seastar_worker_service)
    : seastar_channel_(seastar_channel),
      seastar_worker_service_(seastar_worker_service) {}

SeastarServerTag::~SeastarServerTag() {
  delete[] req_body_buf_.data_;
  delete[] resp_header_buf_.data_;
  delete[] resp_body_buf_.data_;

  delete[] resp_message_buf_.data_;
  if (resp_tensor_buf_.owned_) {
    delete[] resp_tensor_buf_.data_;
  }
}

// Called by seastar engine, call the handler.
void SeastarServerTag::RecvReqDone(Status s) {
  if (!s.ok()) {
    this->send_resp_(s);
    // TODO(handle clear)
    return;
  }

  SeastarWorkerService::HandleRequestFunction handle =
      seastar_worker_service_->GetHandler(method_);
  (seastar_worker_service_->*handle)(this);
}

// Called by seastar engine.
void SeastarServerTag::SendRespDone() {
  clear_(Status());
  delete this;
}

// Serialize and send response.
void SeastarServerTag::ProcessDone(Status s) {
  // LOG(INFO) << "enter seastarServerTag::ProcessDone";
  send_resp_(s);
}

uint64_t SeastarServerTag::GetRequestBodySize() { return req_body_buf_.len_; }

char* SeastarServerTag::GetRequestBodyBuffer() { return req_body_buf_.data_; }

void SeastarServerTag::StartResp() { seastar_channel_->put(ToUserPacket()); }

void SeastarServerTag::StartRespWithTensor() {
  seastar_channel_->put(ToUserPacketWithTensor());
}

seastar::user_packet* SeastarServerTag::ToUserPacket() {
  seastar::net::fragment respHeader{resp_header_buf_.data_,
                                    resp_header_buf_.len_};
  seastar::net::fragment respBody{resp_body_buf_.data_, resp_body_buf_.len_};

  std::vector<seastar::net::fragment> frags = {respHeader, respBody};
  seastar::user_packet* up = new seastar::user_packet;
  up->_fragments = frags;
  up->_done = [this]() { this->SendRespDone(); };
  return up;
}

seastar::user_packet* SeastarServerTag::ToUserPacketWithTensor() {
  auto up = new seastar::user_packet;
  std::vector<seastar::net::fragment> frags;
  frags.emplace_back(
      seastar::net::fragment{resp_header_buf_.data_, resp_header_buf_.len_});

  frags.emplace_back(
      seastar::net::fragment{resp_message_buf_.data_, resp_message_buf_.len_});

  if (resp_tensor_buf_.len_ > 0) {
    frags.emplace_back(
        seastar::net::fragment{resp_tensor_buf_.data_, resp_tensor_buf_.len_});
  }
  up->_fragments = frags;
  up->_done = [this]() { this->SendRespDone(); };
  return up;
}

}  // namespace tensorflow
