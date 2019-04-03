#include <assert.h>
#include <climits>
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/contrib/seastar/seastar_message.h"
#include "tensorflow/contrib/seastar/seastar_server_tag.h"
#include "tensorflow/contrib/seastar/seastar_tensor_coding.h"
#include "tensorflow/contrib/seastar/seastar_worker_service.h"

namespace tensorflow {
namespace {

void SerializeTensorMessage(const Tensor& in, const TensorProto& inp, bool is_dead,
                            SeastarBuf* message_buf, SeastarBuf* tensor_buf)
{
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

} // end of anonymous namespace

void InitSeastarServerTag(protobuf::Message* request,
			  protobuf::Message* response,
			  SeastarServerTag* tag) {
  request->ParseFromArray(tag->req_body_buf_.data_,
                          tag->req_body_buf_.len_);

  StatusCallback done = [response, tag] (const Status& s) {
    tag->resp_header_buf_.len_ = SeastarServerTag::HEADER_SIZE;
    if (!s.ok()) {
      tag->resp_header_buf_.len_ += s.error_message().length();
    }
    tag->resp_header_buf_.data_ = new char[tag->resp_header_buf_.len_]();

    memcpy(tag->resp_header_buf_.data_, "BBBBBBBB", 8);
    memcpy(tag->resp_header_buf_.data_ + 8, &tag->client_tag_id_, 8);
    memcpy(tag->resp_header_buf_.data_ + 16, &tag->method_, 4);
    int16_t code = static_cast<int16_t>(s.code());
    memcpy(tag->resp_header_buf_.data_ + 20, &code, 2);

    if (s.ok()) {
      tag->resp_body_buf_.len_ = response->ByteSize();
      tag->resp_body_buf_.data_ = new char[tag->resp_body_buf_.len_]();
      response->SerializeToArray(tag->resp_body_buf_.data_, tag->resp_body_buf_.len_);

      memcpy(tag->resp_header_buf_.data_ + 24, &tag->resp_body_buf_.len_, 8);  
    } else {
      //TODO: RemoteWorker::LoggingRequest doesn't need to response.
      //      can be more elegant.
      //LOG(INFO) << "process done with NOT ok" 
      //         << ", response:" << response->DebugString()
      //         << ", status message:" << s.error_message();

      // Send err msg back to client
      uint16_t err_len = std::min(UINT16_MAX, (int)(s.error_message().length()));
      memcpy(tag->resp_header_buf_.data_ + 22, &err_len, 2);
      memcpy(tag->resp_header_buf_.data_ + 32, s.error_message().c_str(), err_len);
    }

    tag->StartResp();
  };

  tag->send_resp_ = std::move(done);
  tag->clear_ = [](const Status& s) {};
}

void InitSeastarServerTag(protobuf::Message* request,
			  SeastarTensorResponse* response,
			  SeastarServerTag* tag,
                          StatusCallback clear) {
  request->ParseFromArray(tag->req_body_buf_.data_, tag->req_body_buf_.len_);

  StatusCallback done = [request, response, tag] (const Status& s) {
    tag->resp_header_buf_.len_ = SeastarServerTag::HEADER_SIZE;
    if (!s.ok()) {
      tag->resp_header_buf_.len_ += s.error_message().length();
    }
    tag->resp_header_buf_.data_ = new char[tag->resp_header_buf_.len_]();

    memcpy(tag->resp_header_buf_.data_, "BBBBBBBB", 8);
    memcpy(tag->resp_header_buf_.data_ + 8, &tag->client_tag_id_, 8);
    memcpy(tag->resp_header_buf_.data_ + 16, &tag->method_, 4);
    int16_t code = static_cast<int16_t>(s.code());
    memcpy(tag->resp_header_buf_.data_ + 20, &code, 2);

    if (s.ok()) {
      tag->InitFuse(1);
      SerializeTensorMessage(response->GetTensor(), response->GetTensorProto(),
                             response->GetIsDead(), &tag->resp_message_bufs_[0],
                             &tag->resp_tensor_bufs_[0]);
      // for tensor response, the 'length' segment indicates the fuse tensor count
      uint64_t fuse_count = 1;
      memcpy(tag->resp_header_buf_.data_ + 24, &fuse_count, 8);
    } else {
      //LOG(WARNING) << "InitSeastarServerTag callback process done with failure:" << s.error_message();

      uint16_t err_len = std::min(UINT16_MAX, (int)(s.error_message().length()));
      memcpy(tag->resp_header_buf_.data_ + 22, &err_len, 2);
      memcpy(tag->resp_header_buf_.data_ + 32, s.error_message().c_str(), err_len);
    }

    tag->StartRespWithTensors();
  };

  // used for zero copy sending tensor
  tag->clear_ = std::move(clear);
  tag->send_resp_ = std::move(done);
}

void InitSeastarServerTag(protobuf::Message* request,
			  SeastarFuseTensorResponse* response,
			  SeastarServerTag* tag,
                          StatusCallback clear) {
  // LOG(INFO) << "InitSeastarServerTag for fuse recv";
  request->ParseFromArray(tag->req_body_buf_.data_, tag->req_body_buf_.len_);

  StatusCallback done = [request, response, tag] (const Status& s) {
    tag->resp_header_buf_.len_ = SeastarServerTag::HEADER_SIZE;
    if (!s.ok()) {
      tag->resp_header_buf_.len_ += s.error_message().length();
    }
    tag->resp_header_buf_.data_ = new char[tag->resp_header_buf_.len_]();

    memcpy(tag->resp_header_buf_.data_, "BBBBBBBB", 8);
    memcpy(tag->resp_header_buf_.data_ + 8, &tag->client_tag_id_, 8);
    memcpy(tag->resp_header_buf_.data_ + 16, &tag->method_, 4);
    int16_t code = static_cast<int16_t>(s.code());
    memcpy(tag->resp_header_buf_.data_ + 20, &code, 2);

    if (s.ok()) {
      tag->InitFuse(response->GetFuseCount());
      for (int idx = 0; idx < tag->fuse_count_; ++idx) {
        SerializeTensorMessage(response->GetTensorByIndex(idx),
                               response->GetTensorProtoByIndex(idx),
                               response->GetIsDeadByIndex(idx),
                               &tag->resp_message_bufs_[idx],
                               &tag->resp_tensor_bufs_[idx]);
      }
      uint64_t fuse_count = tag->fuse_count_;
      memcpy(tag->resp_header_buf_.data_ + 24, &fuse_count, 8);
    } else {
      //LOG(WARNING) << "InitSeastarServerTag callback process done with failure:"
      //             << s.error_message();
      uint16_t err_len = std::min(UINT16_MAX, (int)(s.error_message().length()));
      memcpy(tag->resp_header_buf_.data_ + 22, &err_len, 2);
      memcpy(tag->resp_header_buf_.data_ + 32, s.error_message().c_str(), err_len);
    }

    tag->StartRespWithTensors();
  };

  // used for zero copy sending tensor, unref Tensor object after seastar send done
  tag->clear_ = std::move(clear);
  tag->send_resp_ = std::move(done);
}

SeastarServerTag::SeastarServerTag(seastar::channel* seastar_channel,
                                   SeastarWorkerService* seastar_worker_service)
  : fuse_count_(0),
    seastar_channel_(seastar_channel),
    seastar_worker_service_(seastar_worker_service) {
}

SeastarServerTag::~SeastarServerTag() {
  delete [] req_body_buf_.data_;
  delete [] resp_header_buf_.data_;
  delete [] resp_body_buf_.data_;
  
  for (int i = 0; i < fuse_count_; ++i) {
    delete [] resp_message_bufs_[i].data_;
    if (resp_tensor_bufs_[i].owned_) {
      delete [] resp_tensor_bufs_[i].data_;
    }
  }
}

void SeastarServerTag::InitFuse(int32_t fuse_count) {
  fuse_count_ = fuse_count;
  resp_message_bufs_.resize(fuse_count);
  resp_tensor_bufs_.resize(fuse_count);
}

// Called by seastar engine, call the handler.
void SeastarServerTag::RecvReqDone(Status s) {
  if (!s.ok()) {
    this->send_resp_(s);
    // TODO(handle clear)
    return;
  }

  HandleRequestFunction handle = seastar_worker_service_->GetHandler(method_);
  (seastar_worker_service_->*handle)(this);
}

// Called by seastar engine.
void SeastarServerTag::SendRespDone() {
  clear_(Status());
  delete this;
}

// called when request has been processed, mainly serialize resp to wire-format,
// and send response
void SeastarServerTag::ProcessDone(Status s) {
  //LOG(INFO) << "enter seastarServerTag::ProcessDone";
  send_resp_(s);
}

uint64_t SeastarServerTag::GetRequestBodySize() {
  return req_body_buf_.len_;
}

char* SeastarServerTag::GetRequestBodyBuffer() {
  return req_body_buf_.data_;
}

void SeastarServerTag::StartResp() {
  seastar_channel_->put(ToUserPacket());
}

void SeastarServerTag::StartRespWithTensors() {
  seastar_channel_->put(ToUserPacketWithTensors());
}

seastar::user_packet* SeastarServerTag::ToUserPacket() {
  seastar::net::fragment respHeader {resp_header_buf_.data_, resp_header_buf_.len_};
  seastar::net::fragment respBody {resp_body_buf_.data_, resp_body_buf_.len_};

  std::vector<seastar::net::fragment> frags = { respHeader, respBody };
  seastar::user_packet* up = new seastar::user_packet;
  up->_fragments = frags;
  up->_done = [this](){ this->SendRespDone(); };
  return up;
}

/*seastar::user_packet* SeastarServerTag::ToUserPacketWithTensors() {
  seastar::net::fragment respHeader {resp_header_buf_.data_, resp_header_buf_.len_};

  std::vector<seastar::net::fragment> frags;
  frags.emplace_back(respHeader);

  for (int i = 0; i < fuse_count_; ++i) {
    seastar::net::fragment respMessage {resp_message_bufs_[i].data_,
        resp_message_bufs_[i].len_};
    frags.emplace_back(respMessage);

    if (resp_tensor_bufs_[i].len_ > 0) {
      seastar::net::fragment respTensor {resp_tensor_bufs_[i].data_,
          resp_tensor_bufs_[i].len_};
      frags.emplace_back(respTensor);
    }
  }

  seastar::user_packet* up = new seastar::user_packet;
  up->_fragments = frags;
  up->_done = [this](){ this->SendRespDone(); };
  return up;
}*/

std::vector<seastar::user_packet*> SeastarServerTag::ToUserPacketWithTensors() {
  std::vector<seastar::user_packet*> ret;
  auto up = new seastar::user_packet;

  std::vector<seastar::net::fragment> frags;
  frags.emplace_back(seastar::net::fragment {resp_header_buf_.data_,
      resp_header_buf_.len_});

  for (auto i = 0; i < fuse_count_; ++i) {
    if (frags.size() > IOV_MAX / 2) {
      std::swap(up->_fragments, frags);
      auto left_frags = (fuse_count_ - i) * 2;

      up->_done = []() {};
      ret.emplace_back(up);
      up = new seastar::user_packet;
    }
    frags.emplace_back(seastar::net::fragment {resp_message_bufs_[i].data_,
        resp_message_bufs_[i].len_});

    if (resp_tensor_bufs_[i].len_ > 0) {
      frags.emplace_back(seastar::net::fragment {resp_tensor_bufs_[i].data_,
        resp_tensor_bufs_[i].len_});
    }
  }
  up->_fragments = frags;
  up->_done = [this]() { this->SendRespDone(); };
  ret.emplace_back(up);
  return ret;
}

} // end of namespace tensorflow
