#include "tensorflow/contrib/seastar/seastar_tag_factory.h"

#include "tensorflow/contrib/seastar/seastar_message.h"
#include "tensorflow/contrib/seastar/seastar_tensor_coding.h"

namespace tensorflow {

SeastarTagFactory::SeastarTagFactory(SeastarWorkerService* worker_service)
    : worker_service_(worker_service) {}

SeastarClientTag* SeastarTagFactory::CreateSeastarClientTag(
    seastar::temporary_buffer<char>& header) {
  char* p = const_cast<char*>(header.get());
  SeastarClientTag* tag = nullptr;
  memcpy(&tag, p + 8, 8);
  // ignore the method segment 4B
  memcpy(&tag->status_, p + 20, 2);
  memcpy(&tag->resp_err_msg_len_, p + 22, 2);

  if (!tag->IsRecvTensor()) {
    memcpy(&tag->resp_body_buf_.len_, p + 24, 8);
    tag->resp_body_buf_.data_ = new char[tag->resp_body_buf_.len_];
  }
  return tag;
}

SeastarServerTag* SeastarTagFactory::CreateSeastarServerTag(
    seastar::temporary_buffer<char>& header,
    seastar::channel* seastar_channel) {
  char* p = const_cast<char*>(header.get());
  SeastarServerTag* tag =
      new SeastarServerTag(seastar_channel, worker_service_);
  memcpy(&tag->client_tag_id_, p + 8, 8);
  memcpy(&tag->method_, p + 16, 4);
  // ignore the status segment 2B
  memcpy(&(tag->req_body_buf_.len_), p + 24, 8);
  tag->req_body_buf_.data_ = new char[tag->req_body_buf_.len_];

  return tag;
}

}  // namespace tensorflow
