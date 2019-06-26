#include "tensorflow/contrib/seastar/seastar_client.h"
#include "core/channel.hh"
#include "core/reactor.hh"
#include "core/sleep.hh"
#include "tensorflow/contrib/seastar/seastar_client_tag.h"
#include "tensorflow/contrib/seastar/seastar_tag_factory.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
SeastarClient::Connection::Connection(seastar::connected_socket&& fd,
                                      seastar::channel* chan,
                                      SeastarTagFactory* tag_factory,
                                      seastar::socket_address addr)
    : channel_(chan), tag_factory_(tag_factory), addr_(addr) {
  fd_ = std::move(fd);
  fd_.set_nodelay(true);
  read_buf_ = fd_.input();
  channel_->init(seastar::engine().get_packet_queue(), std::move(fd_.output()));
}

seastar::future<> SeastarClient::Connection::Read() {
  return read_buf_.read_exactly(SeastarClientTag::HEADER_SIZE)
      .then([this](auto&& header) {
        if (header.size() == 0) {
          return seastar::make_ready_future();
        }
        auto tag = tag_factory_->CreateSeastarClientTag(header);
        if (tag->status_ != 0) {
          return read_buf_.read_exactly(tag->resp_err_msg_len_)
              .then([this, tag](auto&& err_msg) {
                std::string msg =
                    std::string(err_msg.get(), tag->resp_err_msg_len_);
                if (tag->resp_err_msg_len_ == 0) {
                  msg = "Empty error msg.";
                }
                tag->RecvRespDone(
                    Status(static_cast<error::Code>(tag->status_), msg));
                return seastar::make_ready_future();
              });
        }

        if (tag->IsRecvTensor()) {
          // handle tensor response
          auto message_size = tag->GetResponseMessageSize();
          auto message_buffer = tag->GetResponseMessageBuffer();
          return read_buf_.read_exactly(message_size)
              .then([this, tag, message_size, message_buffer](auto&& message) {
                memcpy(message_buffer, message.get(), message.size());
                tag->ParseMessage();
                auto tensor_size = tag->GetResponseTensorSize();
                auto tensor_buffer = tag->GetResponseTensorBuffer();
                if (tensor_size == 0) {
                  tag->RecvRespDone(tensorflow::Status());
                  return seastar::make_ready_future();
                }
                return read_buf_.read_exactly(tensor_size)
                    .then(
                        [this, tag, tensor_size, tensor_buffer](auto&& tensor) {
                          if (tensor.size() != tensor_size) {
                            LOG(WARNING)
                                << "Expected read size is:" << tensor_size
                                << ", but real tensor size:" << tensor.size();
                            tag->RecvRespDone(Status(
                                error::UNKNOWN,
                                "Seastar Client: read invalid tensorbuf"));
                            return seastar::make_ready_future();
                          }
                          memcpy(tensor_buffer, tensor.get(), tensor.size());
                          tag->RecvRespDone(tensorflow::Status());
                          return seastar::make_ready_future();
                        });
              });
        } else {
          // handle general response
          auto resp_body_size = tag->GetResponseBodySize();
          if (resp_body_size == 0) {
            tag->RecvRespDone(tensorflow::Status());
            return seastar::make_ready_future();
          }

          auto resp_body_buffer = tag->GetResponseBodyBuffer();
          return read_buf_.read_exactly(resp_body_size)
              .then([this, tag, resp_body_size, resp_body_buffer](auto&& body) {
                if (body.size() != resp_body_size) {
                  LOG(WARNING) << "Expected read size is:" << resp_body_size
                               << ", but real size is:" << body.size();
                  tag->RecvRespDone(tensorflow::Status(
                      error::UNKNOWN, "Seastar Client: read invalid msgbuf"));
                  return seastar::make_ready_future();
                }
                memcpy(resp_body_buffer, body.get(), body.size());
                tag->RecvRespDone(tensorflow::Status());
                return seastar::make_ready_future();
              });
        }
      });
}

void SeastarClient::Connect(seastar::ipv4_addr server_addr, std::string s,
                            seastar::channel* chan,
                            SeastarTagFactory* tag_factory) {
  seastar::socket_address local =
      seastar::socket_address(::sockaddr_in{AF_INET, INADDR_ANY, {0}});

  seastar::engine()
      .net()
      .connect(seastar::make_ipv4_address(server_addr), local,
               seastar::transport::TCP)
      .then([this, chan, s, server_addr,
             tag_factory](seastar::connected_socket fd) {
        auto conn = new Connection(std::move(fd), chan, tag_factory,
                                   seastar::socket_address(server_addr));

        seastar::do_until([conn] { return conn->read_buf_.eof(); },
                          [conn] { return conn->Read(); })
            .then_wrapped([this, conn, s, chan](auto&& f) {
              try {
                f.get();
                VLOG(2) << "Remote closed the connection: addr = " << s;
              } catch (std::exception& ex) {
                LOG(WARNING) << "Read got an exception: " << errno
                             << ", addr = " << s;
              }
              chan->set_channel_broken();
            });
        return seastar::make_ready_future();
      })
      .handle_exception([this, chan, server_addr, s, tag_factory](auto ep) {
        using namespace std::chrono_literals;
        return seastar::sleep(1s).then(
            [this, chan, server_addr, s, tag_factory] {
              this->Connect(server_addr, s, chan, tag_factory);
            });
      });
}
}
