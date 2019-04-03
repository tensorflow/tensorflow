#include "core/channel.hh"
#include "core/reactor.hh"
#include "core/sleep.hh"
#include "tensorflow/contrib/seastar/seastar_client.h"
#include "tensorflow/contrib/seastar/seastar_client_tag.h"
#include "tensorflow/contrib/seastar/seastar_tag_factory.h"
#include "tensorflow/core/platform/logging.h"

using namespace std::chrono_literals;

namespace tensorflow {
#ifdef SEASTAR_STATISTIC
SeastarClient::Connection::Connection(seastar::connected_socket&& fd,
                                      seastar::channel* chan,
                                      SeastarTagFactory* tag_factory,
                                      seastar::socket_address addr,
                                      SeastarStat* stat)
  : _channel(chan),
    _tag_factory(tag_factory),
    _addr(addr),
    _stat(stat) {
#else
SeastarClient::Connection::Connection(seastar::connected_socket&& fd,
                                      seastar::channel* chan,
                                      SeastarTagFactory* tag_factory,
                                      seastar::socket_address addr)
  : _channel(chan),
    _tag_factory(tag_factory),
    _addr(addr) {
#endif
  _fd = std::move(fd);
  _fd.set_nodelay(true);
  _read_buf = _fd.input();
  _channel->init(seastar::engine().get_packet_queue(), std::move(_fd.output()));
}

seastar::future<> SeastarClient::Connection::Read() {
  return _read_buf.read_exactly(SeastarClientTag::HEADER_SIZE).then([this] (auto&& header) {
      if (header.size() == 0) {
        return seastar::make_ready_future();
      }

      auto tag = _tag_factory->CreateSeastarClientTag(header);
      if (tag->status_ != 0) {
        return _read_buf.read_exactly(tag->resp_err_msg_len_).then([this, tag] (auto&& err_msg) {
          std::string msg = std::string(err_msg.get(), tag->resp_err_msg_len_);
          if (tag->resp_err_msg_len_ == 0) {
            msg = "Empty error msg.";
          }
          tag->RecvRespDone(tensorflow::Status(static_cast<tensorflow::error::Code>(tag->status_),
                                              msg));
          return seastar::make_ready_future();
        });
      }

      if (tag->IsRecvTensor()) { // headle tensor response
        int32_t *fuse_count = new int32_t(tag->GetFuseCount());
        // LOG(INFO) << "fuse_count is: " << *fuse_count;

        int32_t *idx = new int32_t(0);
        bool *error = new bool(false);

        return seastar::do_until(
            [this, tag, fuse_count, idx, error] {
              if (*error || *idx == *fuse_count) {
                delete fuse_count;
                delete idx;
                // NOTE(rangeng.llb): If error happens, tag->RecvRespDone has been called.
                if (!(*error)) {
                  tag->RecvRespDone(tensorflow::Status());
                }
                delete error;
                return true;
              } else {
                return false;
              }
            },
            [this, tag, idx, error] {
              auto message_size = tag->GetResponseMessageSize(*idx);
              auto message_buffer = tag->GetResponseMessageBuffer(*idx);
              return _read_buf.read_exactly(message_size)
                .then([this, tag, idx, error, message_size, message_buffer] (auto&& message) {
                    memcpy(message_buffer, message.get(), message.size());
                    tag->ParseMessage(*idx);
                    auto tensor_size = tag->GetResponseTensorSize(*idx);
                    auto tensor_buffer = tag->GetResponseTensorBuffer(*idx);
                    ++(*idx);
                    
                    if (tensor_size == 0) {
                      return seastar::make_ready_future();
                    }
                    
                    return _read_buf.read_exactly(tensor_size)
                      .then([this, tag, error, tensor_size, tensor_buffer] (auto&& tensor) {
                          if (tensor.size() != tensor_size) {
                            LOG(WARNING) << "warning expected read size is:" << tensor_size
                                         << ", actual read tensor size:" << tensor.size();
                            tag->RecvRespDone(tensorflow::Status(error::UNKNOWN, "Seastar Client: read invalid tensorbuf"));
                            *error = true;
                            return seastar::make_ready_future();
                          }
                          memcpy(tensor_buffer, tensor.get(), tensor.size());
                          return seastar::make_ready_future();
                        });
                  });
            });
      } else {
        // handle no-tensor response
        auto resp_body_size = tag->GetResponseBodySize();
        if (resp_body_size == 0) {
          tag->RecvRespDone(tensorflow::Status());
          return seastar::make_ready_future();
        }

        auto resp_body_buffer = tag->GetResponseBodyBuffer();
        return _read_buf.read_exactly(resp_body_size)
          .then([this, tag, resp_body_size, resp_body_buffer](auto&& body) {
              if (body.size() != resp_body_size) {
                LOG(WARNING) << "warning expected read size is:" << resp_body_size
                             << ", body size:" << body.size();
                tag->RecvRespDone(tensorflow::Status(error::UNKNOWN, "Seastar Client: read invalid msgbuf"));
                return seastar::make_ready_future();
              }
              memcpy(resp_body_buffer, body.get(), body.size());
              tag->RecvRespDone(tensorflow::Status());
              return seastar::make_ready_future();
            });
      }
    });
}

void SeastarClient::start(seastar::ipv4_addr server_addr, std::string s, seastar::channel* chan, SeastarTagFactory* tag_factory) {
  seastar::socket_address local = seastar::socket_address(::sockaddr_in{AF_INET, INADDR_ANY, {0}});
  seastar::engine().net().connect(seastar::make_ipv4_address(server_addr), local, seastar::transport::TCP).then(
      [this, chan, tag_factory, s, server_addr] (seastar::connected_socket fd) {
#ifdef SEASTAR_STATISTIC
      auto conn = new Connection(std::move(fd), chan, tag_factory, seastar::socket_address(server_addr), &_stat);
#else
      auto conn = new Connection(std::move(fd), chan, tag_factory, seastar::socket_address(server_addr));
#endif

      //LOG(INFO) << "connected...." << s;
      seastar::do_until([conn] {return conn->_read_buf.eof(); }, [conn] {
        return conn->Read();
      }).then_wrapped([this, conn, s, chan] (auto&& f) {
        try {
          f.get();
          LOG(INFO) << "Remote closed the connection: addr = " << s;
        } catch(std::exception& ex) {
          LOG(INFO) << "Read got an exception: " << errno << ", addr = " << s;
        }
        LOG(INFO) << "Set channel broken, connection:" << s;
        chan->set_channel_broken();
      });

      return seastar::make_ready_future();
    }).handle_exception([this, chan, tag_factory, server_addr, s](auto ep) {
      return seastar::sleep(1s).then([this, chan, tag_factory, server_addr, s] {
        //LOG(INFO) << "connected failure...." << s;
        this->start(server_addr, s, chan, tag_factory);
      });
    });
}

seastar::future<> SeastarClient::stop() {
  return seastar::make_ready_future();
}

}
