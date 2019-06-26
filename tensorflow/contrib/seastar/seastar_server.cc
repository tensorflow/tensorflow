#include "boost/asio/ip/address_v4.hpp"
#include "core/reactor.hh"
#include "core/channel.hh"

#include "tensorflow/contrib/seastar/seastar_server.h"
#include "tensorflow/contrib/seastar/seastar_server_tag.h"
#include "tensorflow/contrib/seastar/seastar_tag_factory.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
void SeastarServer::start(uint16_t port, SeastarTagFactory* tag_factory) {
  seastar::listen_options lo;
  lo.reuse_address = true;

  listener_ = seastar::engine().listen(seastar::make_ipv4_address(port), lo);

  seastar::keep_doing([this, tag_factory] {
    return listener_->accept()
    .then([this, tag_factory] (seastar::connected_socket fd,
                               seastar::socket_address addr) mutable {
      auto conn = new Connection(std::move(fd), tag_factory, addr);
      seastar::do_until([conn] {return conn->read_buf_.eof(); }, [conn] {
        return conn->Read();
      }).then_wrapped([this, conn] (auto&& f) {
        try {
          f.get();
          LOG(INFO) << "Remote close the connection:  addr = " << conn->addr_;
        } catch (std::exception& ex) {
          LOG(INFO) << "Read got an exception: "
                    << errno << ", addr = " << conn->addr_;
        }
      });
    });
  }).or_terminate();
}
  
seastar::future<> SeastarServer::stop() {
  return seastar::make_ready_future<>();
}

SeastarServer::Connection::Connection(seastar::connected_socket&& fd,
    SeastarTagFactory* tag_factory, seastar::socket_address addr)
  : tag_factory_(tag_factory),
    addr_(addr) {
  seastar::ipv4_addr ip_addr(addr);
  boost::asio::ip::address_v4 addr_v4(ip_addr.ip);
  string addr_str = addr_v4.to_string() + ":" + std::to_string(ip_addr.port);
  channel_ = new seastar::channel(addr_str);
  fd_ = std::move(fd);
  fd_.set_nodelay(true);
  read_buf_ = fd_.input();
  channel_->init(seastar::engine().get_packet_queue(), std::move(fd_.output()));
}

SeastarServer::Connection::~Connection() {
  delete channel_;
}

seastar::future<> SeastarServer::Connection::Read() {
  return read_buf_.read_exactly(SeastarServerTag::HEADER_SIZE)
  .then([this] (auto&& header) {
    if (header.size() == 0 ||
        header.size() != SeastarServerTag::HEADER_SIZE) {
      return seastar::make_ready_future();
    }

    auto tag = tag_factory_->CreateSeastarServerTag(header, channel_);
    auto req_body_size = tag->GetRequestBodySize();
    if (req_body_size == 0) {
      tag->RecvReqDone(tensorflow::Status());
      return seastar::make_ready_future();
    }
          
    auto req_body_buffer = tag->GetRequestBodyBuffer();
    return read_buf_.read_exactly(req_body_size)
    .then([this, tag, req_body_size, req_body_buffer] (auto&& body) {
      if (req_body_size != body.size()) {
        LOG(WARNING) << "warning expected body size is:"
                     << req_body_size << ", actual body size:" << body.size();
        tag->RecvReqDone(
          tensorflow::Status(error::UNKNOWN, 
                             "Seastar Server: read invalid msgbuf"));
        return seastar::make_ready_future<>();
      }
              
      memcpy(req_body_buffer, body.get(), body.size());
      tag->RecvReqDone(tensorflow::Status()); 
      return seastar::make_ready_future();
    });
  });
}

}
