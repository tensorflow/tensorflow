#include <cstdio>
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

  _listener = seastar::engine().listen(seastar::make_ipv4_address(port), lo);

  seastar::keep_doing([this, tag_factory] {
    return _listener->accept()
    .then([this, tag_factory] (seastar::connected_socket fd,
                               seastar::socket_address addr) mutable {
#ifdef SEASTAR_STATISTIC
      auto conn = new Connection(std::move(fd), tag_factory, addr, &_stat);
#else
      auto conn = new Connection(std::move(fd), tag_factory, addr);
#endif
      seastar::do_until([conn] {return conn->_read_buf.eof(); }, [conn] {
        return conn->Read();
      }).then_wrapped([this, conn] (auto&& f) {
        try {
          f.get();
          LOG(INFO) << "Remote close the connection:  addr = " << conn->_addr;
        } catch (std::exception& ex) {
          LOG(INFO) << "Read got an exception: "
                    << errno << ", addr = " << conn->_addr;
        }
      });
    });
  }).or_terminate();
}
  
seastar::future<> SeastarServer::stop() {
  return seastar::make_ready_future<>();
}

#ifdef SEASTAR_STATISTIC
SeastarServer::Connection::Connection(seastar::connected_socket&& fd,
    SeastarTagFactory* tag_factory,
    seastar::socket_address addr,
    SeastarStat* stat)
  : _tag_factory(tag_factory),
    _addr(addr),
    _stat(stat) {
#else
SeastarServer::Connection::Connection(seastar::connected_socket&& fd,
    SeastarTagFactory* tag_factory, seastar::socket_address addr)
  : _tag_factory(tag_factory),
    _addr(addr) {
#endif
  seastar::ipv4_addr ip_addr(addr);
  boost::asio::ip::address_v4 addr_v4(ip_addr.ip);
  string addr_str = addr_v4.to_string() + ":" + std::to_string(ip_addr.port);
  _channel = new seastar::channel(addr_str);
  _fd = std::move(fd);
  _fd.set_nodelay(true);
  _read_buf = _fd.input();
  _channel->init(seastar::engine().get_packet_queue(), std::move(_fd.output()));
}

SeastarServer::Connection::~Connection() {
  delete _channel;
}

seastar::future<> SeastarServer::Connection::Read() {
  return _read_buf.read_exactly(SeastarServerTag::HEADER_SIZE)
  .then([this] (auto&& header) {
    if (header.size() == 0 ||
        header.size() != SeastarServerTag::HEADER_SIZE) {
      return seastar::make_ready_future();
    }

#ifdef SEASTAR_STATISTIC
    _stat->Request();
#endif

    auto tag = _tag_factory->CreateSeastarServerTag(header, _channel);
    auto req_body_size = tag->GetRequestBodySize();
    if (req_body_size == 0) {
      tag->RecvReqDone(tensorflow::Status());
      return seastar::make_ready_future();
    }
          
    auto req_body_buffer = tag->GetRequestBodyBuffer();
    return _read_buf.read_exactly(req_body_size)
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
