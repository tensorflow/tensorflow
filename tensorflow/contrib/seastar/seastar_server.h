#ifndef TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_SERVER_H_
#define TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_SERVER_H_
#include "core/channel.hh"
#include "core/future-util.hh"
#include "core/shared_ptr.hh"
#include "net/api.hh"

namespace seastar {
class channel;
}
namespace tensorflow {
class SeastarTagFactory;
class SeastarServer {
 public:
  // here should named start & stop, which used by seastar template class distributed<>
  void start(uint16_t port, SeastarTagFactory* tag_factory);
  seastar::future<> stop();

 private:
  struct Connection {
    seastar::connected_socket fd_;
    seastar::input_stream<char> read_buf_;
    seastar::channel* channel_;
    SeastarTagFactory* tag_factory_;
    seastar::socket_address addr_;

    Connection(seastar::connected_socket&& fd,
               SeastarTagFactory* tag_factory,
               seastar::socket_address addr);
    seastar::future<> Read();
    ~Connection();
  };

 private:
  seastar::lw_shared_ptr<seastar::server_socket> listener_;
};
}
#endif // TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_SERVER_H_
