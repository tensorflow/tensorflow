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
  seastar::lw_shared_ptr<seastar::server_socket> _listener;
public:
  // here should named start & stop, which used by seastar template class distributed<>
  void start(uint16_t port, SeastarTagFactory* tag_factory);
  seastar::future<> stop();

private:
  struct Connection {
    seastar::connected_socket _fd;
    seastar::input_stream<char> _read_buf;
    seastar::channel* _channel;
    SeastarTagFactory* _tag_factory;
    seastar::socket_address _addr;

    Connection(seastar::connected_socket&& fd,
               SeastarTagFactory* tag_factory,
               seastar::socket_address addr);
    seastar::future<> Read();
    ~Connection();
  };
};
}
#endif // TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_SERVER_H_
