#ifndef TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_CLIENT_H_
#define TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_CLIENT_H_

#include "core/future-util.hh"
#include "net/api.hh"

namespace seastar {
class channel;
}
namespace tensorflow {
class SeastarTagFactory;

class SeastarClient {
 public:
  void Connect(seastar::ipv4_addr server_addr, std::string s,
               seastar::channel* chan, SeastarTagFactory* tag_factory);

 private:
  struct Connection {
    seastar::connected_socket fd_;
    seastar::input_stream<char> read_buf_;
    seastar::channel* channel_;
    SeastarTagFactory* tag_factory_;
    seastar::socket_address addr_;
    Connection(seastar::connected_socket&& fd, seastar::channel* chan,
               SeastarTagFactory* tag_factory, seastar::socket_address addr);
    seastar::future<> Read();
  };
};
}
#endif  // TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_CLIENT_H_
