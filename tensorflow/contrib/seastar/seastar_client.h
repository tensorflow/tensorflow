#ifndef TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_CLIENT_H_
#define TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_CLIENT_H_

#include "core/future-util.hh"
#include "net/api.hh"

#include <iostream>
#include <string>

namespace seastar {
class channel;
}

namespace tensorflow {
class SeastarTagFactory;

class SeastarClient {
public:
  void Connect(seastar::ipv4_addr server_addr,
               std::string s,
               seastar::channel* chan,
               SeastarTagFactory* tag_factory);

private:
  struct Connection {
    seastar::connected_socket _fd;
    seastar::input_stream<char> _read_buf;
    seastar::channel* _channel;
    SeastarTagFactory* _tag_factory;
    seastar::socket_address _addr;
    Connection(seastar::connected_socket&& fd,
               seastar::channel* chan,
               SeastarTagFactory* tag_factory,
               seastar::socket_address addr);
    seastar::future<> Read();
  };
};
}
#endif // TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_CLIENT_H_
