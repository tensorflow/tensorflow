#ifndef TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_CLIENT_H_
#define TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_CLIENT_H_

#include "tensorflow/contrib/seastar/seastar_tag_factory.h"
#include "third_party/seastar/core/channel.hh"
#include "third_party/seastar/core/future-util.hh"
#include "third_party/seastar/net/api.hh"

namespace tensorflow {

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

}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_CLIENT_H_
