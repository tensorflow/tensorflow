#ifndef TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_CLIENT_H_
#define TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_CLIENT_H_

#include "core/future-util.hh"
#include "net/api.hh"

#include <iostream>
#include <string>

#ifdef SEASTAR_STATISTIC
#include "tensorflow/contrib/seastar/seastar_stat.h"
#endif

namespace seastar {
class channel;
}

namespace tensorflow {
class SeastarTagFactory;

class SeastarClient {
public:
  // should named start & stop which used by seastar template class distributed<>
  void start(seastar::ipv4_addr server_addr,
             std::string s,
             seastar::channel* chan,
             SeastarTagFactory* tag_factory);

  seastar::future<> stop();

private:
  struct Connection {
    seastar::connected_socket _fd;
    seastar::input_stream<char> _read_buf;
    seastar::channel* _channel;
    SeastarTagFactory* _tag_factory;
    seastar::socket_address _addr;
#ifdef SEASTAR_STATISTIC
    SeastarStat* _stat;
    Connection(seastar::connected_socket&& fd,
               seastar::channel* chan,
               SeastarTagFactory* tag_factory,
               seastar::socket_address addr,
               SeastarStat* stat);
#else
    Connection(seastar::connected_socket&& fd,
               seastar::channel* chan,
               SeastarTagFactory* tag_factory,
               seastar::socket_address addr);
#endif
    seastar::future<> Read();
  };
#ifdef SEASTAR_STATISTIC
private:
  SeastarStat _stat;
#endif
};
}
#endif // TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_CLIENT_H_
