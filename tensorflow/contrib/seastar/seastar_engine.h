#ifndef TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_ENGINE_H_
#define TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_ENGINE_H_

#include <string>
#include <thread>

#include "tensorflow/contrib/seastar/seastar_client.h"
#include "tensorflow/contrib/seastar/seastar_tag_factory.h"
#include "tensorflow/contrib/seastar/seastar_worker_service.h"
#include "third_party/seastar/core/app-template.hh"
#include "third_party/seastar/core/channel.hh"
#include "third_party/seastar/core/distributed.hh"

namespace tensorflow {

class SeastarEngine {
 public:
  SeastarEngine(uint16_t local, SeastarWorkerService* worker_service);
  virtual ~SeastarEngine();

  seastar::channel* GetChannel(const std::string& server_ip);

 private:
  class Server {
   public:
    // Used by Seastar template class distributed<>
    void start(uint16_t port, SeastarTagFactory* tag_factory);
    seastar::future<> stop();

   private:
    struct Connection {
      seastar::connected_socket fd_;
      seastar::input_stream<char> read_buf_;
      seastar::channel* channel_;
      SeastarTagFactory* tag_factory_;
      seastar::socket_address addr_;

      Connection(seastar::connected_socket&& fd, SeastarTagFactory* tag_factory,
                 seastar::socket_address addr);
      seastar::future<> Read();
      ~Connection();
    };

    seastar::lw_shared_ptr<seastar::server_socket> listener_;
  };

  void AsyncStartServer();
  void ConstructArgs(int* argc, char*** argv);
  void GetCpuset(char**);
  seastar::channel* AsyncConnect(const std::string& ip);

  seastar::distributed<Server> server_;
  SeastarClient* client_;
  SeastarTagFactory* tag_factory_;

  std::thread thread_;
  std::string cpuset_;
  uint16_t local_;
  std::atomic_size_t core_id_;
  std::atomic<bool> is_server_ready_;
  size_t core_number_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_ENGINE_H_
