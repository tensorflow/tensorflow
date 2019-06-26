#ifndef TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_ENGINE_H_
#define TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_ENGINE_H_

#include <string>
#include <thread>
#include "core/app-template.hh"
#include "core/distributed.hh"

namespace seastar {
class channel;
}

namespace tensorflow {
class SeastarClient;
class SeastarServer;
class SeastarWorkerService;
class SeastarTagFactory;

using namespace seastar;

class SeastarEngine {
public:
  SeastarEngine(uint16_t local,
                SeastarWorkerService* worker_service);
  virtual ~SeastarEngine();

  seastar::channel* GetChannel(const std::string& server_ip);

private:
  void AsyncStartServer();
  void ConstructArgs(int* argc, char*** argv);
  void GetCpuset(char**);
  seastar::channel* AsyncConnect(const std::string& ip);

  seastar::distributed<SeastarServer> server_;
  SeastarClient* client_;
  SeastarTagFactory* tag_factory_;

  std::thread thread_;
  std::string cpuset_;
  uint16_t local_;
  std::atomic_size_t core_id_;
  std::atomic<bool> is_server_ready_;
  size_t core_number_;
};
}
#endif // TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_ENGINE_H_
