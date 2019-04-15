#ifndef TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_CHANNEL_CACHE_H_
#define TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_CHANNEL_CACHE_H_
#include <map>
#include <set>
#include <string>
#include <vector>

#include "tensorflow/core/lib/core/status.h"

namespace seastar {
class channel;
}
namespace tensorflow {
class SeastarEngine;
class SeastarChannelSpec {
public:
  struct HostPortsJob {
    HostPortsJob(const std::string& job_id, const std::map<int, std::string>& host_ports)
        : job_id(job_id), host_ports(host_ports) {}
    const std::string job_id;
    const std::map<int, std::string> host_ports;
  };
  virtual ~SeastarChannelSpec() {}

  Status AddHostPortsJob(const std::string& job_id,
                         const std::map<int, std::string>& host_ports);

  const std::vector<HostPortsJob>& host_ports_jobs() const {
    return host_ports_jobs_;
  }

 private:
  std::vector<HostPortsJob> host_ports_jobs_;
  std::set<std::string> job_ids_;
};

class SeastarChannelCache {
public:
  virtual ~SeastarChannelCache() {}

  virtual void ListWorkers(std::vector<std::string>* workers) const = 0;
  virtual void ListWorkersInJob(const string& job_name,
                                std::vector<string>* workers) = 0; 
  virtual seastar::channel* FindWorkerChannel(const std::string& target) = 0;
  virtual std::string TranslateTask(const std::string& task) = 0;
};

SeastarChannelCache* NewSeastarChannelCache(
    SeastarEngine* engine, const SeastarChannelSpec& channel_spec);
} // tensorflow
#endif // TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_CHANNEL_CACHE_H_
