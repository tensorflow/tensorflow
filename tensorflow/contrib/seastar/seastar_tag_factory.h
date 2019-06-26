#ifndef TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_TAG_FACTORY_H_
#define TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_TAG_FACTORY_H_

#include "core/temporary_buffer.hh"

namespace seastar {
class channel;
}
namespace tensorflow {
class SeastarClientTag;
class SeastarServerTag;
class SeastarWorkerService;

class SeastarTagFactory {
 public:
  explicit SeastarTagFactory(SeastarWorkerService* worker_service);
  virtual ~SeastarTagFactory() {}

  SeastarClientTag* CreateSeastarClientTag(
      seastar::temporary_buffer<char>& header);

  SeastarServerTag* CreateSeastarServerTag(
      seastar::temporary_buffer<char>& header,
      seastar::channel* seastar_channel);

 private:
  SeastarWorkerService* worker_service_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_TAG_FACTORY_H_
