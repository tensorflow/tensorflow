#ifndef TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_TAG_FACTORY_H_
#define TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_TAG_FACTORY_H_

#include "tensorflow/contrib/seastar/seastar_client_tag.h"
#include "tensorflow/contrib/seastar/seastar_server_tag.h"
#include "tensorflow/contrib/seastar/seastar_worker_service.h"
#include "third_party/seastar/core/channel.hh"
#include "third_party/seastar/core/temporary_buffer.hh"

namespace tensorflow {

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
