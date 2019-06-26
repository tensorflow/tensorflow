#ifndef TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_WOKRER_INTERFACE_H_
#define TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_WOKRER_INTERFACE_H_

#include "tensorflow/contrib/seastar/seastar_tensor_coding.h"
#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/distributed_runtime/worker_cache.h"

namespace tensorflow {

class SeastarWorkerInterface {
 public:
  virtual void RecvTensorAsync(CallOptions* call_opts,
                               const RecvTensorRequest* request,
                               SeastarTensorResponse* response,
                               StatusCallback done) = 0;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_WOKRER_INTERFACE_H_
