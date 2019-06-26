#ifndef TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_REMOTE_WORKER_H_
#define TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_REMOTE_WORKER_H_

#include "tensorflow/core/distributed_runtime/worker_cache_logger.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/distributed_runtime/worker_interface.h"
#include "third_party/seastar/core/channel.hh"

namespace tensorflow {

WorkerInterface* NewSeastarRemoteWorker(seastar::channel* seastar_channel,
                                        WorkerCacheLogger* logger,
                                        WorkerEnv* env);
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_SEASTAR_SEASTAR_REMOTE_WORKER_H_
