#ifndef TENSORFLOW_CONTRIB_SEASTAR_WORKER_CACHE_H_
#define TENSORFLOW_CONTRIB_SEASTAR_WORKER_CACHE_H_

#include "tensorflow/contrib/seastar/seastar_channel_cache.h"
#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/distributed_runtime/worker_interface.h"

namespace tensorflow {

WorkerCacheInterface* NewSeastarWorkerCache(SeastarChannelCache* channel_cache,
                                            WorkerEnv* env);

WorkerCacheInterface* NewSeastarWorkerCacheWithLocalWorker(
    SeastarChannelCache* channel_cache, WorkerInterface* local_worker,
    const string& local_target, WorkerEnv* env);

}  // namespace tensorflow

#endif
