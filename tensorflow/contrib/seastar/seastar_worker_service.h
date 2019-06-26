#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_SEASTAR_SEASTAR_WORKER_SERVICE_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_SEASTAR_SEASTAR_WORKER_SERVICE_H_

#include <map>
#include "tensorflow/contrib/seastar/seastar_worker_service_method.h"
#include "tensorflow/contrib/seastar/seastar_worker_interface.h"
#include "tensorflow/core/distributed_runtime/worker.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
class SeastarServerTag;
class CallOptions;
class RecvTensorRequest;
class SeastarTensorResponse;
struct WorkerEnv;

class SeastarWorker : public Worker, public SeastarWorkerInterface {
 public:
  typedef std::function<void(const Status&)> StatusCallback;
  explicit SeastarWorker(WorkerEnv* worker_env);
  virtual ~SeastarWorker() {}

  // Specialized version of RecvTensor for seastar.
  void RecvTensorAsync(CallOptions* opts, const RecvTensorRequest* request,
                       SeastarTensorResponse *response, StatusCallback done);
  WorkerEnv* env();
};

class SeastarWorkerService {
public:
  using HandleRequestFunction = void (SeastarWorkerService::*)(SeastarServerTag*);

  explicit SeastarWorkerService(SeastarWorker* worker);
  virtual ~SeastarWorkerService() {}

  HandleRequestFunction GetHandler(SeastarWorkerServiceMethod methodId);

  void RunGraphHandler(SeastarServerTag* tag);
  void GetStatusHandler(SeastarServerTag* tag);
  void CreateWorkerSessionHandler(SeastarServerTag* tag);
  void DeleteWorkerSessionHandler(SeastarServerTag* tag);
  void CleanupAllHandler(SeastarServerTag* tag);
  void RegisterGraphHandler(SeastarServerTag* tag);
  void DeregisterGraphHandler(SeastarServerTag* tag);
  void CleanupGraphHandler(SeastarServerTag* tag);
  void LoggingHandler(SeastarServerTag* tag);
  void TracingHandler(SeastarServerTag* tag);
  void RecvTensorHandlerRaw(SeastarServerTag* tag);
  void RecvBufHandler(SeastarServerTag* tag);
  void CompleteGroupHandler(SeastarServerTag* tag);
  void CompleteInstanceHandler(SeastarServerTag* tag);
  void GetStepSequenceHandler(SeastarServerTag* tag);

private:
  void Schedule(std::function<void()> f);

  std::map<SeastarWorkerServiceMethod, HandleRequestFunction> handler_map_;
  SeastarWorker* worker_;
};

std::unique_ptr<SeastarWorker> NewSeastarWorker(WorkerEnv* worker_env);
std::unique_ptr<SeastarWorkerService> NewSeastarWorkerService(SeastarWorker* worker);

}  // namespace tensorflow

#endif // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_SEASTAR_SEASTAR_WORKER_SERVICE_H_
