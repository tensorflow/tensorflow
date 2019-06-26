#ifndef TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_WORKER_SERVICE_METHOD_H_
#define TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_WORKER_SERVICE_METHOD_H_

namespace tensorflow {

enum class SeastarWorkerServiceMethod {
  kGetStatus = 0,
  kCreateWorkerSession,
  kDeleteWorkerSession,
  kRegisterGraph,
  kDeregisterGraph,
  kRunGraph,
  kCleanupGraph,
  kCleanupAll,
  kRecvTensor,
  kFuseRecvTensor,
  kLogging,
  kTracing,
  kRecvBuf,
  kCompleteGroup,
  kCompleteInstance,
  kGetStepSequence,
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_WORKER_SERVICE_METHOD_H_
