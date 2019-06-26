#ifndef TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_RENDEZVOUS_MGR_H_
#define TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_RENDEZVOUS_MGR_H_

#include "tensorflow/core/distributed_runtime/base_rendezvous_mgr.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
class DeviceMgr;
class SeastarRendezvousMgr : public BaseRendezvousMgr {
 public:
  explicit SeastarRendezvousMgr(const WorkerEnv* env);

 protected:
  BaseRemoteRendezvous* Create(int64 step_id, const WorkerEnv* worker_env);

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(SeastarRendezvousMgr);
};
}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_RENDEZVOUS_MGR_H_
