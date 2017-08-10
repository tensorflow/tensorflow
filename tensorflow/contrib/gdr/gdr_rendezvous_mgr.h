#ifndef GDR_RENDEZVOUS_MGR_H_
#define GDR_RENDEZVOUS_MGR_H_

#include "tensorflow/contrib/gdr/gdr_memory_manager.h"
#include "tensorflow/core/distributed_runtime/base_rendezvous_mgr.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

class GdrRendezvousMgr : public BaseRendezvousMgr {
 public:
  explicit GdrRendezvousMgr(const WorkerEnv* env,
                            RemoteMemoryManager* remote_memory_manager);

 protected:
  BaseRemoteRendezvous* Create(int64 step_id, const WorkerEnv* worker_env);

 private:
  RemoteMemoryManager* remote_memory_manager_;  // Not owned

  TF_DISALLOW_COPY_AND_ASSIGN(GdrRendezvousMgr);
};

}  // end namespace tensorflow

#endif  // GDR_RENDEZVOUS_MGR_H_
