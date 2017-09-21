/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

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
