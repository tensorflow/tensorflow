/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/numa_device.h"

#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/allocator_registry.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.pb_text.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/types.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session_options.h"

#include <sys/syscall.h>
#include <thread>

#ifdef INTEL_MKL
#include "tensorflow/core/common_runtime/mkl_cpu_allocator.h"
#endif

namespace tensorflow {

static void *run_task(void *args) {
  NumaDevice *device = (NumaDevice *)args;

  // Set thread affinity
  std::vector<int> &proc_set = device->run_state_.proc_set_;
  omp_set_num_threads(proc_set.size());

  int    thread_bound[proc_set.size()];
  pid_t  thread_id   [proc_set.size()];

  #pragma omp parallel num_threads(proc_set.size())
  {
    int omp_thread_id = omp_get_thread_num();
    thread_id[omp_thread_id] = syscall(SYS_gettid);

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(proc_set[omp_thread_id], &cpuset);
    pthread_t thread = pthread_self();
    int s = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    if (s != 0) {
      std::cout << "Failed to set thread affinity\n";
    }

    // Check the actual affinity mask assigned to the thread 
    s = pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    if (s != 0){
      std::cout << "failed to call pthread_getaffinity_np\n";
    }
    else {
      for (int j = 0; j < CPU_SETSIZE; j++)
        if (CPU_ISSET(j, &cpuset))
          thread_bound[omp_thread_id] = j;
    }
  }

  std::cout << "device = " << device->name() << " thread number = " << (int)proc_set.size() << ", bound to core:\n";
  for(int i=0; i<(int)proc_set.size(); i++)
    std::cout << "(" <<thread_bound[i]<<"," << " " << (unsigned long)thread_id[i] << ")\n";
  std::cout << "\nFinish initialization\n";

  while(device->run_state_.status_ != NumaDevice::Cancel) {
    //std::cout << "device = " << (long) device << " status = " << device->run_state_.status_ << "\n";
    if(device->run_state_.status_ == NumaDevice::RunTask) {
      //std::cout << "NUMA device run task\n";
      OpKernel* op_kernel      = device->run_state_.op_kernel_;
      OpKernelContext* context = device->run_state_.context_;
      // When Xprof/ThreadScape profiling is off (which is the default), the
      // following code is simple enough that its overhead is negligible.
      tracing::ScopedActivity activity(op_kernel->name(), op_kernel->type_string(),
                                   op_kernel->IsExpensive());
      tracing::ScopedRegion region(tracing::EventCategory::kCompute,
                                   op_kernel->name());
      op_kernel->Compute(context);

      device->run_state_.status_ = NumaDevice::NoTask;
      //std::cout << "NUMA device run task done\n";
      device->run_state_.notification_->Notify();
    }
    //sleep(1);
    //nanosleep((const struct timespec[]){{0, 1L}}, NULL);
  }

  return NULL;
}

NumaDevice::NumaDevice(const SessionOptions& options,
                                   const string& name, Bytes memory_limit,
                                   const DeviceLocality& locality,
                                   Allocator* allocator, std::vector<int> &proc_set)
    : LocalDevice(options, proc_set, Device::BuildDeviceAttributes(
                      name, DEVICE_CPU, memory_limit, locality)),
      allocator_(allocator) {
  std::cout << "***************In NumaDevice constructor\n";

  memory = 0;

  run_state_.status_   = NumaDevice::NoTask;
  run_state_.proc_set_ = proc_set;
  pthread_create(&task_thread_, NULL, &run_task, this);
}

NumaDevice::~NumaDevice()
{
  std::cout << "allocated memory from MakeTensorFromProto " << memory << "\n";
  run_state_.status_ = Cancel;
}

void NumaDevice::Compute(OpKernel* op_kernel, OpKernelContext* context) {
  mutex_lock lock(mutex_);

  //std::cout << "NumaDevice:: " << (long) this << " compute start\n";
  Notification n;

  run_state_.op_kernel_    = op_kernel;
  run_state_.context_      = context;
  run_state_.notification_ = &n;
  run_state_.status_       = NumaDevice::RunTask;
  n.WaitForNotification();

  //std::cout << "NumaDevice:: " << (long) this << " compute done\n"; 
}

Allocator* NumaDevice::GetAllocator(AllocatorAttributes attr) {
  return allocator_;
}

Status NumaDevice::MakeTensorFromProto(
    const TensorProto& tensor_proto, const AllocatorAttributes alloc_attrs,
    Tensor* tensor) {
  if (tensor_proto.dtype() > 0 && tensor_proto.dtype() <= DataType_MAX) {
    Tensor parsed(tensor_proto.dtype());
    if (parsed.FromProto(cpu_allocator(), tensor_proto)) {
      *tensor = std::move(parsed);
      memory += tensor->AllocatedBytes();
      return Status::OK();
    }
  }
  return errors::InvalidArgument("Cannot parse tensor from proto: ",
                                 ProtoDebugString(tensor_proto));
}

#ifdef INTEL_MKL
REGISTER_MEM_ALLOCATOR("MklCPUAllocator", 200, MklCPUAllocator);
#endif

}  // namespace tensorflow
