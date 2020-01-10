#ifndef TENSORFLOW_COMMON_RUNTIME_GPU_PROCESS_STATE_H_
#define TENSORFLOW_COMMON_RUNTIME_GPU_PROCESS_STATE_H_

#include <functional>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {

class Allocator;
class VisitableAllocator;
class PoolAllocator;

// Singleton that manages per-process state, e.g. allocation
// of shared resources.
class ProcessState {
 public:
  static ProcessState* singleton();

  // Descriptor for memory allocation attributes, used by optional
  // runtime correctness analysis logic.
  struct MemDesc {
    enum MemLoc { CPU, GPU };
    MemLoc loc;
    int dev_index;
    bool gpu_registered;
    bool nic_registered;
    MemDesc()
        : loc(CPU),
          dev_index(0),
          gpu_registered(false),
          nic_registered(false) {}
    string DebugString();
  };

  // Records the number of GPUs available in the local process.
  // It is a fatal error to call this with a value != to the value
  // in a prior call.
  void SetGPUCount(int c);

  // Returns number of GPUs available in local process, as set by
  // SetGPUCount();  Returns 0 if SetGPUCount has not been called.
  int GPUCount() const;

  // Returns what we know about the memory at ptr.
  // If we know nothing, it's called CPU 0 with no other attributes.
  MemDesc PtrType(const void* ptr);

  // Returns the one CPUAllocator used for the given numa_node.
  // TEMPORY: ignores numa_node.
  Allocator* GetCPUAllocator(int numa_node);

  // Returns the one GPU allocator used for the indexed GPU.
  // Note that this is a system GPU index, not (necessarily) a brain
  // device index.
  //
  // 'total_bytes' is the total number of bytes that should be made
  // available to the allocator.  The first call to this function for
  // a given gpu_id creates the allocator, so only the total_bytes
  // used on that first call is used.
  //
  // REQUIRES: gpu_id must be a valid ordinal for a GPU available in the
  // current system environment.  Otherwise returns nullptr.
  Allocator* GetGPUAllocator(int gpu_id, size_t total_bytes);

  Allocator* GetCUDAHostAllocator(int numa_node);

  // Registers a function to be called once on every new Region
  // allocated by every GPURegionAllocator proximate to the specified
  // bus.  The AllocVisitor is provided with a memory pointer and the
  // size of the area it identifies.  The pointer is not guaranteed to
  // be valid after the call terminates.  The intention is for this
  // interface to be used for network device memory registration.
  // "bus_id" is platform-specific.  On many platforms it
  // should be 0.  On machines with multiple PCIe buses, it should be
  // the index of one of the PCIe buses.  If the the bus_id is invalid,
  // results are undefined.
  typedef std::function<void(void*, size_t)> AllocVisitor;
  void AddGPUAllocVisitor(int bus_id, AllocVisitor visitor);

  typedef std::unordered_map<const void*, MemDesc> MDMap;

 protected:
  ProcessState();

  static ProcessState* instance_;

  mutex mu_;
  int gpu_count_;

  std::vector<PoolAllocator*> cpu_allocators_ GUARDED_BY(mu_);
  std::vector<VisitableAllocator*> gpu_allocators_ GUARDED_BY(mu_);
  std::vector<std::vector<AllocVisitor>> gpu_visitors_ GUARDED_BY(mu_);
  std::vector<PoolAllocator*> cuda_host_allocators_ GUARDED_BY(mu_);

  virtual ~ProcessState();

  // Optional RecordingAllocators that wrap the corresponding
  // Allocators for runtime attribute use analysis.
  MDMap mem_desc_map_;
  std::vector<Allocator*> cpu_al_ GUARDED_BY(mu_);
  std::vector<Allocator*> gpu_al_ GUARDED_BY(mu_);
  std::vector<Allocator*> cuda_al_ GUARDED_BY(mu_);
};

namespace internal {
class RecordingAllocator : public Allocator {
 public:
  RecordingAllocator(ProcessState::MDMap* mm, Allocator* a,
                     ProcessState::MemDesc md, mutex* mu)
      : mm_(mm), a_(a), md_(md), mu_(mu) {}

  string Name() override { return a_->Name(); }
  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    void* p = a_->AllocateRaw(alignment, num_bytes);
    mutex_lock l(*mu_);
    (*mm_)[p] = md_;
    return p;
  }
  void DeallocateRaw(void* p) override {
    mutex_lock l(*mu_);
    auto iter = mm_->find(p);
    mm_->erase(iter);
    a_->DeallocateRaw(p);
  }
  bool TracksAllocationSizes() override { return a_->TracksAllocationSizes(); }
  size_t RequestedSize(void* p) override { return a_->RequestedSize(p); }
  size_t AllocatedSize(void* p) override { return a_->AllocatedSize(p); }
  ProcessState::MDMap* mm_;  // not owned
  Allocator* a_;             // not owned
  ProcessState::MemDesc md_;
  mutex* mu_;
};
}  // namespace internal
}  // namespace tensorflow
#endif  // TENSORFLOW_COMMON_RUNTIME_GPU_PROCESS_STATE_H_
