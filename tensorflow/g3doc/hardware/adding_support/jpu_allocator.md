# JPU Memory Allocator

There is a memory allocator framework

* [class Allocator](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/gpu/process_state.h)
  * [class TrackingAllocator : public Allocator](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tracking_allocator.h)
  * [class TestableSizeTrackingAllocator : public Allocator](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tracking_allocator_test.cc)
  * [class RecordingAllocator : public Allocator](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/gpu/process_state.h)
  * [class NoMemoryAllocator : public Allocator](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tracking_allocator_test.cc)
  * [class CPUAllocator : public Allocator](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/allocator.cc)
  * [class DummyCPUAllocator : public Allocator](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor_test.cc)
  * [class VisitableAllocator : public Allocator](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/visitable_allocator.h)
    * [class PoolAllocator : public VisitableAllocator](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/gpu/pool_allocator.h)
    * [class GPUDebugAllocator : public VisitableAllocator](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/gpu/gpu_debug_allocator.h)
    * [class GPUNanResetAllocator : public VisitableAllocator](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/gpu/gpu_debug_allocator.h)
    * [class BFCAllocator : public VisitableAllocator](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/bfc_allocator.h)
      * [class GPUBFCAllocator : public BFCAllocator](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.h)

* [class ScratchAllocator](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/stream_executor/scratch_allocator.h)
  * [class OneTimeScratchAllocator : public ScratchAllocator](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/stream_executor/scratch_allocator.h)
  
* [class SubAllocator](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/allocator.h)
  * [class BasicCPUAllocator : public SubAllocator](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/gpu/pool_allocator.h)
  * [class CUDAHostAllocator : public SubAllocator](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/gpu/pool_allocator.h)
  * [class GPUMemAllocator : public SubAllocator](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.h)

* [class AllocatorRetry](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/allocator_retry.h)

## Allocator vs. SubAllocator

Allocators use various algorithms and tracking mechanisms to control the 
flow of memory. They often use SubAllocators (eg. `BFCAllocator` and 
`PoolAllocator`) to do the actual memory allocation. Note that the 
available SubAllocators are `BasicCPUAllocator`, `CUDAHostAllocator`,
and `GPUMemAllocator` which are basically wrappers around `malloc`,
CUDA DMA, and CUDA Memory Managment respectively.

This stucture allows hardware to take advantage of smart allocation
schemes for their given allocation methods. For example, [class GPUBFCAllocator : public BFCAllocator](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.h)
is basically a wrapper around BFCAllocator which interfaces with the
GPUMemAllocator just through the constructor. Novel hardware should
implement a SubAllocator for their specific hardware and then implement
and Allocator that wraps around.

## Simple CPU Allocator

For the sake of simplicity, extending the CPU Allocator
will allow us to work in a GPU free environment (which 
I happen to be). This allocator does not make use of
a SubAllocator and is as slim as possible.

```C++
namespace tensorflow {

class JPUAllocator : public Allocator {
 public:
  JPUAllocator() {}
  ~JPUAllocator() override {}

  string Name() override { return "jpu"; }

  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    void* p = port::aligned_malloc(num_bytes, alignment);
    return p;
  }

  void DeallocateRaw(void* ptr) override {
    port::aligned_free(ptr);
  }

  size_t AllocatedSizeSlow(void* ptr) override {
    return port::MallocExtension_GetAllocatedSize(ptr);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(JPUAllocator);
};

} // namespace tensorflow 
```

For a very simple example of using a SubAllocator,
see [class GPUBFCAllocator : public BFCAllocator](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.h)
and perhaps try switching the above Allocator with
a BFCAllocator subclass that uses the [class BasicCPUAllocator : public SubAllocator](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/gpu/pool_allocator.h)
as the backend for allocation.

## Use in the GPU Case

Many classes use `cpu_allocator()` to create an Allocator for 
themselves. This function is found in [allocator.cc](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/allocator.cc#L121).
For example, many of the device factories use this.

* [ThreadPoolDeviceFactory](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/threadpool_device_factory.cc#L41)
* [GPUCompatibleCPUDeviceFactory](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/gpu/gpu_device_factory.cc#L102)

In the `GPUCompatibleCPUDevice` code, we can see how the
allocator are dealt with when a special allocators are needed
to offer optimized interfaces.

```C++
class GPUCompatibleCPUDevice : public ThreadPoolDevice {
 public:
  GPUCompatibleCPUDevice(const SessionOptions& options, const string& name,
                         Bytes memory_limit, BusAdjacency bus_adjacency,
                         Allocator* allocator)
      : ThreadPoolDevice(options, name, memory_limit, bus_adjacency,
                         allocator) {}
  ~GPUCompatibleCPUDevice() override {}

  Allocator* GetAllocator(AllocatorAttributes attr) override {
    ProcessState* ps = ProcessState::singleton();
    if (attr.gpu_compatible()) {
      return ps->GetCUDAHostAllocator(0);
    } else {
      // Call the parent's implementation.
      return ThreadPoolDevice::GetAllocator(attr);
    }
  }
};
```

Note that `ps->GetCUDAHostAllocator(0)` gives a special allocator
for GPU interfacing and in the case this isn't supported,
the standard `CPUAllocator` is returned from the `ThreadPoolDevice`
superclass.