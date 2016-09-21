# Adding a New Device

PREREQUISITES:

* Some familiarity with C++.
* Must have installed the
  [TensorFlow binary](../../get_started/os_setup.md#pip-installation), or must
  have
  [downloaded TensorFlow source](../../get_started/os_setup.md#installing-from-sources),
  and be able to build it.

[TOC]

## Important Classes

There are several important classes to become familiar with
as you will be implementing subclasses in order to get your device to work

For a local PCI style device, you will need to make subclasses like the following,

* [class FPUDeviceContext : public DeviceContext](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/device_base.h#L64)
* [class FPUDevice : public LocalDevice](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/local_device.h#L31)
* [class FPUAllocator : public Allocator](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/allocator.h#L67)
* [class FPUDeviceFactory : public DeviceFactory](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/device_factory.h#L30)


The standard includes you will need are,
```c++
#include <vector>
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.pb_text.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session_options.h"
```

## Basic Code

As a basic introduction, I show the minimal code needed to make
a device work. I call it the FPU (Fake Processing Unit). Note that
no kernels have been written for this device, so nothing will run on it,
but you can load it into the TensorFlow graph where is will just fail
when you actually try to run things on it.

To write kernels for this device, please follow [Adding a New Op](../adding_an_op/index.md).

### Device

```c++
  class FPUDevice : public LocalDevice {
  public:
    FPUDevice(const SessionOptions& options, const string& name, Bytes memory, 
	      Allocator *fpu_allocator, Allocator *cpu_allocator)
      : LocalDevice(options, Device::BuildDeviceAttributes(name, DEVICE_FPU, memory,
							   BUS_ANY, "physical description"),
		    fpu_allocator),
	fpu_allocator_(fpu_allocator),
	cpu_allocator_(cpu_allocator) {
      // Do I need to free these contexts later?
      device_contexts_.push_back(new FPUDeviceContext());
    }

    ~FPUDevice() override {
      delete fpu_allocator_;
    }
      
    Allocator* GetAllocator(AllocatorAttributes attr) override {
      if (attr.on_host())
	return this->cpu_allocator_;
      else
	return this->fpu_allocator_;
    }
    
    void Compute(OpKernel* op_kernel, OpKernelContext* context) override {
      op_kernel->Compute(context);
    }
    
    Status FillContextMap(const Graph *graph, DeviceContextMap *device_context_map) override {
      size_t N = device_contexts_.size();
      for (Node *n : graph->nodes()) {
	auto ctx = device_contexts_[n->id() % N];
	ctx->Ref();
	device_context_map->insert(std::make_pair(n->id(), ctx));
      }
      return Status::OK();
    }

    Status MakeTensorFromProto(const TensorProto& tensor_proto,
			       const AllocatorAttributes alloc_attrs,
			       Tensor* tensor) override {
      Tensor parsed(tensor_proto.dtype());
      if (!parsed.FromProto(GetAllocator(alloc_attrs), tensor_proto)) {
	return errors::InvalidArgument("Cannot parse tensor from proto: ",
				       ProtoDebugString(tensor_proto));
      }
      *tensor = parsed;
      return Status::OK();
    }
    
    Status Sync() override { return Status::OK(); }
    
  protected:
    Allocator *fpu_allocator_;
    Allocator *cpu_allocator_;  // Not owned
    std::vector<FPUDeviceContext*> device_contexts_;
  };
```

### DeviceFactory

```c++
  class FPUDeviceFactory : public DeviceFactory {
  public:
    void CreateDevices(const SessionOptions& options,
		       const string& name_prefix,
		       std::vector<Device*>* devices) override {
      int n = INT_MAX;
      auto iter = options.config.device_count().find("FPU");
      if (iter != options.config.device_count().end()) {
	n = iter->second;
      }
      std::vector<int> valid_fpu_ids;
      GetValidDeviceIds(&valid_fpu_ids);
      if (static_cast<size_t>(n) > valid_fpu_ids.size()) {
	n = valid_fpu_ids.size();
      }
      for (int i = 0; i < n; i++) {
	devices->push_back(new FPUDevice(options, strings::StrCat(name_prefix, "/device:FPU:", i), 
					 Bytes(256 << 20), new FPUAllocator(), 
					 cpu_allocator()));
      }
    }
    
  private:
    void GetValidDeviceIds(std::vector<int>* ids) {
      for (int i=0;i<4;i++)
	ids->push_back(i);
    }
  };
  REGISTER_LOCAL_DEVICE_FACTORY("FPU", FPUDeviceFactory);
```

### Allocator

```c++
  class FPUAllocator : public Allocator {
  public:
    FPUAllocator() {}
    ~FPUAllocator() override {}
    
    string Name() override { return "device:FPU"; }
    
    /* The void* could be a handle to a hardware allocation descriptor
     * See tensorflow/stream_executor/device_memory.h
     */
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
    TF_DISALLOW_COPY_AND_ASSIGN(FPUAllocator);
  };
```

### DeviceContext

```c++
  class FPUDeviceContext : public DeviceContext {
    void CopyCPUTensorToDevice(const Tensor* cpu_tensor, Device* device,
			       Tensor* device_tensor,
			       StatusCallback done) const override {  
      *device_tensor = *cpu_tensor;
      done(Status::OK());
    }

    void CopyDeviceTensorToCPU(const Tensor* device_tensor,
			       StringPiece tensor_name, Device* device,
			       Tensor* cpu_tensor, StatusCallback done) override {
      *cpu_tensor = *device_tensor;
      done(Status::OK());
    }
  };
```

## Compilation

TODO(vrv)