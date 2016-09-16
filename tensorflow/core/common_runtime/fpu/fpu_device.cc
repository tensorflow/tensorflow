/* Copyright 2016 Knupath.
   TensorFlow might have some rights
   I am not a lawyer. Please sue someone else!
*/



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

namespace tensorflow {

  const char* const DEVICE_FPU = "FPU";

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

  class FPUDevice : public LocalDevice {
  public:
    FPUDevice(const SessionOptions& options, const string& name,
	      Bytes memory_needed, int fpu_id, int clusters,
	      Allocator *fpu_allocator, Allocator *cpu_allocator)
      : LocalDevice(options, Device::BuildDeviceAttributes(name, DEVICE_FPU, memory_needed,
							   BUS_ANY, "physical description"),
		    fpu_allocator),
	fpu_allocator_(fpu_allocator),
	cpu_allocator_(cpu_allocator) {
      // Do I need to free these contexts later?
      device_contexts_.push_back(new FPUDeviceContext());
    }
      
    Allocator* GetAllocator(AllocatorAttributes attr) override {
      if (attr.on_host())
	return this->cpu_allocator_;
      else
	return this->fpu_allocator_;
    }

    ~FPUDevice() override { }
    
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
    Allocator *fpu_allocator_, *cpu_allocator_;  // Not owned
    std::vector<FPUDeviceContext*> device_contexts_;
  };


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
	devices->push_back(new FPUDevice(options, strings::StrCat(name_prefix, "/fpu:", i), 
					 Bytes(256 << 20), valid_fpu_ids[i], 1, 
					 cpu_allocator() /* fpu_allocator */, cpu_allocator()));
      }
    }
    
  private:
    void GetValidDeviceIds(std::vector<int>* ids) {
      for (int i=0;i<4;i++)
	ids->push_back(i);
    }
  };
  REGISTER_LOCAL_DEVICE_FACTORY("FPU", FPUDeviceFactory);
  
} // tensorflow
