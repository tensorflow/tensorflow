// TODO(opensource): Use a more generic sounding preprocessor name than
// GOOGLE_CUDA
#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/common_runtime/gpu/gpu_device.h"

#include <stdlib.h>
#include <string.h>

//#include "base/commandlineflags.h"
#include "tensorflow/stream_executor/cuda/cuda_activation.h"
#include "tensorflow/stream_executor/multi_platform_manager.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/stream_executor.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/common_runtime/gpu/gpu_stream_util.h"
#include "tensorflow/core/common_runtime/gpu/gpu_util.h"
#include "tensorflow/core/common_runtime/gpu/process_state.h"
#include "tensorflow/core/common_runtime/gpu_device_context.h"
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/types.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/status.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/util/device_name_utils.h"

#if defined(PLATFORM_GOOGLE)
DEFINE_bool(brain_gpu_sync_every_op, false,
            "If true, call GPUUtil::Sync() between every dispatched opkernel.");

DEFINE_int32(brain_gpu_max_streams, 1,
             "Max number of GPU streams to use for computation.");
#else
// TODO(opensource): These should be made options in some options struct,
// rather than flags.
bool FLAGS_brain_gpu_sync_every_op = false;
tensorflow::int32 FLAGS_brain_gpu_max_streams = 1;
#endif

namespace gpu = ::perftools::gputools;

namespace tensorflow {

// Eigen Ops directly allocate memory only for temporary buffers used
// during OpKernel::Compute().  The recommended way of allocating such
// memory is via OpKernelContext::allocate_temp().  However, Eigen Ops
// don't have access to OpKernelContext, instead they get access to
// memory directly through the device allocator.  As an Open Source
// project, Eigen assumes allocator semantics similar to those of the
// CUDA memory allocator, and may not work correctly due to race
// conditions if used with some other allocator.  For safety, we need
// to delay deallocation calls out of Eigen until all events on the
// corresponding stream have completed.  The following two classes
// serve this purpose in two different compilation environments.

#if defined(__GCUDACC__) || defined(__GCUDACC_HOST__)
class EigenAllocator : public ::Eigen::Allocator {
 public:
  explicit EigenAllocator(gpu::Stream* stream, ::tensorflow::Allocator* alloc,
                          EventMgr* em)
      : stream_(stream), allocator_(alloc), em_(em) {}

  void* allocate(size_t num_bytes) const override {
    void* ret = allocator_->AllocateRaw(32 /* alignment */, num_bytes);
    // Eigen doesn't typically check the return pointer from allocate,
    // so we do it here and die with a more helpful error message.
    if (ret == nullptr) {
      LOG(FATAL) << "EigenAllocator for GPU ran out of memory when allocating "
                 << num_bytes << ". See error logs for more detailed info.";
    }
    return ret;
  }

  void deallocate(void* buffer) const override {
    em_->ThenDeleteBuffer(stream_, {allocator_, buffer});
  }

 private:
  gpu::Stream* stream_;            // Not owned.
  ::tensorflow::Allocator* allocator_;  // Not owned.
  ::tensorflow::EventMgr* em_;          // Not owned.

  TF_DISALLOW_COPY_AND_ASSIGN(EigenAllocator);
};

#else
class EigenCudaStreamDevice : public ::Eigen::StreamInterface {
 public:
  EigenCudaStreamDevice(const cudaStream_t* cuda_stream, int gpu_id,
                        ::tensorflow::Allocator* alloc)
      : stream_(cuda_stream), allocator_(alloc) {
    Eigen::initializeDeviceProp();
    device_prop_ = &Eigen::m_deviceProperties[gpu_id];
  }

  const cudaStream_t& stream() const override { return *stream_; }
  const cudaDeviceProp& deviceProperties() const override {
    return *device_prop_;
  }

  void* allocate(size_t num_bytes) const override {
    void* ret = allocator_->AllocateRaw(32 /* alignment */, num_bytes);
    if (ret == nullptr) {
      LOG(FATAL) << "EigenAllocator for GPU ran out of memory when allocating "
                 << num_bytes << ". See error logs for more detailed info.";
    }

    return ret;
  }
  void deallocate(void* buffer) const override {
    AsyncFreeData* afData = new AsyncFreeData(allocator_, buffer);
    cudaError_t err = cudaStreamAddCallback(*stream_, asyncFree, afData, 0);
    CHECK_EQ(err, cudaSuccess);
  }

 private:
  struct AsyncFreeData {
    AsyncFreeData(::tensorflow::Allocator* a, void* p)
        : allocator_(a), address_(p) {}
    ::tensorflow::Allocator* allocator_;
    void* address_;
  };

  static void CUDART_CB asyncFree(cudaStream_t stream, cudaError_t status,
                                  void* userData) {
    AsyncFreeData* data = static_cast<AsyncFreeData*>(userData);
    data->allocator_->DeallocateRaw(data->address_);
    delete data;
  }

  const cudaStream_t* stream_;         // Not owned.
  const cudaDeviceProp* device_prop_;  // Not owned.
  ::tensorflow::Allocator* allocator_;  // Not owned.

  TF_DISALLOW_COPY_AND_ASSIGN(EigenCudaStreamDevice);
};

#endif

BaseGPUDevice::BaseGPUDevice(const SessionOptions& options, const string& name,
                             Bytes memory_limit, BusAdjacency bus_adjacency,
                             int gpu_id, const string& physical_device_desc,
                             Allocator* gpu_allocator, Allocator* cpu_allocator)
    : LocalDevice(options, Device::BuildDeviceAttributes(
                               name, DEVICE_GPU, memory_limit, bus_adjacency,
                               physical_device_desc),
                  gpu_allocator),
      gpu_allocator_(gpu_allocator),
      cpu_allocator_(cpu_allocator),
      gpu_id_(gpu_id) {
  gpu::StreamExecutor* executor =
      GPUMachineManager()->ExecutorForDevice(gpu_id_).ValueOrDie();
  if (!executor) {
    LOG(ERROR) << "Failed to get StreamExecutor for device " << gpu_id_;
    return;
  }
  em_.reset(new EventMgr(executor));

  if (FLAGS_brain_gpu_max_streams < 1) {
    LOG(FATAL) << "Invalid value for brain_gpu_max_streams.";
  }

  // Create the specified number of GPU streams
  for (int i = 0; i < FLAGS_brain_gpu_max_streams; i++) {
    auto stream = new gpu::Stream(executor);
    stream->Init();
    VLOG(2) << "Created stream[" << i << "] = " << stream;
    streams_.push_back(stream);
    device_contexts_.push_back(new GPUDeviceContext(i, stream));
  }
  gpu_device_info_ = new GpuDeviceInfo;
  gpu_device_info_->stream = streams_[0];
  gpu_device_info_->default_context = device_contexts_[0];
  gpu_device_info_->event_mgr = em_.get();
  set_tensorflow_gpu_device_info(gpu_device_info_);
}

BaseGPUDevice::~BaseGPUDevice() {
  delete gpu_device_info_;
  for (auto ctx : device_contexts_) ctx->Unref();
  gtl::STLDeleteElements(&streams_);
}

Status BaseGPUDevice::FillContextMap(const Graph* graph,
                                     DeviceContextMap* device_context_map) {
  VLOG(2) << "FillContextMap";

  const auto num_streams = streams_.size();
  // Special case for single stream.
  if (num_streams == 1) {
    return Status::OK();
  }
  const int64 before = Env::Default()->NowMicros();
  gpu_stream_util::AssignStreamsOpts opts;
  opts.max_streams = num_streams;
  std::unordered_map<int, int> node_to_stream_id;
  TF_RETURN_IF_ERROR(
      gpu_stream_util::AssignStreams(graph, opts, &node_to_stream_id));
  int64 elapsed = Env::Default()->NowMicros() - before;
  VLOG(3) << "AssignStreams took " << elapsed << "us";

  // Fill in the context map.  It is OK for this map to contain
  // duplicate DeviceContexts so long as we increment the refcount.
  for (Node* n : graph->nodes()) {
    auto mapped_stream = node_to_stream_id[n->id()];
    CHECK_LE(mapped_stream, num_streams);
    auto ctx = device_contexts_[mapped_stream];
    VLOG(3) << "Assigned stream " << node_to_stream_id[n->id()]
            << " ==> stream[" << ctx->stream_id() << "] for node id " << n->id()
            << " " << n->type_string() << " " << n->name();
    ctx->Ref();
    device_context_map->insert(std::make_pair(n->id(), ctx));
  }

  return Status::OK();
}

void BaseGPUDevice::Compute(OpKernel* op_kernel, OpKernelContext* context) {
  // ScopedActivity is cheap when tracing is not active, but we
  // can avoid computing the Hash64.
  // TODO(pbar) This would no longer be needed if Ops have a unique id.
  const uint64 id = port::Tracing::IsActive() ? Hash64(op_kernel->name()) : 0;
  port::Tracing::ScopedActivity region(port::Tracing::EventCategory::kCompute,
                                       id);

  GPUDeviceContext* gpu_device_context = device_contexts_[0];
  if (context->op_device_context() != nullptr) {
    gpu_device_context =
        static_cast<GPUDeviceContext*>(context->op_device_context());
  }
  gpu::Stream* stream = gpu_device_context->stream();
  const auto stream_id = gpu_device_context->stream_id();

  VLOG(1) << "GpuDevice::Compute " << op_kernel->name() << " op "
          << op_kernel->def().op() << " on GPU" << gpu_id_ << " stream["
          << stream_id << "]";

  // NOTE(tucker): We need to discriminate between Eigen GPU
  // operations and all others.  If an operation is Eigen
  // implemented (or otherwise tries to launch a cuda kernel
  // directly), we need to establish a stacked-scoped environment
  // that directs it to execute on the proper device.  Otherwise we
  // expect the Op to use StreamExecutor directly and correctly.  The
  // way we make this discrimination is quite hacky: At the moment
  // the only non-Eigen GPU Op is the recv-op, which is known to be
  // asynchronous.
  if (op_kernel->type_string() == "_Recv") {
    context->SetStatus(errors::Internal(
        "Invalid synchronous 'Compute' on GPU for '_Recv' op"));
  } else {
    const string label =
        strings::StrCat(op_kernel->name(), ":", op_kernel->type_string());
    port::Tracing::ScopedAnnotation annotation(label);

    const auto num_streams = streams_.size();
    if (num_streams > 1) {
      // If this op's device context is different from the other contexts,
      // we must wait on the stream.
      for (int i = 0; i < context->num_inputs(); ++i) {
        const GPUDeviceContext* idc =
            static_cast<GPUDeviceContext*>(context->input_device_context(i));
        OP_REQUIRES(context, idc != nullptr,
                    errors::Internal("Input device context ", i,
                                     " was not set properly."));
        if (VLOG_IS_ON(2)) {
          const void* base;
          size_t len;
          if (context->has_input(i)) {
            if (IsRefType(context->input_dtype(i))) {
              Tensor tensor = context->mutable_input(i, false);
              base = DMAHelper::base(&tensor);
              len = tensor.TotalBytes();
            } else {
              const Tensor& tensor = context->input(i);
              base = DMAHelper::base(&tensor);
              len = tensor.TotalBytes();
            }
            VLOG(2) << "Input " << i << " " << base << "  " << len;
            VLOG(2) << "  stream[" << stream_id << "].ThenWaitFor(stream["
                    << idc->stream_id() << "])"
                    << ((idc->stream() == stream) ? " not needed" : "");
          }
        }
        if (idc->stream() != stream) stream->ThenWaitFor(idc->stream());
      }
    }
    gpu::cuda::ScopedActivateExecutorContext scoped_activation{
        stream->parent(), gpu::cuda::MultiOpActivation::kYes};
    // Keep a copy of the inputs before Compute runs, in case they get
    // deleted. TODO(misard) this will be fixed when the tracking is
    // done right.
    std::vector<Tensor>* tensor_refs = nullptr;
    if (!FLAGS_brain_gpu_sync_every_op) {
      tensor_refs = new std::vector<Tensor>;
      tensor_refs->reserve(context->num_inputs() + context->num_outputs());
      for (int ii = 0; ii < context->num_inputs(); ++ii) {
        if (context->has_input(ii)) {
          if (IsRefType(context->input_dtype(ii))) {
            Tensor in = context->mutable_input(ii, false);
            tensor_refs->push_back(in);
          } else {
            const Tensor& in = context->input(ii);
            tensor_refs->push_back(in);
          }
        }
      }
    }
    op_kernel->Compute(context);
    if (context->status().ok()) {
      if (FLAGS_brain_gpu_sync_every_op) {
        // Note: GPUUtil::Sync() only syncs the default stream.
        // We need to either sync the stream used by this op, or
        // all streams.  Given that this flag is typically used for
        // debugging it makes more sense to sync all GPU activity.
        context->SetStatus(GPUUtil::SyncAll(this));
      } else {
        // The GPU kernel has been queued, but may not complete for some
        // time.  As soon as this function completes, the caller will
        // discard its refs on the inputs, outputs and any scratch
        // tensors it created. Create additional refs here that will be
        // held until the kernel completes.
        for (int ii = 0; ii < context->num_temps(); ++ii) {
          Tensor* temp = context->temp(ii);
          VLOG(2) << "Saving ref to temp Tensor @ " << DMAHelper::base(temp);
          tensor_refs->push_back(*temp);
        }
        for (int ii = 0; ii < context->num_outputs(); ++ii) {
          Tensor* temp = context->mutable_output(ii);
          if (nullptr != temp) {
            tensor_refs->push_back(*temp);
          }
        }
        em_->ThenDeleteTensors(stream, tensor_refs);
      }
    } else {
      if (!FLAGS_brain_gpu_sync_every_op) {
        delete tensor_refs;
      }
    }
  }
}

Status BaseGPUDevice::Sync() { return GPUUtil::Sync(this); }

void BaseGPUDevice::ComputeAsync(AsyncOpKernel* op_kernel,
                                 OpKernelContext* context,
                                 AsyncOpKernel::DoneCallback done) {
  GPUDeviceContext* gpu_device_context = device_contexts_[0];
  if (context->op_device_context() != nullptr) {
    gpu_device_context =
        static_cast<GPUDeviceContext*>(context->op_device_context());
  }
  const auto stream_id = gpu_device_context->stream_id();

  VLOG(1) << "GpuDevice::ComputeAsync " << op_kernel->name() << " op "
          << op_kernel->def().op() << " on GPU" << gpu_id_ << " stream["
          << stream_id << "]";

  port::Tracing::TraceMe activity(
      strings::StrCat(op_kernel->name(), ":", op_kernel->type_string()));
  op_kernel->ComputeAsync(context, done);
}

Status BaseGPUDevice::MakeTensorFromProto(const TensorProto& tensor_proto,
                                          const AllocatorAttributes alloc_attrs,
                                          Tensor* tensor) {
  AllocatorAttributes attr;
  attr.set_on_host(true);
  attr.set_gpu_compatible(true);
  Allocator* host_alloc = GetAllocator(attr);
  Tensor parsed(tensor_proto.dtype());
  if (!parsed.FromProto(host_alloc, tensor_proto)) {
    return errors::InvalidArgument("Cannot parse tensor from proto: ",
                                   tensor_proto.DebugString());
  }
  Status status;
  if (alloc_attrs.on_host()) {
    *tensor = parsed;
  } else {
    if (!DMAHelper::CanUseDMA(&parsed)) {
      return errors::Internal("GPU copy from non-DMA ",
                              DataTypeString(parsed.dtype()), " tensor");
    }
    Tensor copy(GetAllocator(alloc_attrs), parsed.dtype(), parsed.shape());
    port::Tracing::ScopedAnnotation annotation("MakeTensorFromProto");
    Notification n;
    device_contexts_[0]->CopyCPUTensorToDevice(&parsed, this, &copy,
                                               [&n, &status](const Status& s) {
                                                 status = s;
                                                 n.Notify();
                                               });
    n.WaitForNotification();
    *tensor = copy;
  }
  return status;
}

namespace {
#if defined(__GCUDACC__) || defined(__GCUDACC_HOST__)
class ConcretePerOpGpuDevice : public PerOpGpuDevice {
 public:
  explicit ConcretePerOpGpuDevice(gpu::Stream* stream,
                                  EigenAllocator* allocator)
      : device_(stream, allocator), allocator_(allocator) {}
  ~ConcretePerOpGpuDevice() { delete allocator_; }

  const Eigen::GpuDevice& device() const override { return device_; }

 private:
  Eigen::GpuDevice device_;
  EigenAllocator* allocator_;
};
#else
class ConcretePerOpGpuDevice : public PerOpGpuDevice {
 public:
  explicit ConcretePerOpGpuDevice(EigenCudaStreamDevice* stream_device)
      : device_(stream_device), stream_device_(stream_device) {}
  ~ConcretePerOpGpuDevice() { delete stream_device_; }

  const Eigen::GpuDevice& device() const override { return device_; }

 private:
  Eigen::GpuDevice device_;
  EigenCudaStreamDevice* stream_device_;
};
#endif
}  // namespace

const PerOpGpuDevice* BaseGPUDevice::NewDevice(int stream_id,
                                               Allocator* allocator) {
#if defined(__GCUDACC__) || defined(__GCUDACC_HOST__)
  auto ea = new EigenAllocator(streams_[stream_id], allocator, em_.get());
  return new ConcretePerOpGpuDevice(streams_[stream_id], ea);
#else
  const cudaStream_t* cuda_stream = reinterpret_cast<const cudaStream_t*>(
      streams_[stream_id]->implementation()->CudaStreamMemberHack());
  auto es = new EigenCudaStreamDevice(cuda_stream, gpu_id_, allocator);
  return new ConcretePerOpGpuDevice(es);
#endif
}

const PerOpGpuDevice* BaseGPUDevice::MakeGpuDevice(DeviceContext* dc,
                                                   Allocator* allocator) {
  if (dc) {
    const GPUDeviceContext* gpu_dc = static_cast<GPUDeviceContext*>(dc);
    const int stream_id = gpu_dc->stream_id();
    VLOG(1) << "  eigen_gpu_device(" << dc << ") => stream[" << stream_id
            << "]";
    CHECK_LT(stream_id, streams_.size());
    return NewDevice(stream_id, allocator);
  } else {
    return NewDevice(0, allocator);
  }
}

void BaseGPUDeviceFactory::CreateDevices(const SessionOptions& options,
                                         const string& name_prefix,
                                         std::vector<Device*>* devices) {
  int n = INT_MAX;
  auto iter = options.config.device_count().find("GPU");
  if (iter != options.config.device_count().end()) {
    n = iter->second;
  }
  std::vector<int> valid_gpu_ids;
  GetValidDeviceIds(&valid_gpu_ids);
  if (static_cast<size_t>(n) > valid_gpu_ids.size()) {
    n = valid_gpu_ids.size();
  }
  for (int i = 0; i < n; i++) {
    devices->push_back(CreateGPUDevice(
        options, strings::StrCat(name_prefix, "/gpu:", i), valid_gpu_ids[i]));
  }
}

namespace {
int64 MinSystemMemory(int64 available_memory) {
  // We use the following heuristic for now:
  //
  // If the available_memory is < 2GiB, we allocate 200MiB to system memory.
  // Otherwise, allocate 300MiB to system memory.
  //
  // In the future we could be more sophisticated by using a table of
  // devices.
  if (available_memory < (1LL << 31)) {
    // 200MiB
    return 209715200LL;
  } else {
    // max(300 MiB, 0.95 * available_memory)
    return std::max(314572800LL, static_cast<int64>(available_memory * 0.05));
  }
}
}  // namespace

static string GetShortDeviceDescription(int device_id,
                                        const gpu::DeviceDescription& desc) {
  return strings::StrCat("device: ", device_id, ", name: ", desc.name(),
                         ", pci bus id: ", desc.pci_bus_id());
}

LocalDevice* BaseGPUDeviceFactory::CreateGPUDevice(
    const SessionOptions& options, const string& name, int gpu_id) {
  CHECK_GE(gpu_id, 0);

  // Look up the device, to see its attributes.
  gpu::Platform* gpu_platform = GPUMachineManager();
  CHECK_LT(gpu_id, gpu_platform->VisibleDeviceCount());
  gpu::StreamExecutor* se =
      gpu_platform->ExecutorForDevice(gpu_id).ValueOrDie();
  const gpu::DeviceDescription& desc = se->GetDeviceDescription();

  int64 total_memory, available_memory;
  CHECK(se->DeviceMemoryUsage(&available_memory, &total_memory));

  int64 allocated_memory = available_memory;
  double config_memory_fraction =
      options.config.gpu_options().per_process_gpu_memory_fraction();
  if (config_memory_fraction == 0) {
    const int64 min_system_memory = MinSystemMemory(available_memory);
    if (min_system_memory < allocated_memory) {
      allocated_memory -= min_system_memory;
    }
  } else {
    allocated_memory *= config_memory_fraction;
  }

  Bytes allocated_bytes = static_cast<Bytes>(allocated_memory);

  // Get GPU BusAdjacency from its reported NUMA affinity.
  // Because GPUs are virtualized in some environments, we can't just
  // use the GPU id.
  BusAdjacency bus_adjacency = BUS_ANY;
  switch (desc.numa_node()) {
    case 0:
      bus_adjacency = BUS_0;
      break;
    case 1:
      bus_adjacency = BUS_1;
      break;
    default:
      bus_adjacency = BUS_ANY;
  }
  VLOG(1) << "GPUDevice id " << gpu_id << " on bus " << bus_adjacency
          << " numa: " << desc.numa_node() << " pci: " << desc.pci_bus_id();

  ProcessState* process_state = ProcessState::singleton();
  return CreateGPUDevice(
      options, name, allocated_bytes, bus_adjacency, gpu_id,
      GetShortDeviceDescription(gpu_id, desc),
      process_state->GetGPUAllocator(gpu_id, allocated_memory),
      process_state->GetCPUAllocator(desc.numa_node()));
}

static int GetMinGPUMultiprocessorCount() {
  static const int kDefaultMinGPUMultiprocessorCount = 8;

  const char* tf_min_gpu_core_count = getenv("TF_MIN_GPU_MULTIPROCESSOR_COUNT");

  if (tf_min_gpu_core_count == nullptr ||
      strcmp(tf_min_gpu_core_count, "") == 0) {
    return kDefaultMinGPUMultiprocessorCount;
  }

  int min_gpu_core_count = -1;
  if (strings::safe_strto32(tf_min_gpu_core_count, &min_gpu_core_count)) {
    if (min_gpu_core_count >= 0) {
      return min_gpu_core_count;
    }
  }

  LOG(ERROR) << "Invalid minimum GPU multiprocessor count: ["
             << tf_min_gpu_core_count << "]. "
             << "Using the default value: "
             << kDefaultMinGPUMultiprocessorCount;
  return kDefaultMinGPUMultiprocessorCount;
}

void BaseGPUDeviceFactory::GetValidDeviceIds(std::vector<int>* ids) {
  auto gpu_manager = GPUMachineManager();
  int min_gpu_core_count = GetMinGPUMultiprocessorCount();
  if (gpu_manager) {
    auto visible_device_count = gpu_manager->VisibleDeviceCount();
    for (int i = 0; i < gpu_manager->VisibleDeviceCount(); ++i) {
      auto exec_status = gpu_manager->ExecutorForDevice(i);
      if (!exec_status.ok()) {
        continue;
      }
      gpu::StreamExecutor* se = exec_status.ValueOrDie();
      const gpu::DeviceDescription& desc = se->GetDeviceDescription();
      int major, minor;
      if (!desc.cuda_compute_capability(&major, &minor)) {
        continue;
      }
      // Only consider GPUs with compute capability >= 3.5 (Kepler or
      // higher)
      if (major < 3 || (major == 3 && minor < 5)) {
        LOG(INFO) << "Ignoring gpu device "
                  << "(" << GetShortDeviceDescription(i, desc) << ") "
                  << "with Cuda compute capability " << major << "." << minor
                  << ". The minimum required Cuda capability is 3.5.";
        continue;
      }

      // TensorFlow currently places computation on devices assuming
      // they have similar capability.
      //
      // If there are multiple GPUs available on the machine, only
      // consider GPUs with 8 or more multiprocessors.
      //
      // TODO(vrv): In the medium term: we should only filter out GPUs
      // that are slow relative to the fastest GPU. In the long term,
      // TensorFlow should support automatic placement based on
      // capability.
      if (visible_device_count > 1) {
        if (desc.core_count() < min_gpu_core_count) {
          LOG(INFO) << "Ignoring gpu device "
                    << "(" << GetShortDeviceDescription(i, desc) << ") "
                    << "with Cuda multiprocessor count: " << desc.core_count()
                    << ". The minimum required count is " << min_gpu_core_count
                    << ". You can adjust this requirement with the env var "
                       "TF_MIN_GPU_MULTIPROCESSOR_COUNT.";
          continue;
        }
      }

      int new_id = ids->size();
      ids->push_back(i);

      LOG(INFO) << "Creating TensorFlow device (/gpu:" << new_id << ") -> "
                << "(" << GetShortDeviceDescription(i, desc) << ")";
    }
  }
}

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
