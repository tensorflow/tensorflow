#include "tensorflow/core/distributed_runtime/rpc/rdma.h"

#include <atomic>
#include <fstream>
#include <functional>
#include <iostream>
#include <list>
#include <map>
#include <memory>

#include <sys/fcntl.h>

#include <rdma/rdma_cma.h>
#include <rdma/rdma_verbs.h>

#include "tensorflow/core/common_runtime/bfc_allocator.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#if GOOGLE_CUDA
#include "tensorflow/core/common_runtime/gpu/gpu_util.h"
#include "tensorflow/core/common_runtime/gpu/process_state.h"
#endif
#include "tensorflow/core/framework/allocator_registry.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/protobuf/worker.pb.h"

namespace tensorflow {

namespace {

void EndpointDeleter(rdma_cm_id* id) {
  if (id) {
    rdma_destroy_id(id);
  }
}

void MRDeleter(ibv_mr* mr) {
  if (mr) {
    ibv_dereg_mr(mr);
  }
}

using RdmaEndpointPtr = std::unique_ptr<rdma_cm_id, decltype(&EndpointDeleter)>;

using MemoryRegionPtr = std::unique_ptr<ibv_mr, decltype(&MRDeleter)>;

bool IsGDRAvailable() {
  std::ifstream ifs("/proc/modules");
  string line;
  while (std::getline(ifs, line)) {
    auto sep = line.find(' ');
    CHECK_NE(sep, std::string::npos);
    if (line.substr(0, sep) == "nv_peer_mem") {
      return true;
    }
  }
  return false;
}

static int TryToReadNumaNode(ibv_device* device) {

  static const int kUnknownNumaNode = -1;

  string filename = string(device->ibdev_path) + "/device/numa_node";

  std::ifstream ifs(filename.c_str());
  string content;
  CHECK(std::getline(ifs, content));

  int32 value;
  if (strings::safe_strto32(content, &value)) {
    if (value < 0) {
      LOG(INFO) << "Successful NUMA node read from SysFS had negative value ("
                << value << "), but there must be at least one NUMA node"
                            ", so returning NUMA node zero";
      return 0;
    }
    LOG(INFO) << "NUMA node for device: " << device->name << " is " << value;
    return value;
  }
  return -kUnknownNumaNode;
}

class BasicCPUAllocator : public SubAllocator {
 public:
  ~BasicCPUAllocator() override {}

  void* Alloc(size_t alignment, size_t num_bytes) override {
    return port::AlignedMalloc(num_bytes, alignment);
  }
  void Free(void* ptr, size_t) override { port::AlignedFree(ptr); }
};

class BFCRdmaAllocator : public BFCAllocator {
 public:
  BFCRdmaAllocator() :
    BFCAllocator(new BasicCPUAllocator(), 1LL << 36, true, "cpu_rdma_bfc") {}
};

REGISTER_MEM_ALLOCATOR("BFCRdmaAllocator", 300, BFCRdmaAllocator);

class RdmaMemoryManager {
 public:
  RdmaMemoryManager() : pd_(nullptr) {

    using namespace std::placeholders;
    VisitableAllocator::Visitor alloc_visitor =
      std::bind(&RdmaMemoryManager::InsertMemoryRegion, this, _1, _2);
    VisitableAllocator::Visitor free_visitor =
      std::bind(&RdmaMemoryManager::EvictMemoryRegion, this, _1, _2);

    // Bind GPU using bus_id
    int num_device;
    ibv_context** devices = rdma_get_devices(&num_device);
    if (num_device == 0 || devices == nullptr) {
      LOG(WARNING) << "No RDMA device found";
      return;
    }

    // Used for host memory
    pd_ = ibv_alloc_pd(devices[0]);
    std::set<Allocator*> instrumented_;

    Allocator* allocators[] = {
#if GOOGLE_CUDA
      ProcessState::singleton()->GetCUDAHostAllocator(0),
      ProcessState::singleton()->GetCPUAllocator(0),
#endif
      cpu_allocator(),
    };

    for (Allocator* allocator : allocators) {
      CHECK(allocator);
      auto* visitable_allocator = dynamic_cast<VisitableAllocator*>(allocator);
      if (visitable_allocator &&
          instrumented_.find(allocator) == std::end(instrumented_)) {
        visitable_allocator->AddAllocVisitor(alloc_visitor);
        visitable_allocator->AddFreeVisitor(free_visitor);
        instrumented_.insert(allocator);
        LOG(INFO) << "Instrumenting allocator " << allocator->Name();
      }
    }

#if GOOGLE_CUDA
    if (IsGDRAvailable()) {
      // Note we don't free allocated GPU memory so there is no free visitor
      for (int i = 0; i < num_device; i++) {
        ibv_context* verbs = devices[i];
        int bus_id = TryToReadNumaNode(verbs->device) + 1;
        ProcessState::singleton()->AddGPUAllocVisitor(bus_id, alloc_visitor);
      }
    }
#endif
  }

  ~RdmaMemoryManager() {
    ibv_dealloc_pd(pd_);
  }

  void InsertMemoryRegion(void* addr, size_t length) {
    if (length == 0) return;
    const void* end = reinterpret_cast<char*>(addr) + length;
    int access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ;
    ibv_mr* mr = ibv_reg_mr(pd_, addr, length, access_flags);
    if (mr != nullptr) {
      mutex_lock l(mu_);
      auto iter = std::upper_bound(mrs_.begin(), mrs_.end(),
                                   end, &Comparator);
      mrs_.insert(iter, {mr, &MRDeleter});
    } else {
      LOG(WARNING) << "Cannot register memory region";
    }
  }

  void EvictMemoryRegion(void* addr, size_t length) {
    if (length == 0) return;
    const void* end = reinterpret_cast<char*>(addr) + length;
    mutex_lock l(mu_);
    auto iter = std::upper_bound(mrs_.begin(), mrs_.end(),
                                 end, &Comparator);
    if (iter != std::end(mrs_) && iter->get()->addr == addr) {
      mrs_.erase(iter);
    } else {
      LOG(WARNING) << "Failed to de-register memory region";
    }
  }

  ibv_mr* FindMemoryRegion(void* addr, size_t length) {
    if (length == 0) return nullptr;
    const void* end = reinterpret_cast<char*>(addr) + length;
    mutex_lock l(mu_);
    auto iter = std::upper_bound(mrs_.begin(), mrs_.end(),
                                 end, &Comparator);
    if (iter == std::end(mrs_) || iter->get()->addr > addr) {
      return nullptr;
    } else {
      return iter->get();
    }
  }

  static RdmaMemoryManager* Get() {
    static RdmaMemoryManager* rdma_manager = new RdmaMemoryManager();
    return rdma_manager;
  }

  ibv_pd* ProtectionDomain() const {
    return pd_;
  }

 private:
  static bool Comparator(const void* ptr, const MemoryRegionPtr& other) {
    return ptr < reinterpret_cast<char*>(other->addr) + other->length;
  }

  ibv_pd* pd_;
  std::list<MemoryRegionPtr> mrs_ GUARDED_BY(mu_);
  mutex mu_;
};

class RdmaReadClient : public RdmaClient {
 public:
  virtual Status ReadTensorViaDMA(Tensor* tensor,
                                  Device* dst_device,
                                  DeviceContext* dst_device_context,
                                  bool on_host,
                                  const Any& transport_options) override {

    RemoteMemoryRegion remote_mr;
    if (!transport_options.UnpackTo(&remote_mr)) {
      return errors::NotFound("No RDMA transport options found");
    }

    decltype(clients_)::iterator iter;
    bool success;
    {
      mutex_lock l(mu_);
      std::tie(iter, success) = clients_.insert(
          std::make_pair(std::make_pair(remote_mr.host(), remote_mr.port()),
                         RdmaEndpointPtr(nullptr, EndpointDeleter)));
      if (success || iter->second.get() == nullptr) {
        TF_RETURN_IF_ERROR(CreateEndpoint(remote_mr.host(), remote_mr.port(),
                                          iter->second));
      }
    }
    rdma_cm_id* id = iter->second.get();

    auto buffer = DMAHelper::buffer(tensor);
    void* addr = buffer->data();
    size_t length = buffer->size();
    ibv_mr* mr = RdmaMemoryManager::Get()->FindMemoryRegion(addr, length);

    Tensor host_copy;
    if (mr == nullptr &&
        dst_device->tensorflow_gpu_device_info() && (!on_host)) {
#if GOOGLE_CUDA
      Allocator* alloc = ProcessState::singleton()->GetCUDAHostAllocator(0);
      host_copy = Tensor(alloc, tensor->dtype(), tensor->shape());
      buffer = DMAHelper::buffer(&host_copy);
      addr = buffer->data();
      buffer->size();
      mr = RdmaMemoryManager::Get()->FindMemoryRegion(addr, length);
#endif
    }

    if (mr == nullptr) {
      return errors::Unavailable("Cannot find pinned memory region");
    }

    uint64_t start = Env::Default()->NowMicros();

    if (rdma_post_read(id, nullptr, buffer->data(), buffer->size(), mr,
                       0, remote_mr.addr(), remote_mr.rkey())) {
      perror("rdma_post_read");
      return errors::Unavailable("rdma_post_read failed");
    }

    ibv_send_wr wr = {};
    wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
    wr.imm_data = htonl(remote_mr.tensor_key());
    wr.send_flags = IBV_SEND_FENCE | IBV_SEND_SIGNALED;
    ibv_send_wr* bad_wr;
    if (ibv_post_send(id->qp, &wr, &bad_wr)) {
      return errors::Unavailable("ibv_post_send failed");
    }

    ibv_wc wc = {};
    int ret = rdma_get_send_comp(id, &wc);
    if (ret < 0 || wc.status) {
      return errors::Unavailable(ibv_wc_status_str(wc.status));
    }

#if GOOGLE_CUDA
    if (host_copy.NumElements() > 0) {
      Status s;
      Notification n;
      GPUUtil::CopyCPUTensorToGPU(&host_copy, dst_device_context,
                                  dst_device, tensor,
                                  [&s, &n](const Status& status) {
                                    s.Update(status);
                                    n.Notify();
                                  });
      n.WaitForNotification();
      if (!s.ok()) {
        return s;
      }
    }
#endif

    // TODO: Remove code used for debugging purposes only
    string tensor_debug_string;
    if (dst_device->tensorflow_gpu_device_info() && (!on_host)) {
#if GOOGLE_CUDA
      tensor_debug_string = GPUUtil::MemoryDebugString(dst_device, tensor);
#else
      return errors::Internal("No GPU device in process");
#endif
    } else {
      tensor_debug_string = tensor->DebugString();
    }

    uint64_t checksum = 0;
#if GOOGLE_CUDA
    if (dst_device->tensorflow_gpu_device_info() && (!on_host)) {
      checksum = GPUUtil::Checksum(dst_device, dst_device_context, *tensor);
    } else {
      checksum = GPUUtil::Checksum(*tensor);
    }
    CHECK(checksum == remote_mr.checksum())
        << "Checksum mismatch for "
        << tensor_debug_string;
#endif

    uint64_t end = Env::Default()->NowMicros();

    VLOG(2) << "RDMA from remote memory region " << remote_mr.rkey()
            << " to " << tensor_debug_string
            << " of size " << buffer->size()
            << " with tensor key " << remote_mr.tensor_key()
            << " took " << (end - start) << " micros";

    return Status::OK();
  }

 private:
  Status CreateEndpoint(const string& host, const string& port,
                        RdmaEndpointPtr& endpoint) {

    ibv_pd* pd = RdmaMemoryManager::Get()->ProtectionDomain();

    rdma_addrinfo* addrinfo;
    rdma_addrinfo hints = {};
    hints.ai_port_space = RDMA_PS_TCP;
    if (rdma_getaddrinfo(const_cast<char*>(host.c_str()),
                          const_cast<char*>(port.c_str()),
                          &hints, &addrinfo)) {
      return errors::InvalidArgument("Cannot connect to rdma://",
                                      host, ":", port);
    }

    ibv_qp_init_attr init_attr = {};
    init_attr.qp_type = IBV_QPT_RC;
    init_attr.cap.max_recv_wr = 1;
    init_attr.cap.max_send_wr = 32;
    init_attr.cap.max_recv_sge = 1;
    init_attr.cap.max_send_sge = 1;

    rdma_cm_id* id;
    if (rdma_create_ep(&id, addrinfo, pd, &init_attr)) {
      rdma_freeaddrinfo(addrinfo);
      return errors::Unavailable("Cannot connect to endpoint rdma://",
                                  host, ":", port);
    }
    rdma_freeaddrinfo(addrinfo);

    if (rdma_connect(id, nullptr)) {
      rdma_destroy_ep(id);
      return errors::Unavailable("Cannot connect to endpoint rdma://",
                                  host, ":", port);
    }

    LOG(INFO) << "RDMA endpoint connected to rdma://" << host << ":" << port;
    endpoint = RdmaEndpointPtr(id, EndpointDeleter);
    return Status::OK();
}

  mutex mu_;
  std::map<std::pair<string, string>, RdmaEndpointPtr> clients_ GUARDED_BY(mu_);
};

class RdmaReadServer : public RdmaServer {
 public:
  RdmaReadServer(const string& host, const string& port)
    : host_(host), port_(port),
      listening_(nullptr, EndpointDeleter),
      stopped_(false), next_key_(0) {}

  virtual ~RdmaReadServer() {
    Stop();
  }

  virtual Status Init() override {

    ibv_pd* pd = RdmaMemoryManager::Get()->ProtectionDomain();

    // Resolve address passed from ConfigProto
    rdma_addrinfo* addrinfo;
    rdma_addrinfo hints = {};
    hints.ai_port_space = RDMA_PS_TCP;
    hints.ai_flags = RAI_PASSIVE;
    if (rdma_getaddrinfo(const_cast<char*>(host_.c_str()),
                         const_cast<char*>(port_.c_str()),
                         &hints, &addrinfo)) {
      return errors::Unavailable("Cannot resolve rdma://", host_, ":", port_);
    }

    ibv_qp_init_attr init_attr = {};
    init_attr.qp_type = IBV_QPT_RC;
    init_attr.cap.max_recv_wr = 1024;
    init_attr.cap.max_send_wr = 1;
    init_attr.cap.max_recv_sge = 1;
    init_attr.cap.max_send_sge = 1;

    // Create listening endpoint
    rdma_cm_id* id;
    if (rdma_create_ep(&id, addrinfo, pd, &init_attr)) {
      return errors::Unavailable("Cannot bind to rdma://", host_, ":", port_);
    }
    listening_.reset(id);
    rdma_freeaddrinfo(addrinfo);

    // Listen without backlog
    if (rdma_listen(id, 0)) {
      rdma_destroy_ep(id);
      return errors::Unavailable("Cannot listen on rdma://", host_, ":", port_);
    }
    LOG(INFO) << "RDMA server is listening on " << host_ << ":" << port_;

    int flags = fcntl(id->channel->fd, F_GETFL, 0);
    if (fcntl(id->channel->fd, F_SETFL, flags | O_NONBLOCK)) {
      return errors::Unavailable("Cannot set server to non-blocking mode");
    }

    return Status::OK();
  }

  virtual void Run() override {

    stopped_ = false;
    while (!stopped_) {
      // Accept incoming connections
      rdma_cm_id* id = nullptr;
      if (!rdma_get_request(listening_.get(), &id)) {
        if (!rdma_accept(id, nullptr)) {
          LOG(INFO) << "Accepted new RDMA connection";
          if (rdma_post_recvv(id, nullptr, nullptr, 0)) {
            perror("rdma_post_recvv");
            LOG(ERROR) << "rdma_post_recvv failed";
            EndpointDeleter(id);
          } else {
            clients_.push_back({id, EndpointDeleter});
          }
        }
      }

      // Polling work completions
      const bool error = true;
      clients_.remove_if([this](const RdmaEndpointPtr& client) {
        ibv_wc wc[32];
        int ret = ibv_poll_cq(client->recv_cq, 32, wc);
        if (ret < 0) {
          LOG(ERROR) << "ibv_poll_cq failed";
          return error;
        }
        for (int i = 0; i < ret; i++) {
          if (wc[i].opcode != IBV_WC_RECV_RDMA_WITH_IMM) {
            LOG(ERROR) << "Received unknown operation " << wc[i].opcode;
            return error;
          }
          if (wc[i].status != 0) {
            LOG(ERROR) << ibv_wc_status_str(wc[i].status);
            return error;
          }
          uint32_t tensor_key = ntohl(wc[i].imm_data);
          {
            mutex_lock l(mu_);
            auto iter = tensor_buffers_.find(tensor_key);
            if (iter == std::end(tensor_buffers_)) {
              LOG(ERROR) << "Cannot find tensor buffer for tensor key "
                          << tensor_key;
              return error;
            } else {
              const TensorBuffer* buffer = iter->second;
              buffer->Unref();
              tensor_buffers_.erase(iter);
            }
          }
          if (rdma_post_recvv(client.get(), nullptr, nullptr, 0)) {
            perror("rdma_post_recvv");
            LOG(ERROR) << "rdma_post_recvv failed";
            return error;
          }
        }
        return !error;
      });
    }
  }

  virtual Status RegisterTensorDMA(const Tensor& tensor,
                                   Device* src_device,
                                   DeviceContext* src_device_context,
                                   bool on_host,
                                   Any* mutable_transport_options) override {

    auto buffer = DMAHelper::buffer(&tensor);
    void* addr = buffer->data();
    size_t length = buffer->size();
    if (length == 0) {
      return errors::Unavailable("Cannot register tensor buffer of size 0");
    }

    ibv_mr* mr = RdmaMemoryManager::Get()->FindMemoryRegion(addr, length);

    Tensor host_copy;
    if (mr == nullptr &&
        src_device->tensorflow_gpu_device_info() && (!on_host)) {
#if GOOGLE_CUDA
      Allocator* alloc = ProcessState::singleton()->GetCUDAHostAllocator(0);
      host_copy = Tensor(alloc, tensor.dtype(), tensor.shape());
      Status s;
      Notification n;
      GPUUtil::CopyGPUTensorToCPU(src_device,
                                  src_device_context,
                                  &tensor,
                                  &host_copy,
                                  [&s, &n](const Status& status) {
                                    s.Update(status);
                                    n.Notify();
                                  });
      n.WaitForNotification();
      if (!s.ok()) {
        return s;
      }
      buffer = DMAHelper::buffer(&host_copy);
      addr = buffer->data();
      buffer->size();
      mr = RdmaMemoryManager::Get()->FindMemoryRegion(addr, length);
#endif
    }

    if (mr == nullptr) {
      return errors::Unavailable("Cannot find pinned memory region");
    }

    buffer->Ref();
    uint32_t tensor_key = next_key_++;
    {
      mutex_lock l(mu_);
      tensor_buffers_.insert({tensor_key, buffer});
    }

    // TODO: Remove code used for debugging purposes only
    uint64_t checksum = 0;
#if GOOGLE_CUDA
    if (src_device->tensorflow_gpu_device_info() && (!on_host)) {
      checksum = GPUUtil::Checksum(src_device, src_device_context, tensor);
    } else {
      checksum = GPUUtil::Checksum(tensor);
    }
#endif

    RemoteMemoryRegion remote_mr;
    remote_mr.set_host(host_);
    remote_mr.set_port(port_);
    remote_mr.set_addr(reinterpret_cast<uint64_t>(addr));
    remote_mr.set_rkey(mr->rkey);
    remote_mr.set_tensor_key(tensor_key);
    remote_mr.set_checksum(checksum);
    mutable_transport_options->PackFrom(remote_mr);

    return Status::OK();
  }

  virtual void Stop() override {
    stopped_ = true;
  }

 private:
  const string host_;
  const string port_;
  RdmaEndpointPtr listening_;

  std::list<RdmaEndpointPtr> clients_;
  std::atomic<bool> stopped_;

  mutex mu_;
  std::map<uint32_t, const TensorBuffer*> tensor_buffers_ GUARDED_BY(mu_);
  std::atomic<uint32_t> next_key_;
};

}  // namespace

RdmaServer* NewRdmaServer(const string& host, const string& port) {
  return new RdmaReadServer(host, port);
}

RdmaClient* NewRdmaClient() {
  return new RdmaReadClient;
}

}  // namespace tensorflow
