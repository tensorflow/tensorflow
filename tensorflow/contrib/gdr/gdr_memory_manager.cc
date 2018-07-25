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

#ifdef TENSORFLOW_USE_GDR

#include "tensorflow/contrib/gdr/gdr_memory_manager.h"

#include <atomic>
#include <cerrno>
#include <fstream>
#include <list>
#include <map>
#include <set>

#include <fcntl.h>
#include <rdma/rdma_cma.h>
#include <rdma/rdma_verbs.h>
#include <sys/epoll.h>

#include "tensorflow/contrib/gdr/gdr.pb.h"
#include "tensorflow/core/common_runtime/bfc_allocator.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/pool_allocator.h"
#include "tensorflow/core/common_runtime/process_state.h"
#if GOOGLE_CUDA
#include "tensorflow/core/common_runtime/gpu/gpu_process_state.h"
#include "tensorflow/core/common_runtime/gpu/gpu_util.h"
#endif  // GOOGLE_CUDA
#include "tensorflow/core/framework/allocator_registry.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {

namespace {

bool IsGDRAvailable() {
#if defined(__APPLE__)
  return false;
#elif defined(PLATFORM_WINDOWS)
  return false;
#else
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
#endif
}

int TryToReadNumaNode(ibv_device* device) {
#if defined(__APPLE__)
  LOG(INFO) << "OS X does not support NUMA - returning NUMA node 0";
  return 0;
#elif defined(PLATFORM_WINDOWS)
  // Windows support for NUMA is not currently implemented. Return node 0.
  return 0;
#else
  VLOG(2) << "Trying to read NUMA node for device: " << device->name;
  static const int kUnknownNumaNode = -1;

  auto filename = string(device->ibdev_path) + "/device/numa_node";

  std::ifstream ifs(filename.c_str());
  string content;
  CHECK(std::getline(ifs, content));

  int32 value;
  if (strings::safe_strto32(content, &value)) {
    if (value < 0) {
      LOG(INFO) << "Successful NUMA node read from SysFS had negative value ("
                << value
                << "), but there must be at least one NUMA node"
                   ", so returning NUMA node zero";
      return 0;
    }
    LOG(INFO) << "NUMA node for device: " << device->name << " is " << value;
    return value;
  }
  return kUnknownNumaNode;
#endif
}

void EndpointDeleter(rdma_cm_id* id) {
  if (id) {
    rdma_destroy_ep(id);
  }
}

void MRDeleter(ibv_mr* mr) {
  if (mr) {
    rdma_dereg_mr(mr);
  }
}

using RdmaEndpointPtr = std::unique_ptr<rdma_cm_id, decltype(&EndpointDeleter)>;

using MemoryRegionPtr = std::unique_ptr<ibv_mr, decltype(&MRDeleter)>;

class GdrMemoryManager : public RemoteMemoryManager {
 public:
  GdrMemoryManager(const string& host, const string& port);

  virtual ~GdrMemoryManager();

  virtual Status Init() override;

  virtual void Run() override;

  virtual void Stop() override;

  virtual void TransportOptionsFromTensor(
      ::google::protobuf::Any* mutable_transport_options, const Tensor& tensor,
      Device* device, DeviceContext* device_context, bool on_host,
      StatusCallback done) override;

  virtual void TensorFromTransportOptions(
      Tensor* tensor, const ::google::protobuf::Any& transport_options,
      Device* device, DeviceContext* device_context, bool on_host,
      StatusCallback done) override;

 protected:
  Status CreateEndpoint(const string& host, const string& port,
                        RdmaEndpointPtr& endpoint);

  static bool Comparator(const void* ptr, const MemoryRegionPtr& other) {
    return ptr < reinterpret_cast<char*>(other->addr) + other->length;
  }

  ibv_mr* FindMemoryRegion(void* addr, size_t length);

  void InsertMemoryRegion(void* addr, size_t length);

  void EvictMemoryRegion(void* addr, size_t length);

 private:
  const string host_;
  const string port_;
  RdmaEndpointPtr listening_;
  std::atomic<bool> stopped_;
  int epfd_;

  // Server side endpoints
  // Accessed sequentially in Run() so not protected by lock
  std::list<RdmaEndpointPtr> server_clients_;

  using TensorKey = uint32_t;
  std::atomic<TensorKey> next_key_;

  // Server side on-the-fly tensor buffers
  mutex server_mu_;
  std::map<TensorKey, const TensorBuffer*> tensor_buffers_
      GUARDED_BY(server_mu_);

  // Client side endpoints
  mutex client_mu_;
  std::map<std::pair<string, string>, RdmaEndpointPtr> clients_
      GUARDED_BY(cient_mu_);

  // Managed memory regions
  mutex alloc_mu_;
  std::vector<MemoryRegionPtr> mrs_ GUARDED_BY(alloc_mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(GdrMemoryManager);
};

// TODO(byronyi): remove this class and its registration when the default
// cpu_allocator() returns visitable allocator, or cpu_allocator() is no
// longer in use.
class BFCRdmaAllocator : public BFCAllocator {
 public:
  BFCRdmaAllocator()
      : BFCAllocator(new BasicCPUAllocator(port::kNUMANoAffinity), 1LL << 36,
                     true, "cpu_rdma_bfc") {}
};
class BFCRdmaAllocatorFactory : public AllocatorFactory {
 public:
  Allocator* CreateAllocator() override { return new BFCRdmaAllocator; }

  virtual SubAllocator* CreateSubAllocator(int numa_node) {
    return new BasicCPUAllocator(numa_node);
  }
};

REGISTER_MEM_ALLOCATOR("BFCRdmaAllocator", 101, BFCRdmaAllocatorFactory);

GdrMemoryManager::GdrMemoryManager(const string& host, const string& port)
    : host_(host),
      port_(port),
      listening_(nullptr, EndpointDeleter),
      stopped_(true),
      next_key_(0) {}

GdrMemoryManager::~GdrMemoryManager() { close(epfd_); }

Status GdrMemoryManager::Init() {
  epfd_ = epoll_create1(0);
  if (epfd_ == -1) {
    return errors::Unavailable(strerror(errno), ": ", "epoll_create");
  }

  rdma_addrinfo* addrinfo;
  rdma_addrinfo hints = {};
  hints.ai_port_space = RDMA_PS_TCP;
  hints.ai_flags = RAI_PASSIVE;
  if (rdma_getaddrinfo(const_cast<char*>(host_.c_str()),
                       const_cast<char*>(port_.c_str()), &hints, &addrinfo)) {
    return errors::Unavailable(strerror(errno), ": ", "cannot resolve rdma://",
                               host_, ":", port_);
  }

  ibv_qp_init_attr init_attr = {};
  init_attr.qp_type = IBV_QPT_RC;
  init_attr.cap.max_recv_wr = 32;
  init_attr.cap.max_send_wr = 1;
  init_attr.cap.max_recv_sge = 1;
  init_attr.cap.max_send_sge = 1;

  // Create listening endpoint
  rdma_cm_id* id;
  if (rdma_create_ep(&id, addrinfo, nullptr, &init_attr)) {
    return errors::Unavailable(strerror(errno), ": ", "cannot bind to rdma://",
                               host_, ":", port_);
  }
  listening_.reset(id);
  rdma_freeaddrinfo(addrinfo);

  // Listen without backlog
  if (rdma_listen(listening_.get(), 0)) {
    return errors::Unavailable(strerror(errno), ": ",
                               "cannot listen on rdma://", host_, ":", port_);
  }
  LOG(INFO) << "RDMA server is listening on " << host_ << ":" << port_;

  if (listening_->verbs == nullptr) {
    return errors::Unimplemented(
        "Unsupported address ", host_, ":", port_,
        " as it does not bind to a particular RDMA device");
  }

  int flags = fcntl(listening_->channel->fd, F_GETFL, 0);
  if (fcntl(listening_->channel->fd, F_SETFL, flags | O_NONBLOCK)) {
    return errors::Unavailable(strerror(errno), ": ",
                               "cannot set server to non-blocking mode");
  }

  epoll_event event = {};
  event.events = EPOLLIN | EPOLLPRI;
  event.data.ptr = listening_.get();
  if (epoll_ctl(epfd_, EPOLL_CTL_ADD, listening_->channel->fd, &event)) {
    return errors::Unavailable(strerror(errno), ": ",
                               "cannot add server to epoll");
  }

  Allocator* allocators[] = {
#if GOOGLE_CUDA
    GPUProcessState::singleton()->GetCUDAHostAllocator(0),
#endif  // GOOGLE_CUDA
    ProcessState::singleton()->GetCPUAllocator(0),
    cpu_allocator(),
  };

  using namespace std::placeholders;
  VisitableAllocator::Visitor alloc_visitor =
      std::bind(&GdrMemoryManager::InsertMemoryRegion, this, _1, _2);
  VisitableAllocator::Visitor free_visitor =
      std::bind(&GdrMemoryManager::EvictMemoryRegion, this, _1, _2);

  std::set<Allocator*> instrumented_;

  // Host memory allocators
  for (Allocator* allocator : allocators) {
    auto* visitable_allocator = dynamic_cast<VisitableAllocator*>(allocator);
    CHECK(visitable_allocator)
        << "is not visitable for instrumentation" << allocator->Name();
    // Make sure we don't instrument the same allocator twice
    if (instrumented_.find(allocator) == std::end(instrumented_)) {
      visitable_allocator->AddAllocVisitor(alloc_visitor);
      visitable_allocator->AddFreeVisitor(free_visitor);
      instrumented_.insert(allocator);
      LOG(INFO) << "Instrumenting CPU allocator " << allocator->Name();
    }
  }

#if GOOGLE_CUDA
  VisitableAllocator::Visitor cuda_alloc_visitor =
      std::bind(&GdrMemoryManager::InsertMemoryRegion, this, _1, _2);
  if (IsGDRAvailable()) {
    // Note we don't free allocated GPU memory so there is no free visitor
    int32_t bus_id = TryToReadNumaNode(listening_->verbs->device) + 1;
    GPUProcessState::singleton()->AddGPUAllocVisitor(bus_id,
                                                     cuda_alloc_visitor);
    LOG(INFO) << "Instrumenting GPU allocator with bus_id " << bus_id;
  }
#endif  // GOOGLE_CUDA

  return Status::OK();
}

void GdrMemoryManager::Run() {
  stopped_ = false;
  while (!stopped_) {
    epoll_event events[32];
    int ret = epoll_wait(epfd_, events, 32, 1);
    if (ret == -1) {
      LOG(ERROR) << "epoll_wait: " << strerror(errno);
      return;
    }
    for (int i = 0; i < ret; i++) {
      rdma_cm_id* id = static_cast<rdma_cm_id*>(events[i].data.ptr);
      if (id == listening_.get()) {
        // Accept incoming connections
        if (!rdma_get_request(listening_.get(), &id)) {
          if (!rdma_accept(id, nullptr)) {
            LOG(INFO) << "Accepted new RDMA connection";
            if (ibv_req_notify_cq(id->recv_cq, 0)) {
              LOG(ERROR) << strerror(errno) << ": ibv_req_notify_cq failed";
              EndpointDeleter(id);
              continue;
            }
            for (int i = 0; i < 32; i++) {
              if (rdma_post_recvv(id, nullptr, nullptr, 0)) {
                LOG(ERROR) << strerror(errno) << ": rdma_post_recvv failed";
                EndpointDeleter(id);
                continue;
              }
            }
            int flags = fcntl(id->recv_cq_channel->fd, F_GETFL, 0);
            if (fcntl(id->recv_cq_channel->fd, F_SETFL, flags | O_NONBLOCK)) {
              LOG(ERROR) << strerror(errno)
                         << ": cannot set server_client to non-blocking mode";
              EndpointDeleter(id);
              continue;
            }
            epoll_event event = {};
            event.events = EPOLLIN | EPOLLPRI;
            event.data.ptr = id;
            if (epoll_ctl(epfd_, EPOLL_CTL_ADD, id->recv_cq_channel->fd,
                          &event)) {
              LOG(ERROR) << strerror(errno)
                         << ": cannot add server client to epoll";
              EndpointDeleter(id);
              continue;
            }
            server_clients_.push_back({id, EndpointDeleter});
          }
        }
      } else {
        // Polling work completions
        ibv_cq* cq;
        void* context;
        if (!ibv_get_cq_event(id->recv_cq_channel, &cq, &context)) {
          ibv_ack_cq_events(id->recv_cq, 1);
          if (ibv_req_notify_cq(id->recv_cq, 0)) {
            LOG(ERROR) << strerror(errno) << ": ibv_req_notify_cq failed";
            continue;
          }
          ibv_wc wc[32];
          int ret = ibv_poll_cq(id->recv_cq, 32, wc);
          if (ret < 0) {
            LOG(ERROR) << "ibv_poll_cq failed";
            continue;
          }
          for (int i = 0; i < ret; i++) {
            if (wc[i].opcode != IBV_WC_RECV_RDMA_WITH_IMM) {
              LOG(ERROR) << "Received unknown operation " << wc[i].opcode;
            }
            if (wc[i].status != 0) {
              LOG(ERROR) << ibv_wc_status_str(wc[i].status);
            }
            TensorKey tensor_key = ntohl(wc[i].imm_data);
            {
              mutex_lock l(server_mu_);
              auto iter = tensor_buffers_.find(tensor_key);
              if (iter == std::end(tensor_buffers_)) {
                LOG(ERROR) << "Cannot find tensor buffer for tensor key "
                           << tensor_key;
              } else {
                const TensorBuffer* buffer = iter->second;
                buffer->Unref();
                tensor_buffers_.erase(iter);
              }
            }
            if (rdma_post_recvv(id, nullptr, nullptr, 0)) {
              perror("rdma_post_recvv");
              LOG(ERROR) << "rdma_post_recvv failed";
              continue;
            }
          }
        }
      }
    }
  }
}

void GdrMemoryManager::Stop() { stopped_ = true; }

void GdrMemoryManager::TransportOptionsFromTensor(
    ::google::protobuf::Any* mutable_transport_options, const Tensor& tensor,
    Device* device, DeviceContext* device_context, bool on_host,
    StatusCallback done) {
  auto buffer = DMAHelper::buffer(&tensor);
  void* addr = buffer->data();
  size_t length = buffer->size();
  if (length == 0) {
    done(errors::Unavailable("Cannot register tensor buffer of size 0"));
    return;
  }

  ibv_mr* mr = FindMemoryRegion(addr, length);

#if GOOGLE_CUDA
  if (!on_host) {
    Allocator* alloc = GPUProcessState::singleton()->GetCUDAHostAllocator(0);
    Tensor* host_copy = new Tensor(alloc, tensor.dtype(), tensor.shape());
    GPUUtil::CopyGPUTensorToCPU(
        device, device_context, &tensor, host_copy,
        [done, host_copy, mutable_transport_options, this](const Status& s) {
          if (!s.ok()) {
            done(s);
            delete host_copy;
            return;
          }
          auto buffer = DMAHelper::buffer(host_copy);
          void* addr = buffer->data();
          size_t length = buffer->size();
          ibv_mr* mr = FindMemoryRegion(addr, length);

          if (mr == nullptr) {
            done(errors::Unavailable("Cannot find pinned memory region"));
            delete host_copy;
            return;
          }

          buffer->Ref();
          TensorKey tensor_key = next_key_++;
          {
            mutex_lock l(server_mu_);
            tensor_buffers_.insert(std::make_pair(tensor_key, buffer));
          }

          uint64_t checksum = 0;
          if (VLOG_IS_ON(2)) {
            checksum = GPUUtil::Checksum(*host_copy);
          }

          RemoteMemoryRegion remote_mr;
          remote_mr.set_host(host_);
          remote_mr.set_port(port_);
          remote_mr.set_addr(reinterpret_cast<uint64_t>(addr));
          remote_mr.set_rkey(mr->rkey);
          remote_mr.set_tensor_key(tensor_key);
          remote_mr.set_checksum(checksum);
          mutable_transport_options->PackFrom(remote_mr);

          done(Status::OK());
          delete host_copy;
        });
    return;
  }
#endif

  if (mr == nullptr) {
    done(errors::Unavailable("Cannot find pinned memory region"));
    return;
  }

  buffer->Ref();
  TensorKey tensor_key = next_key_++;
  {
    mutex_lock l(server_mu_);
    tensor_buffers_.insert(std::make_pair(tensor_key, buffer));
  }

  uint64_t checksum = 0;
  if (VLOG_IS_ON(2)) {
#ifdef GOOGLE_CUDA
    if (!on_host) {
      checksum = GPUUtil::Checksum(device, device_context, tensor);
    } else {
      checksum = GPUUtil::Checksum(tensor);
    }
#endif
  }

  RemoteMemoryRegion remote_mr;
  remote_mr.set_host(host_);
  remote_mr.set_port(port_);
  remote_mr.set_addr(reinterpret_cast<uint64_t>(addr));
  remote_mr.set_rkey(mr->rkey);
  remote_mr.set_tensor_key(tensor_key);
  remote_mr.set_checksum(checksum);
  mutable_transport_options->PackFrom(remote_mr);

  done(Status::OK());
}

void GdrMemoryManager::TensorFromTransportOptions(
    Tensor* tensor, const ::google::protobuf::Any& transport_options,
    Device* device, DeviceContext* device_context, bool on_host,
    StatusCallback done) {
  RemoteMemoryRegion remote_mr;
  if (!transport_options.UnpackTo(&remote_mr)) {
    done(errors::NotFound("No RDMA transport options found"));
    return;
  }

  auto buffer = DMAHelper::buffer(tensor);
  void* addr = buffer->data();
  size_t length = buffer->size();
  ibv_mr* mr = FindMemoryRegion(addr, length);

  Tensor host_copy;
#if GOOGLE_CUDA
  if (mr == nullptr && !on_host) {
    Allocator* alloc = GPUProcessState::singleton()->GetCUDAHostAllocator(0);
    host_copy = Tensor(alloc, tensor->dtype(), tensor->shape());
    buffer = DMAHelper::buffer(&host_copy);
    addr = buffer->data();
    length = buffer->size();
    mr = FindMemoryRegion(addr, length);
  }
#endif  // GOOGLE_CUDA

  if (mr == nullptr) {
    done(errors::Unavailable("Cannot find pinned memory region"));
    return;
  }

  decltype(clients_)::iterator iter;
  bool success;
  {
    mutex_lock l(client_mu_);
    std::tie(iter, success) = clients_.insert(
        std::make_pair(std::make_pair(remote_mr.host(), remote_mr.port()),
                       RdmaEndpointPtr(nullptr, EndpointDeleter)));
    if (success || iter->second.get() == nullptr) {
      Status s =
          CreateEndpoint(remote_mr.host(), remote_mr.port(), iter->second);
      if (!s.ok()) {
        done(s);
        return;
      }
    }
  }
  rdma_cm_id* id = iter->second.get();

  uint64_t start = Env::Default()->NowMicros();

  if (rdma_post_read(id, nullptr, buffer->data(), buffer->size(), mr, 0,
                     remote_mr.addr(), remote_mr.rkey())) {
    done(errors::Unavailable(strerror(errno), ": ", "rdma_post_read failed"));
    return;
  }

  ibv_send_wr wr = {};
  wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  wr.imm_data = htonl(remote_mr.tensor_key());
  wr.send_flags = IBV_SEND_SIGNALED;
  ibv_send_wr* bad_wr;
  if (ibv_post_send(id->qp, &wr, &bad_wr)) {
    done(errors::Unavailable(strerror(errno), ": ", "ibv_post_send failed"));
    return;
  }

  ibv_wc wc = {};
  int ret;
  while ((ret = ibv_poll_cq(id->send_cq, 1, &wc)) == 0)
    ;
  if (ret < 0 || wc.status) {
    done(errors::Unavailable(ibv_wc_status_str(wc.status)));
    return;
  }

#if GOOGLE_CUDA
  if (host_copy.NumElements() > 0) {
    uint64_t checksum = 0;
    if (VLOG_IS_ON(2)) {
      checksum = GPUUtil::Checksum(host_copy);
      CHECK(checksum == remote_mr.checksum())
          << "Checksum mismatch: " << checksum << "!=" << remote_mr.checksum();
    }
    Tensor* ref = new Tensor;
    std::swap(host_copy, *ref);
    GPUUtil::CopyCPUTensorToGPU(
        ref, device_context, device, tensor,
        [ref, done, buffer, remote_mr, start](const Status& s) {
          if (!s.ok()) {
            done(s);
            delete ref;
            return;
          }
          uint64_t end = Env::Default()->NowMicros();

          VLOG(2) << "RDMA from remote memory region " << remote_mr.rkey()
                  << " of size " << buffer->size() << " with tensor key "
                  << remote_mr.tensor_key() << " took " << (end - start)
                  << " micros";
          done(Status::OK());
          delete ref;
        });
    return;
  }
#endif  // GOOGLE_CUDA

  uint64_t end = Env::Default()->NowMicros();

  VLOG(2) << "RDMA from remote memory region " << remote_mr.rkey()
          << " of size " << buffer->size() << " with tensor key "
          << remote_mr.tensor_key() << " took " << (end - start) << " micros";

  uint64_t checksum = 0;
  if (VLOG_IS_ON(2)) {
#ifdef GOOGLE_CUDA
    if (device->tensorflow_gpu_device_info() && (!on_host)) {
      checksum = GPUUtil::Checksum(device, device_context, *tensor);
    } else {
      checksum = GPUUtil::Checksum(*tensor);
    }
    CHECK(checksum == remote_mr.checksum())
        << "Checksum mismatch: " << checksum << "!=" << remote_mr.checksum();
#endif
  }
  done(Status::OK());
}

Status GdrMemoryManager::CreateEndpoint(const string& host, const string& port,
                                        RdmaEndpointPtr& endpoint) {
  rdma_addrinfo* addrinfo;
  rdma_addrinfo hints = {};
  hints.ai_port_space = RDMA_PS_TCP;
  if (rdma_getaddrinfo(const_cast<char*>(host.c_str()),
                       const_cast<char*>(port.c_str()), &hints, &addrinfo)) {
    return errors::InvalidArgument(
        strerror(errno), ": ", "cannot connect to rdma://", host, ":", port);
  }

  ibv_qp_init_attr init_attr = {};
  init_attr.qp_type = IBV_QPT_RC;
  init_attr.cap.max_recv_wr = 1;
  init_attr.cap.max_send_wr = 32;
  init_attr.cap.max_recv_sge = 1;
  init_attr.cap.max_send_sge = 1;

  rdma_cm_id* id;
  if (rdma_create_ep(&id, addrinfo, nullptr, &init_attr)) {
    rdma_freeaddrinfo(addrinfo);
    return errors::Unavailable(strerror(errno), ": ",
                               "cannot create endpoint to rdma://", host, ":",
                               port);
  }
  rdma_freeaddrinfo(addrinfo);

  if (rdma_connect(id, nullptr)) {
    rdma_destroy_ep(id);
    return errors::Unavailable(strerror(errno), ": ",
                               "cannot connect to rdma://", host, ":", port);
  }

  LOG(INFO) << "RDMA endpoint connected to rdma://" << host << ":" << port;
  endpoint = RdmaEndpointPtr(id, EndpointDeleter);
  return Status::OK();
}

ibv_mr* GdrMemoryManager::FindMemoryRegion(void* addr, size_t length) {
  if (length == 0) return nullptr;
  mutex_lock l(alloc_mu_);
  auto iter = std::upper_bound(mrs_.begin(), mrs_.end(), addr, &Comparator);
  if (iter == std::end(mrs_) || iter->get()->addr > addr) {
    return nullptr;
  } else {
    return iter->get();
  }
}

void GdrMemoryManager::InsertMemoryRegion(void* addr, size_t length) {
  if (length == 0) return;
  ibv_mr* mr = rdma_reg_read(listening_.get(), addr, length);
  if (mr != nullptr) {
    mutex_lock l(alloc_mu_);
    auto iter = std::upper_bound(mrs_.begin(), mrs_.end(), addr, &Comparator);
    mrs_.insert(iter, {mr, &MRDeleter});
  } else {
    LOG(WARNING) << "Cannot register memory region";
  }
}

void GdrMemoryManager::EvictMemoryRegion(void* addr, size_t length) {
  if (length == 0) return;
  mutex_lock l(alloc_mu_);
  auto iter = std::upper_bound(mrs_.begin(), mrs_.end(), addr, &Comparator);
  if (iter != std::end(mrs_) && iter->get()->addr == addr) {
    mrs_.erase(iter);
  } else {
    LOG(WARNING) << "Failed to de-register memory region";
  }
}

}  // namespace

RemoteMemoryManager* CreateRemoteMemoryManager(const string& host,
                                               const string& port) {
  return new GdrMemoryManager(host, port);
}

}  // namespace tensorflow

#endif  // TENSORFLOW_USE_GDR
