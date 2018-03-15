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

#ifdef TENSORFLOW_USE_VERBS

#include "tensorflow/contrib/verbs/rdma_mgr.h"
#include <fstream>
#include <vector>
#include "tensorflow/contrib/verbs/grpc_verbs_client.h"
#include "tensorflow/contrib/verbs/verbs_service.pb.h"
#include "tensorflow/core/common_runtime/bfc_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_util.h"
#include "tensorflow/core/common_runtime/gpu/process_state.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_cache.h"
#include "tensorflow/core/distributed_runtime/session_mgr.h"
#include "tensorflow/core/framework/allocator_registry.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

RdmaMgr::RdmaMgr(const WorkerEnv* const worker_env,
                 GrpcChannelCache* const channel_cache)
    : worker_env_(worker_env), channel_cache_(channel_cache) {
  rdma_adapter_ = new RdmaAdapter(worker_env_);
  // hardcoded to default session (legacy_session_)
  // TODO: use WorkerSessionForSession
  // need to pass in session handle
  local_worker_ = worker_env_->session_mgr->LegacySession()->worker_name;
  std::vector<string> workers;
  worker_env_->session_mgr->LegacySession()->worker_cache->ListWorkers(
      &workers);
  num_remote_workers_ = workers.size() - 1;
  VLOG(2) << "rmda_mgr on local worker: " << local_worker_;
  for (size_t i = 0; i < workers.size(); i++) {
    if (local_worker_.compare(workers[i]) != 0) {
      channel_table_.insert(
          {workers[i],
           new RdmaChannel(rdma_adapter_, local_worker_, workers[i])});
    }
  }
}

// Setup Rdma channels between peers.
// This is done at the beginning of the server setup.

void RdmaMgr::SetupChannels() {
  for (const auto& p : channel_table_) {
    string worker_name = p.first;
    RDMA_LOG(2) << "Connecting to remote node " << worker_name;
    RdmaChannel* rc = p.second;
    GetRemoteAddressRequest req;
    GetRemoteAddressResponse resp;
    // get the channel cache
    SharedGrpcChannelPtr client_channel =
        channel_cache_->FindWorkerChannel(worker_name);
    GrpcVerbsClient* client = new GrpcVerbsClient(client_channel);
    CHECK(client != nullptr) << "No worker known as " << worker_name;

    // setting up request
    req.set_host_name(local_worker_);
    Channel* channel_info = req.mutable_channel();
    channel_info->set_lid(rc->self_.lid);
    channel_info->set_qpn(rc->self_.qpn);
    channel_info->set_psn(rc->self_.psn);
    channel_info->set_snp(rc->self_.snp);
    channel_info->set_iid(rc->self_.iid);
    for (int i = 0; i < RdmaChannel::kNumMessageBuffers; i++) {
      MemoryRegion* mr = req.add_mr();
      mr->set_remote_addr(
          reinterpret_cast<uint64_t>(rc->message_buffers_[i]->buffer_));
      mr->set_rkey(rc->message_buffers_[i]->self_->rkey);
    }
    // synchronous call
    Status s;
    int attempts = 0;
    static const int max_num_attempts = 5;
    do {
      s = client->GetRemoteAddress(&req, &resp);
      // save obtained remote addresses
      // connect to the remote channel
      if (s.ok()) {
        CHECK(worker_name.compare(resp.host_name()) == 0);
        RdmaAddress ra;
        ra.lid = resp.channel().lid();
        ra.qpn = resp.channel().qpn();
        ra.psn = resp.channel().psn();
        ra.snp = resp.channel().snp();
        ra.iid = resp.channel().iid();
        rc->SetRemoteAddress(ra, false);
        rc->Connect();
        int i = 0;
        int idx[] = {1, 0};
        for (const auto& mr : resp.mr()) {
          // the connections are crossed, i.e.
          // local tx_message_buffer <---> remote rx_message_buffer_
          // local rx_message_buffer <---> remote tx_message_buffer_
          // hence idx[] = {1, 0}.
          RdmaMessageBuffer* rb = rc->message_buffers_[idx[i]];
          RemoteMR rmr;
          rmr.remote_addr = mr.remote_addr();
          rmr.rkey = mr.rkey();
          rb->SetRemoteMR(rmr, false);
          i++;
        }
        CHECK(i == RdmaChannel::kNumMessageBuffers);
      } else {
        LOG(ERROR) << "Connecting to " << worker_name << ": Got "
                   << s.error_message() << ". Retrying (" << (attempts + 1)
                   << "/" << max_num_attempts << ")...";
        if (++attempts == max_num_attempts) {
          break;
        }
        worker_env_->env->SleepForMicroseconds(2000000);
      }
    } while (!s.ok());
    RDMA_LOG(0) << "Connected to remote node " << worker_name;
    delete client;
  }
}

// Check connectivity by pinging every channel
bool RdmaMgr::ConnectivityCheck() {
  int i, rcnt = 0, scnt = 0;

  for (const auto& p : channel_table_) {
    string worker_name = p.first;
    RdmaChannel* rc = p.second;

    VLOG(2) << "Ping to " << worker_name;
    CHECK(rc->PingPostSend() == 0) << "Couldn't post send  to " << worker_name
                                   << " with error: " << std::strerror(errno);
    for (i = 0; i < rc->adapter_->params_.queue_depth - 1; i++) {
      rc->Recv();
    }
  }

  while (rcnt < num_remote_workers_ || scnt < num_remote_workers_) {
    int ne;
    do {
      ne = ibv_poll_cq(rdma_adapter_->cq_, 2 * num_remote_workers_,
                       rdma_adapter_->wc_);
      CHECK(ne >= 0) << "poll CQ failed " << ne << "with error"
                     << std::strerror(errno);
    } while (ne < 1);

    for (i = 0; i < ne; ++i) {
      ibv_wc_status s = rdma_adapter_->wc_[i].status;
      // recv complete
      if ((int)rdma_adapter_->wc_[i].wr_id == RdmaChannel::kPingRecvWrid) {
        CHECK(s == IBV_WC_SUCCESS)
            << ": " << ibv_wc_status_str(rdma_adapter_->wc_[i].status) << "("
            << rdma_adapter_->wc_[i].status << ") for PING_RECV_WRID";
        ++rcnt;
        // send complete
      } else {
        RdmaChannel* rc =
            reinterpret_cast<RdmaChannel*>(rdma_adapter_->wc_[i].wr_id);
        CHECK(s == IBV_WC_SUCCESS)
            << ": " << ibv_wc_status_str(rdma_adapter_->wc_[i].status) << "("
            << rdma_adapter_->wc_[i].status << ") to " << rc->remote_name_;
        ++scnt;
      }
    }  // for
  }    // while
  CHECK(rcnt == scnt) << "Connectivity check failed!";
  rdma_adapter_->StartPolling();
  return (num_remote_workers_ == rcnt) && (num_remote_workers_ == scnt);
}

RdmaMgr::~RdmaMgr() {
  for (const auto& p : channel_table_) delete p.second;
  channel_table_.clear();
  delete rdma_adapter_;
}

// Find a channel via the given name.
// Args:
//   name: peer name, e.g. worker1
// Returns
//   channel object that is connected to the named peer.
RdmaChannel* RdmaMgr::FindChannel(const string& name) {
  ChannelTable::iterator iter = channel_table_.find(name);
  CHECK(iter != channel_table_.end());
  return iter->second;
}

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

void MRDeleter(ibv_mr* mr) {
  if (mr) {
    ibv_dereg_mr(mr);
  }
}

// TODO(byronyi): remove this class duplicated from the one in
// common/runtime/gpu/pool_allocator.h when it is available in common_runtime
class BasicCPUAllocator : public SubAllocator {
 public:
  ~BasicCPUAllocator() override {}

  void* Alloc(size_t alignment, size_t num_bytes) override {
    return port::AlignedMalloc(num_bytes, alignment);
  }
  void Free(void* ptr, size_t) override { port::AlignedFree(ptr); }
};

// TODO(byronyi): remove this class and its registration when the default
// cpu_allocator() returns visitable allocator
class BFCRdmaAllocator : public BFCAllocator {
 public:
  BFCRdmaAllocator()
      : BFCAllocator(new BasicCPUAllocator(), 1LL << 36, true, "cpu_rdma_bfc") {
  }
};

REGISTER_MEM_ALLOCATOR("BFCRdmaAllocator", 101, BFCRdmaAllocator);

void RdmaMgr::InitAllocators() {
  RdmaMemoryMgr::Singleton().pd_ = rdma_adapter_->pd_;

  Allocator* allocators[] = {
#if GOOGLE_CUDA
    ProcessState::singleton()->GetCUDAHostAllocator(0),
    ProcessState::singleton()->GetCPUAllocator(0),
#endif  // GOOGLE_CUDA
    cpu_allocator(),
  };

  using namespace std::placeholders;

  std::set<Allocator*> instrumented_;

  // Host memory allocators
  for (Allocator* allocator : allocators) {
    VisitableAllocator::Visitor alloc_visitor =
        std::bind(&RdmaMemoryMgr::InsertMemoryRegion,
                  &RdmaMemoryMgr::Singleton(), _1, _2, allocator->Name());
    VisitableAllocator::Visitor free_visitor = std::bind(
        &RdmaMemoryMgr::EvictMemoryRegion, &RdmaMemoryMgr::Singleton(), _1, _2);

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
  if (IsGDRAvailable()) {
    // Note we don't free allocated GPU memory so there is no free visitor
    int32_t bus_id = TryToReadNumaNode(rdma_adapter_->context_->device) + 1;

    char buf[8];
    sprintf(buf, "gpu");
    VisitableAllocator::Visitor cuda_alloc_visitor =
        std::bind(&RdmaMemoryMgr::InsertMemoryRegion,
                  &RdmaMemoryMgr::Singleton(), _1, _2, std::string(buf));

    ProcessState::singleton()->AddGPUAllocVisitor(bus_id, cuda_alloc_visitor);
    LOG(INFO) << "Instrumenting GPU allocator with bus_id " << bus_id;
  }
#endif  // GOOGLE_CUDA
}

}  // end namespace tensorflow

#endif
