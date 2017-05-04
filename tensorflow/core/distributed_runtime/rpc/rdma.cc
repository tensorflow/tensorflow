#include "tensorflow/core/distributed_runtime/rpc/rdma.h"

#include <atomic>
#include <map>
#include <memory>
#include <functional>
#include <thread>

#include <sys/fcntl.h>

#include <rdma/rdma_cma.h>
#include <rdma/rdma_verbs.h>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/protobuf/worker.pb.h"

namespace tensorflow {

namespace {

void MRDeleter(ibv_mr* mr) {
  if (mr) ibv_dereg_mr(mr);
}

void CQDeleter(ibv_cq* cq) {
  if (cq) ibv_destroy_cq(cq);
}

void EndpointDeleter(rdma_cm_id* id) {
  if (id) {
    rdma_destroy_id(id);
  }
}

Status create_queue_pair(rdma_cm_id* id, ibv_pd* pd, ibv_cq* cq) {
  ibv_qp_init_attr init_attr = {};
  init_attr.cap.max_recv_wr = 32;
  init_attr.cap.max_send_wr = 32;
  init_attr.cap.max_recv_sge = 1;
  init_attr.cap.max_send_sge = 1;
  init_attr.qp_type = IBV_QPT_RC;
  init_attr.sq_sig_all = true;
  init_attr.send_cq = cq;
  init_attr.recv_cq = cq;
  if (rdma_create_qp(id, pd, &init_attr)) {
    return errors::Unavailable("Failed to create QP");
  }
  return Status::OK();
}

using MRPtr = std::unique_ptr<ibv_mr, decltype(&MRDeleter)>;

using CQPtr = std::unique_ptr<ibv_cq, decltype(&CQDeleter)>;

using RdmaEndpointPtr = std::unique_ptr<rdma_cm_id, decltype(&EndpointDeleter)>;

class RdmaReadClient : public RdmaClient {
 public:
  virtual Status ReadTensorViaDMA(const TensorBuffer* buffer,
      const ::google::protobuf::Any& transport_options) override {

    RemoteMemoryRegion remote_mr;
    if (!transport_options.UnpackTo(&remote_mr)) {
      return errors::NotFound("No RDMA transport options found");
    }

    if (buffer->size() != remote_mr.length()) {
      return errors::InvalidArgument("Buffer size mismatch");
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

    uint64_t start = Env::Default()->NowMicros();
    ibv_mr* mr = rdma_reg_msgs(id, buffer->data(), buffer->size());
    if (mr == nullptr) {
      return errors::Unavailable("Cannot register memory region");
    }
    uint64_t end = Env::Default()->NowMicros();
    auto managed_mr = MRPtr(mr, &MRDeleter);
    VLOG(2) << "Pinning RX TensorBuffer@" << buffer->data()
            << " with size " << buffer->size() << " bytes"
            << " and rkey " << reinterpret_cast<void*>(mr->rkey)
            << " took " << (end - start) << " micros";

    start = Env::Default()->NowMicros();

    ibv_wc wc;
    int ret;
    if (rdma_post_read(id, nullptr, buffer->data(), buffer->size(), mr,
                       0, remote_mr.addr(), remote_mr.rkey())) {
      perror("rdma_post_read");
      return errors::Unavailable("rdma_post_read failed");
    } else {
      wc = {};
      while ((ret = ibv_poll_cq(id->send_cq, 1, &wc) == 0)) {}
      if (ret < 0 || wc.status) {
        return errors::Unavailable(ibv_wc_status_str(wc.status));
      }
    }

    end = Env::Default()->NowMicros();

    uint64_t ack_start = Env::Default()->NowMicros();

    ibv_send_wr wr = {};
    wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
    wr.imm_data = htonl(remote_mr.rkey());
    ibv_send_wr* bad_wr;
    if (ibv_post_send(id->qp, &wr, &bad_wr)) {
      return errors::Unavailable("ibv_post_send failed");
    }

    wc = {};
    while ((ret = ibv_poll_cq(id->send_cq, 1, &wc) == 0));
    if (ret < 0 || wc.status) {
      return errors::Unavailable(ibv_wc_status_str(wc.status));
    }

    uint64_t ack_end = Env::Default()->NowMicros();
    VLOG(2) << "RDMA into TensorBuffer@" << buffer->data()
            << " with rkey " << reinterpret_cast<void*>(mr->rkey)
            << " and size " << buffer->size() << " bytes"
            << " took " << (end - start) << " micros"
            << " and ACK took " << (ack_end - ack_start) << " micros";

    return Status::OK();
}

 private:
  Status CreateEndpoint(const string& host, const string& port,
                        RdmaEndpointPtr& endpoint) {
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
    init_attr.cap.max_recv_wr = 32;
    init_attr.cap.max_send_wr = 32;
    init_attr.cap.max_recv_sge = 1;
    init_attr.cap.max_send_sge = 1;
    init_attr.qp_type = IBV_QPT_RC;
    init_attr.sq_sig_all = true;

    rdma_cm_id* id;
    if (rdma_create_ep(&id, addrinfo, nullptr, &init_attr)) {
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
  RdmaReadServer(Env* env, const string& host, const string& port)
    : env_(env), host_(host), port_(port),
      listening_(nullptr, EndpointDeleter) {}

  virtual ~RdmaReadServer() {
    Stop();
  }

  virtual Status Init() override {

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

    // Create listening endpoint
    rdma_cm_id* id;
    if (rdma_create_ep(&id, addrinfo, nullptr, nullptr)) {
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

    mutex_lock l(mu_);

    // Allocate per device CQ for polling
    int num_device;
    ibv_context** devices = rdma_get_devices(&num_device);
    if (!devices || num_device <= 0) {
      return errors::Unavailable("Cannot find RDMA capable devices");
    }
    for (int i = 0; i < num_device; i++) {
      ibv_cq* cq = ibv_create_cq(devices[i], 64, this, nullptr, 0);
      if (cq) {
        cqs_.insert(std::make_pair(devices[i], CQPtr(cq, CQDeleter)));
      } else {
        LOG(WARNING) << "Cannot create ibv_cq for device " << devices[i]->device->name;
      }
    }
    rdma_free_devices(devices);

    // Manage incoming connections
    cm_thread_.reset(env_->StartThread(ThreadOptions(), "TF_rdma_cm",
                                       [this] () {
      while (!stopped_) {
        // Accept incoming connections
        rdma_cm_id* id = nullptr;
        if (!rdma_get_request(listening_.get(), &id)) {

          auto iter = cqs_.find(id->verbs);
          if (iter == std::end(cqs_)) {
            LOG(WARNING) << "Cannot find device " << id->verbs->device->name;
            continue;
          }
          const CQPtr& cq = iter->second;

          Status s = create_queue_pair(id, listening_->pd, cq.get());
          if (!s.ok()) {
            LOG(WARNING) << s.error_message();
            EndpointDeleter(id);
            continue;
          }

          if (!rdma_accept(id, nullptr)) {
            if (rdma_post_recvv(id, id, nullptr, 0)) {
              LOG(ERROR) << "rdma_post_recvv failed";
              EndpointDeleter(id);
            }
          }
        }
      }
    }));

    return Status::OK();
  }

  virtual void Run() override {

    stopped_ = false;

    // Polling work completions
    while (!stopped_) {

      ibv_wc wc[32];
      for (const auto& cq : cqs_) {
        int ret = ibv_poll_cq(cq.second.get(), 32, wc);
        for (int i = 0; i < ret; i++) {
          Status s;
          if (wc[i].opcode & IBV_WC_RECV_RDMA_WITH_IMM) {
            s = (wc[i].status == 0 && wc[i].vendor_err == 0) ? Status::OK()
                  : errors::Unavailable(ibv_wc_status_str(wc[i].status));
          } else {
            s = errors::Unknown("Received unknown operation ", wc[i].opcode);
          }
          uint32_t key = ntohl(wc[i].imm_data);
          decltype(buffers_)::iterator iter;
          {
            mutex_lock l(mu_);
            iter = buffers_.find(key);
            if (iter != std::end(buffers_) && iter->second != nullptr) {
              auto buffer = iter->second;
              if (buffer->RefCountIsOne()) {
                auto p = std::make_pair(buffer->data(), buffer->size());
                auto mr_iter = registered_mr_.find(p);
                if (mr_iter != std::end(registered_mr_)) {
                  CHECK(mr_iter->second->rkey == key)
                      << "Memory region rkey mismatch";
                  registered_mr_.erase(mr_iter);
                } else {
                  LOG(ERROR) << "Cannot find memory region for remote key "
                             << reinterpret_cast<void*>(key);
                }
              }
              buffers_.erase(iter);
              buffer->Unref();
            } else {
              LOG(ERROR) << "Cannot find buffer for remote key "
                         << reinterpret_cast<void*>(key);
            }
          }
          rdma_cm_id* id = reinterpret_cast<rdma_cm_id*>(wc[i].wr_id);
          if (rdma_post_recvv(id, id, nullptr, 0)) {
            LOG(ERROR) << "rdma_post_recvv failed";
            EndpointDeleter(id);
          }
        }
      }
    }
  }

  virtual Status RegisterTensorDMA(const TensorBuffer* buffer,
      ::google::protobuf::Any* mutable_transport_options) override {

    ibv_mr* mr = nullptr;
    {
      decltype(registered_mr_)::iterator iter;
      auto key = std::make_pair(buffer->data(), buffer->size());
      bool success;
      {
        mutex_lock l(mu_);
        std::tie(iter, success) = registered_mr_.insert(
            std::make_pair(std::move(key), MRPtr(nullptr, MRDeleter)));
      }
      if (success || iter->second == nullptr) {
        uint64_t start = Env::Default()->NowMicros();
        mr = rdma_reg_read(listening_.get(),
                           buffer->data(), buffer->size());
        if (mr == nullptr) {
          return errors::Unavailable("Cannot register memory region");
        }
        uint64_t end = Env::Default()->NowMicros();
        iter->second = MRPtr(mr, MRDeleter);
        VLOG(2) << "Pinning TX TensorBuffer@" << buffer->data()
                << " with size " << buffer->size() << " bytes"
                << " and rkey " << reinterpret_cast<void*>(mr->rkey)
                << " took " << (end - start) << " micros";
      } else {
        mr = iter->second.get();
        VLOG(2) << "Reusing TX TensorBuffer@" << buffer->data()
                << " with size " << buffer->size() << " bytes"
                << " and rkey " << reinterpret_cast<void*>(mr->rkey);
      }
    }

    buffer->Ref();
    {
      mutex_lock l(mu_);
      buffers_[mr->rkey] = const_cast<TensorBuffer*>(buffer);
    }

    RemoteMemoryRegion remote_mr;
    remote_mr.set_host(host_);
    remote_mr.set_port(port_);
    remote_mr.set_addr(reinterpret_cast<uint64_t>(mr->addr));
    remote_mr.set_rkey(mr->rkey);
    remote_mr.set_length(mr->length);
    mutable_transport_options->PackFrom(remote_mr);
    return Status::OK();
  }

  virtual void Stop() override {
    stopped_ = true;
    mutex_lock l(mu_);
    cm_thread_.reset();
  }

 private:
  Env* env_;
  string host_;
  string port_;
  RdmaEndpointPtr listening_;
  std::map<ibv_context*, CQPtr> cqs_;

  mutex mu_;
  std::unique_ptr<Thread> cm_thread_ GUARDED_BY(mu_);
  std::map<uint32_t, TensorBuffer*> buffers_ GUARDED_BY(mu_);

  std::atomic<bool> stopped_;
  std::map<std::pair<void*, size_t>, MRPtr> registered_mr_;
};

}  // namespace

RdmaServer* NewRdmaServer(Env* env, const string& host, const string& port) {
  return new RdmaReadServer(env, host, port);
}

RdmaClient* NewRdmaClient() {
  return new RdmaReadClient;
}

}  // namespace tensorflow
