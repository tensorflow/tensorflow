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

#include <fcntl.h>
#include <cstdlib>

#include "tensorflow/contrib/verbs/rdma.h"
#include "tensorflow/contrib/verbs/verbs_service.pb.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/process_util.h"
#if GOOGLE_CUDA
#include "tensorflow/core/common_runtime/gpu/gpu_process_state.h"
#include "tensorflow/core/common_runtime/gpu/gpu_util.h"
#endif
#include "tensorflow/core/distributed_runtime/rendezvous_mgr_interface.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/distributed_runtime/session_mgr.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/random/random.h"

namespace tensorflow {

#define RoCE_V2 "RoCE v2"

namespace {

// convenience function for printing message
string MessageTypeToString(RdmaMessageType rmt) {
  switch (rmt) {
    case RDMA_MESSAGE_META_DATA_UPDATE:
      return "RDMA_MESSAGE_META_DATA_UPDATE";
      break;
    case RDMA_MESSAGE_TENSOR_RE_REQUEST:
      return "RDMA_MESSAGE_TENSOR_RE_REQUEST";
      break;
    case RDMA_MESSAGE_TENSOR_REQUEST:
      return "RDMA_MESSAGE_TENSOR_REQUEST";
      break;
    default:
      return "UNKNOWN MESSAGE";
  }
}
}  // namespace

// Function to get environment variable
// Args:
//    var_name - the name of the environmental variable
// Returns:
//    string with it's value or empty string if not set
string get_env_var(char const* var_name) {
  char const* var_temp = getenv(var_name);

  return (var_temp == NULL) ? string() : string(var_temp);
}

// Function to open device
// Args:
//   ibv_dev device to open
// Returns:
//   context of the opened device
ibv_context* open_device(ibv_device* ibv_dev) {
  ibv_context* context = ibv_open_device(ibv_dev);

  CHECK(context) << "Open context failed for " << ibv_get_device_name(ibv_dev);
  return context;
}

// Function to count the number of active ports for device
// Args:
//   device - to check active ports
// Returns:
//   number of active ports of the given device
int get_dev_active_port_count(ibv_device* device) {
  ibv_device_attr device_att;
  ibv_port_attr port_attr;
  ibv_context* context = NULL;
  int rc, port_index, active_ports = 0;

  context = ibv_open_device(device);
  CHECK(context) << "Open context failed for " << ibv_get_device_name(device);
  rc = ibv_query_device(context, &device_att);
  CHECK(!rc) << "Failed to query the device";

  for (port_index = 1; port_index <= device_att.phys_port_cnt; port_index++) {
    rc = ibv_query_port(context, port_index, &port_attr);
    CHECK(!rc) << "Failed to query the port" << port_index;
    if (port_attr.state == IBV_PORT_ACTIVE) {
      active_ports++;
    }
  }
  ibv_close_device(context);
  return active_ports;
}

// Function to set device. If RDMA_DEVICE not set, search for device with active
// port.
// Fails if more than one device with active port was found.
// Returns:
//   device to use
ibv_device* set_device() {
  ibv_device** dev_list;
  int dev_num, device_index, device_to_open = 0;
  int num_devs_with_active_port = 0;
  string env_p_rdma_device, str_port_num;

  dev_list = ibv_get_device_list(&dev_num);
  CHECK(dev_list) << "No InfiniBand device found";

  env_p_rdma_device = get_env_var("RDMA_DEVICE");
  if (!env_p_rdma_device.empty()) {
    for (device_index = 0; device_index < dev_num; device_index++) {
      if (!env_p_rdma_device.compare(
              ibv_get_device_name(dev_list[device_index]))) {
        CHECK(get_dev_active_port_count(dev_list[device_index]) != 0)
            << "Device " << ibv_get_device_name(dev_list[device_index])
            << " has no active ports";
        return dev_list[device_index];
      }
    }
    // check validity of input device
    CHECK(false) << "The device " << env_p_rdma_device << " wasn't found";
  } else {
    // set default device
    str_port_num = get_env_var("RDMA_DEVICE_PORT");
    CHECK(str_port_num.empty())
        << "RDMA_DEVICE should be provided if RDMA_DEVICE_PORT is set by user";
    for (device_index = 0; device_index < dev_num; device_index++) {
      // get port_num
      if (get_dev_active_port_count(dev_list[device_index]) > 0) {
        num_devs_with_active_port++;
        CHECK(num_devs_with_active_port <= 1) << ". More than one device with "
                                                 "active port in the system. "
                                                 "Please enter RDMA_DEVICE";
        // found device with at least 1 active port
        device_to_open = device_index;
      }
    }
    CHECK(num_devs_with_active_port > 0)
        << "There is no active port in the system";
    return dev_list[device_to_open];
  }
  CHECK(false) << "No device was set!";
  return NULL;  // never happens
}

// Function to set port for device.
// If RDMA_DEVICE_PORT not set, first active port of the device will be set.
// Args:
//   context of the device
// Returns:
//   port to use
uint8_t set_port(ibv_context* context) {
  uint8_t port_num = 0;  // 0 is illegal port number
  string str_port_num;
  ibv_device_attr device_att;
  ibv_port_attr port_attr;
  int rc, port_index;

  rc = ibv_query_device(context, &device_att);
  CHECK(!rc) << "Failed to query the device\n";

  str_port_num = get_env_var("RDMA_DEVICE_PORT");
  // user defined port
  if (!str_port_num.empty()) {
    port_num = stoi(str_port_num);
    CHECK(port_num > 0) << "RDMA_DEVICE_PORT should be positive";
    CHECK(port_num <= device_att.phys_port_cnt) << "RDMA_DEVICE_PORT should be "
                                                   "less or equal to amount of "
                                                   "available ports";
    rc = ibv_query_port(context, port_num, &port_attr);
    CHECK(!rc) << "Failed to query the port" << port_num;
    // check if port id active
    CHECK(port_attr.state == IBV_PORT_ACTIVE)
        << "Selected RDMA_DEVICE_PORT is not active";
  } else {  // set default port
    for (port_index = 1; port_index <= device_att.phys_port_cnt; port_index++) {
      rc = ibv_query_port(context, port_index, &port_attr);
      CHECK(!rc) << "Failed to query the port" << port_index;
      if (port_attr.state == IBV_PORT_ACTIVE) {
        port_num = port_index;
        break;
      }
    }
    CHECK_GT(port_num, 0) << "No active ports";
  }
  return port_num;
}

// Function read from sysfs file
// Args:
//   dir - directory
//   file - file
//   buff - buffer for the result
//   size - buffer size
// Returns:
//   number of bytes were read or -1 if failed
int read_sysfs_file(const char* dir, const char* file, char* buf, size_t size) {
  char* path;
  int fd;
  int len;

  if (asprintf(&path, "%s/%s", dir, file) < 0) return -1;

  fd = open(path, O_RDONLY);
  if (fd < 0) {
    free(path);
    return -1;
  }

  len = read(fd, buf, size);

  close(fd);
  free(path);

  if (len > 0 && buf[len - 1] == '\n') buf[--len] = '\0';

  return len;
}

// Function to check if GID index support RoCE V2
// Args:
//   context - device context
//   port_num - port number
//   index -  GID index
// Returns:
//   if GID supports RoCE V2 - true, otherwise - false.
bool is_gid_type_roce_v2(ibv_context* context, uint8_t port_num,
                         uint8_t index) {
  char name[32];
  char buff[41];

  snprintf(name, sizeof(name), "ports/%d/gid_attrs/types/%d", port_num, index);
  if (read_sysfs_file(context->device->ibdev_path, name, buff, sizeof(buff)) <=
      0) {
    return false;
  }
  return !strcmp(buff, RoCE_V2);
}

// Function to set GID index.
// If the port link is IB, no GID index should be selected.
// If Ethernet but RDMA_GID_INDEX not set gid index that supports
//   RoCE V2 will be chosen(fails if more than one IP is configured)
// Args:
//   context - device context
//   port_num - port number
// Returns:
//   GID index to use
uint8_t set_gid(uint8_t port_num, ibv_context* context) {
  ibv_port_attr port_attr;
  string gid_str;
  int rc, i, gids_num = 0, v2_ip_num = 0;
  union ibv_gid gid;
  uint8_t gid_index = 0;

  rc = ibv_query_port(context, port_num, &port_attr);
  CHECK(!rc) << "Failed to query the port" << port_num;

  for (i = 0; i < port_attr.gid_tbl_len; i++) {
    rc = ibv_query_gid(context, port_num, i, &gid);
    CHECK(!rc) << "Failed to query gid to port " << (int)port_num << " index "
               << i;
    if (gid.global.interface_id) {
      gids_num++;
      if (gid.global.subnet_prefix == 0 &&
          is_gid_type_roce_v2(context, port_num, i)) {
        if (v2_ip_num == 0) {
          // can be overwritten by RDMA_GID_INDEX later
          gid_index = i;
        }
        v2_ip_num++;
      }
    }
  }
  switch (port_attr.link_layer) {
    case (IBV_LINK_LAYER_ETHERNET):
      gid_str = get_env_var("RDMA_GID_INDEX");
      if (!gid_str.empty()) {
        gid_index = stoi(gid_str);
        CHECK(gid_index < gids_num)
            << "RDMA_GID_INDEX should be less than GIDs amount" << gids_num;
      } else {
        CHECK(v2_ip_num <= 1)
            << "More than one IP is available, please specify GID_INDEX";
      }
      break;
    case (IBV_LINK_LAYER_INFINIBAND):  // no need in GID index
      break;
    default:
      LOG(INFO) << "Unknown port link layer. Currently supporting Ethernet and "
                   "InfiniBand only. ";
  }
  if (!is_gid_type_roce_v2(context, port_num, gid_index)) {
    LOG(INFO) << "RoCE v2 is not configured for GID_INDEX " << (int)gid_index;
  }
  return gid_index;
}

// set the default or environment value to the configuration parameter.
// Args:
//   default_val- the default value for this parameter
//   env_param- the environment parameter's name
// Returns:
//   32-bit value
uint32_t set_param(uint32_t default_val, const char* env_param) {
  uint32_t val = default_val;
  string val_s;

  val_s = get_env_var(env_param);

  if (!val_s.empty()) {
    val = stoi(val_s);
  }
  return val;
}

enum ibv_mtu set_mtu(uint8_t port_num, ibv_context* context) {
  ibv_port_attr port_attr;
  enum ibv_mtu mtu = IBV_MTU_512;
  string mtu_s;
  int rc, mtu_i;

  rc = ibv_query_port(context, port_num, &port_attr);
  CHECK(!rc) << "Failed to query the port" << port_num;

  mtu_s = get_env_var("RDMA_MTU");

  if (!mtu_s.empty()) {
    mtu_i = stoi(mtu_s);
    switch (mtu_i) {
      case 256:
        mtu = IBV_MTU_256;
        break;
      case 512:
        mtu = IBV_MTU_512;
        break;
      case 1024:
        mtu = IBV_MTU_1024;
        break;
      case 2048:
        mtu = IBV_MTU_2048;
        break;
      case 4096:
        mtu = IBV_MTU_4096;
        break;
      default:
        CHECK(0) << "Error: MTU input value must be one of the following: 256, "
                    "512, 1024, 2048, 4096. MTU "
                 << mtu << " is invalid\n";
        break;
    }
    CHECK(mtu < port_attr.active_mtu)
        << "MTU configuration for the QPs is larger than active MTU";
  } else {
    mtu = port_attr.active_mtu;
  }
  return mtu;
}

RdmaParams params_init(ibv_context* context) {
  RdmaParams params;

  params.port_num = set_port(context);
  params.sgid_index = set_gid(params.port_num, context);
  params.pkey_index = (uint8_t)set_param(PKEY_DEFAULT, "RDMA_PKEY");
  params.queue_depth = set_param(QUEUE_DEPTH_DEFAULT, "RDMA_QUEUE_DEPTH");
  params.timeout = (uint8_t)set_param(TIMEOUT_DEFAULT, "RDMA_TIMEOUT");
  params.retry_cnt = (uint8_t)set_param(RETRY_CNT_DEFAULT, "RDMA_RETRY_CNT");
  params.sl = (uint8_t)set_param(SL_DEFAULT, "RDMA_SL");
  CHECK(params.sl <= 7) << "SL value is " << (int)params.sl
                        << ". Valid values are 0-7.";
  params.mtu = set_mtu(params.port_num, context);
  params.traffic_class = set_param(TRAFFIC_CLASS, "RDMA_TRAFFIC_CLASS");
  return params;
}

ibv_pd* alloc_protection_domain(ibv_context* context) {
  ibv_pd* pd = ibv_alloc_pd(context);
  CHECK(pd) << "Failed to allocate protection domain";
  return pd;
}

RdmaAdapter::RdmaAdapter(const WorkerEnv* worker_env)
    : context_(open_device(set_device())),
      params_(params_init(context_)),
      pd_(alloc_protection_domain(context_)),
      worker_env_(worker_env) {
  event_channel_ = ibv_create_comp_channel(context_);
  CHECK(event_channel_) << "Failed to create completion channel";
  cq_ = ibv_create_cq(context_, MAX_CONCURRENT_WRITES * 2, NULL, event_channel_,
                      0);
  CHECK(cq_) << "Failed to create completion queue";
  CHECK(!ibv_req_notify_cq(cq_, 0)) << "Failed to request CQ notification";
}

RdmaAdapter::~RdmaAdapter() {
  polling_thread_.reset();
  CHECK(!ibv_destroy_cq(cq_)) << "Failed to destroy CQ";
  CHECK(!ibv_destroy_comp_channel(event_channel_))
      << "Failed to destroy channel";
  CHECK(!ibv_dealloc_pd(pd_)) << "Failed to deallocate PD";
  CHECK(!ibv_close_device(context_)) << "Failed to release context";
}

void RdmaAdapter::StartPolling() {
  polling_thread_.reset(Env::Default()->StartThread(
      ThreadOptions(), "RdmaAdapterCQThread", [this] { Process_CQ(); }));
  VLOG(2) << "Start RdmaAdapter: " << name();
}

string RdmaAdapter::name() const { return string(context_->device->name); }

// Function to process incoming messages
// There are two types of messages:
// 1. IBV_WC_RECV_RDMA_WITH_IMM (receive)
// 2. IBV_WC_RDMA_WRITE (send))
void RdmaAdapter::Process_CQ() {
  while (true) {
    ibv_cq* cq;
    void* cq_context;
    CHECK(!ibv_get_cq_event(event_channel_, &cq, &cq_context));
    CHECK(cq == cq_);
    ibv_ack_cq_events(cq, 1);
    CHECK(!ibv_req_notify_cq(cq_, 0));

    int ne =
        ibv_poll_cq(cq_, MAX_CONCURRENT_WRITES * 2, static_cast<ibv_wc*>(wc_));
    CHECK_GE(ne, 0);
    for (int i = 0; i < ne; ++i) {
      CHECK(wc_[i].status == IBV_WC_SUCCESS)
          << "Failed status \n"
          << ibv_wc_status_str(wc_[i].status) << " " << wc_[i].status << " "
          << static_cast<int>(wc_[i].wr_id) << " " << wc_[i].vendor_err;
      if (wc_[i].opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
        RdmaChannel* rc = reinterpret_cast<RdmaChannel*>(wc_[i].wr_id);
        // put back a recv wr.
        rc->Recv();
        // imm_data is the index of RX buffer in the buffer table.
        uint32_t imm_data = wc_[i].imm_data;
        RdmaMessageBuffer* rb;
        RdmaMessage rm;

        if (imm_data == RDMA_IMM_DATA_ACK) {
          // receive an ack to a message
          rb = rc->tx_message_buffer_;
          rb->SetBufferStatus(remote, idle);
          rb->SendNextItem();
          continue;
        }

        if (imm_data <= RDMA_IMM_MAX_REQUEST_ID) {
          // receive a tensor RDMA write
          uint32_t request_index = imm_data;
          RdmaTensorRequest* request = rc->GetTensorRequest(request_index);
          request->RecvTensorContent();
          continue;
        }

        // receive a control message
        rb = rc->rx_message_buffer_;
        RdmaMessage::ParseMessage(rm, rb->buffer_);
        RdmaMessageBuffer::SendAck(rc);
        RDMA_LOG(1) << "Step 0x" << std::hex << rm.step_id_ << std::dec
                    << ": Received " << MessageTypeToString(rm.type_) << " "
                    << "#" << rm.request_index_ << ": " << rm.name_;

        if (rm.type_ == RDMA_MESSAGE_TENSOR_REQUEST) {
          RdmaTensorResponse* response = rc->AddTensorResponse(rm);
          response->Start();
        } else if (rm.type_ == RDMA_MESSAGE_META_DATA_UPDATE) {
          RdmaTensorRequest* request = rc->GetTensorRequest(rm.request_index_);
          request->RecvTensorMetaData(rm.data_type_, rm.tensor_shape_,
                                      rm.is_dead_, rm.tensor_bytes_);
#ifdef RDMA_DATA_VALIDATION
          request->RecvTensorChecksum(rm.checksum_);
#endif
        } else if (rm.type_ == RDMA_MESSAGE_TENSOR_RE_REQUEST) {
          RdmaTensorResponse* response = rc->UpdateTensorResponse(rm);
          response->Resume();
        } else if (rm.type_ == RDMA_MESSAGE_ERROR_STATUS) {
          RdmaTensorRequest* request = rc->GetTensorRequest(rm.request_index_);
          request->RecvErrorStatus(rm.status_);
        }
      } else if (wc_[i].opcode == IBV_WC_RDMA_WRITE) {
        RdmaWriteID* wr_id = reinterpret_cast<RdmaWriteID*>(wc_[i].wr_id);
        RDMA_LOG(2) << "Write complete of type " << wr_id->write_type;
        switch (wr_id->write_type) {
          case RDMA_WRITE_ID_ACK:
            break;
          case RDMA_WRITE_ID_MESSAGE: {
            RdmaMessageBuffer* rb =
                reinterpret_cast<RdmaMessageBuffer*>(wr_id->write_context);
            rb->SetBufferStatus(local, idle);
            rb->SendNextItem();
            break;
          }
          case RDMA_WRITE_ID_TENSOR_WRITE: {
            RdmaTensorResponse* response =
                reinterpret_cast<RdmaTensorResponse*>(wr_id->write_context);
            response->Destroy();
          }
        }
        delete wr_id;
      }
    }
  }
}

int RdmaChannel::PingPostRecv() {
  struct ibv_recv_wr wr, *bad_wr;
  memset(&wr, 0, sizeof(wr));
  wr.sg_list = &ping_sge_list_;
  wr.num_sge = 1;
  wr.wr_id = kPingRecvWrid;

  return ibv_post_recv(qp_, &wr, &bad_wr);
}

int RdmaChannel::PingPostSend() {
  struct ibv_send_wr wr, *bad_wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = (uint64_t)this;
  wr.sg_list = &ping_sge_list_;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_SEND;
  wr.send_flags = IBV_SEND_SIGNALED;

  return ibv_post_send(qp_, &wr, &bad_wr);
}

RdmaChannel::RdmaChannel(const RdmaAdapter* adapter, const string local_name,
                         const string remote_name)
    : adapter_(adapter),
      local_name_(local_name),
      remote_name_(remote_name),
      request_serial_(0) {
  struct ibv_sge list;

  mr_ = ibv_reg_mr(adapter_->pd_, ping_buff_, kPingBuffSize,
                   IBV_ACCESS_LOCAL_WRITE);
  CHECK(mr_) << "Failed to register memory region";

  memset(&list, 0, sizeof(list));
  list.addr = (uintptr_t)ping_buff_;
  list.length = kPingBuffSize;
  list.lkey = mr_->lkey;

  ping_sge_list_ = list;
  // Create queue pair
  {
    struct ibv_qp_init_attr attr;
    memset(&attr, 0, sizeof(ibv_qp_init_attr));
    attr.send_cq = adapter_->cq_;
    attr.recv_cq = adapter_->cq_;
    attr.cap.max_send_wr = adapter_->params_.queue_depth;
    attr.cap.max_recv_wr = adapter_->params_.queue_depth;
    attr.cap.max_send_sge = 1;
    attr.cap.max_recv_sge = 1;
    attr.qp_type = IBV_QPT_RC;

    qp_ = ibv_create_qp(adapter_->pd_, &attr);
    CHECK(qp_) << "Failed to create queue pair";
  }

  // Init queue pair
  {
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(ibv_qp_attr));
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = adapter_->params_.pkey_index;
    attr.port_num = adapter_->params_.port_num;
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;

    int mask =
        IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
    CHECK(!ibv_modify_qp(qp_, &attr, mask)) << "Failed to set QP to INIT";
  }

  // Local address
  {
    struct ibv_port_attr attr;
    CHECK(
        !ibv_query_port(adapter_->context_, adapter_->params_.port_num, &attr))
        << "Query port";
    self_.lid = attr.lid;
    self_.qpn = qp_->qp_num;
    self_.psn = static_cast<uint32_t>(random::New64()) & 0xffffff;
    union ibv_gid gid;
    CHECK(!ibv_query_gid(adapter_->context_, adapter_->params_.port_num,
                         adapter_->params_.sgid_index, &gid))
        << "Query gid";
    self_.snp = gid.global.subnet_prefix;
    self_.iid = gid.global.interface_id;
  }

  // create message and ack buffers, then initialize the tables.
  {
    const string buffer_names[] = {"tx_message_buffer", "rx_message_buffer"};
    tx_message_buffer_ = new RdmaMessageBuffer(this, buffer_names[0]);
    rx_message_buffer_ = new RdmaMessageBuffer(this, buffer_names[1]);
    message_buffers_.reserve(kNumMessageBuffers);
    message_buffers_.push_back(tx_message_buffer_);
    message_buffers_.push_back(rx_message_buffer_);
    // create buffer on host
    tx_message_buffer_->CreateCPUBuffer(RdmaMessage::kRdmaMessageBufferSize);
    rx_message_buffer_->CreateCPUBuffer(RdmaMessage::kRdmaMessageBufferSize);
  }
  CHECK(PingPostRecv() == 0) << "Couldn't post receive from " << remote_name_
                             << " with error " << std::strerror(errno);
}

RdmaChannel::~RdmaChannel() {
  ibv_dereg_mr(mr_);
  CHECK(!ibv_destroy_qp(qp_)) << "Failed to destroy QP";
  delete tx_message_buffer_;
  delete rx_message_buffer_;
}

void RdmaChannel::SetRemoteAddress(const RdmaAddress& ra, bool override) {
  mutex_lock lock{mu_};
  if ((override) || (!remote_set_)) {
    remote_.lid = ra.lid;
    remote_.qpn = ra.qpn;
    remote_.psn = ra.psn;
    remote_.snp = ra.snp;
    remote_.iid = ra.iid;
    remote_set_ = true;
  } else {
    CHECK(remote_.lid == ra.lid);
    CHECK(remote_.qpn == ra.qpn);
    CHECK(remote_.psn == ra.psn);
    CHECK(remote_.snp == ra.snp);
    CHECK(remote_.iid == ra.iid);
  }
}

// Adding tokens to the completion queue
// Tokens are needed to process future messages.
void RdmaChannel::Recv() {
  struct ibv_recv_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = (uint64_t)this;
  struct ibv_recv_wr* bad_wr;
  CHECK(!ibv_post_recv(qp_, &wr, &bad_wr)) << "Failed to post recv";
}

RdmaTensorRequest* RdmaChannel::InsertTensorRequest(
    const string& key, int64 step_id, Device* dst_dev,
    const Rendezvous::Args recv_args,
    const RdmaTensorRequest::RecvDoneCallback& done) {
  mutex_lock lock{ct_mu_};
  uint32_t request_index = request_serial_++;
  if (request_serial_ > RDMA_IMM_MAX_REQUEST_ID) {
    request_serial_ = 0;
  }
  RdmaTensorRequest request(request_index, key, step_id, this, dst_dev,
                            recv_args, done);
  auto it = request_table_.emplace(request_index, request);
  return &it.first->second;
}

void RdmaChannel::RemoveTensorRequest(uint32_t request_index) {
  mutex_lock lock{ct_mu_};
  request_table_.erase(request_index);
}

RdmaTensorRequest* RdmaChannel::GetTensorRequest(uint32_t request_index) {
  mutex_lock lock{ct_mu_};
  RequestTable::iterator iter = request_table_.find(request_index);
  CHECK(iter != request_table_.end());
  return &iter->second;
}

void RdmaChannel::Connect() {
  {
    mutex_lock lock{mu_};
    CHECK(remote_set_) << "remote channel is not set";
  }
  Connect(remote_);
}

// Setup channel to a remote node
// Args:
//   remoteAddr: the rdma address of a remote channel.
// Returns:
//   None
void RdmaChannel::Connect(const RdmaAddress& remoteAddr) {
  mutex_lock lock{mu_};
  if (!connected_) {
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(ibv_qp_attr));
    attr.qp_state = IBV_QPS_RTR;

    // This assumes both QP's ports are configured with the same MTU
    attr.path_mtu = adapter_->params_.mtu;
    attr.dest_qp_num = remoteAddr.qpn;
    attr.rq_psn = remoteAddr.psn;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    attr.ah_attr.is_global = 1;
    attr.ah_attr.grh.dgid.global.subnet_prefix = remoteAddr.snp;
    attr.ah_attr.grh.dgid.global.interface_id = remoteAddr.iid;
    attr.ah_attr.grh.flow_label = 0;
    attr.ah_attr.grh.hop_limit = 255;
    attr.ah_attr.dlid = remoteAddr.lid;
    attr.ah_attr.sl = adapter_->params_.sl;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = adapter_->params_.port_num;
    attr.ah_attr.grh.sgid_index = adapter_->params_.sgid_index;
    attr.ah_attr.grh.traffic_class = adapter_->params_.traffic_class;

    int r;
    CHECK(!(r = ibv_modify_qp(qp_, &attr,
                              IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                                  IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                                  IBV_QP_MAX_DEST_RD_ATOMIC |
                                  IBV_QP_MIN_RNR_TIMER)))
        << "QP to Ready to Receive " << r;

    memset(&attr, 0, sizeof(ibv_qp_attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.sq_psn = self_.psn;
    attr.timeout = adapter_->params_.timeout;
    attr.retry_cnt = adapter_->params_.retry_cnt;
    attr.rnr_retry = 7; /* infinite */
    attr.max_rd_atomic = 1;

    CHECK(!(r = ibv_modify_qp(qp_, &attr,
                              IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                                  IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                                  IBV_QP_MAX_QP_RD_ATOMIC)))
        << "QP to Ready to Send " << r;

    connected_ = true;
  } else {
    RDMA_LOG(2) << "channel already connected";
  }
}

RdmaMessageBuffer::RdmaMessageBuffer(RdmaChannel* channel, string name)
    : channel_(channel), name_(name) {}

RdmaMessageBuffer::~RdmaMessageBuffer() {
  CHECK(!ibv_dereg_mr(self_)) << "ibv_dereg_mr failed";
  FreeBuffer();
}

void RdmaMessageBuffer::FreeBuffer() {
  if ((buffer_ != nullptr) && buffer_on_host_) {
    free(buffer_);
  }
}

// Allocate CPU memory for the Rdma buffer
// Args:
//   size: to-be-allocated memory size
//   lock: whether or not mutex_lock the process to protect concurrency.
// Returns:
//   None
void RdmaMessageBuffer::CreateCPUBuffer(size_t size, bool lock) {
  CHECK(size > 0);
  if (lock) {
    mu_.lock();
  }
  if (local_status_ != none) {
    // delete existing buffer
    CHECK(!ibv_dereg_mr(self_)) << "ibv_dereg_mr failed";
    FreeBuffer();
  }
  size_ = size;
  buffer_ = malloc(size_);
  self_ = ibv_reg_mr(channel_->adapter_->pd_, buffer_, size_,
                     IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
  CHECK(self_) << "Failed to register memory region";
  buffer_on_host_ = true;
  local_status_ = idle;
  if (lock) {
    mu_.unlock();
  }
}

// Set address of remote memory region
// Args:
//   rmr: address of remote memory region
//   override: whether override existing information
// Returns:
//   None
void RdmaMessageBuffer::SetRemoteMR(RemoteMR rmr, bool override) {
  mutex_lock lock{mu_};
  if ((override) || (remote_status_ == none)) {
    remote_.remote_addr = rmr.remote_addr;
    remote_.rkey = rmr.rkey;
    remote_status_ = idle;
  } else {
    CHECK(remote_.remote_addr == rmr.remote_addr);
    CHECK(remote_.rkey == rmr.rkey);
  }
}

// Put a task in the buffer's job queue
void RdmaMessageBuffer::EnqueueItem(string item) {
  mutex_lock lock{mu_};
  queue_.push(item);
}

// Rdma-Write the content of the buffer
void RdmaMessageBuffer::Write(uint32_t imm_data, size_t buffer_size) {
  Write(channel_, imm_data, buffer_size, (uint64_t)buffer_, self_->lkey,
        remote_.remote_addr, remote_.rkey, RDMA_WRITE_ID_MESSAGE, this);
}

// Generalized Write method
void RdmaMessageBuffer::Write(const RdmaChannel* channel, uint32_t imm_data,
                              size_t buffer_size, uint64_t src_addr,
                              uint32_t lkey, uint64_t remote_addr,
                              uint32_t rkey, RdmaWriteIDType write_type,
                              void* write_context) {
  struct ibv_sge list;
  list.addr = src_addr;
  list.length = buffer_size;
  list.lkey = lkey;

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = (uint64_t) new RdmaWriteID(write_type, write_context);
  wr.sg_list = &list;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.imm_data = imm_data;
  wr.wr.rdma.remote_addr = remote_addr;
  wr.wr.rdma.rkey = rkey;

  struct ibv_send_wr* bad_wr;
  CHECK(!ibv_post_send(channel->qp_, &wr, &bad_wr)) << "Failed to post send";
}

// Send the next ack from the buffer's job queue.
void RdmaMessageBuffer::SendAck(const RdmaChannel* channel) {
  Write(channel, RDMA_IMM_DATA_ACK, 0, 0, 0, 0, 0, RDMA_WRITE_ID_ACK, nullptr);
}

// Send the next message from the buffer's job queue.
void RdmaMessageBuffer::SendNextItem() {
  uint32_t imm_data = RDMA_IMM_DATA_MESSAGE;
  mu_.lock();
  if (!queue_.empty() && (local_status_ == idle) && (remote_status_ == idle)) {
    local_status_ = busy;
    remote_status_ = busy;
    string message = queue_.front();
    queue_.pop();
    // local/remote_status_ won't be set back to idle
    // unitl Write() is successful
    mu_.unlock();
    memcpy(buffer_, message.data(), message.size());
    Write(imm_data, message.size());
  } else {
    mu_.unlock();
  }
}

#if GOOGLE_CUDA
static void CountCopies(const std::string& key, void* src_addr, void* dst_addr,
                        size_t tensor_bytes, bool is_gpu_to_cpu) {
#ifdef RDMA_COUNT_COPIES
  static uint64_t numGPUToCPUCopies = 0;
  static uint64_t numGPUToCPUCopiedBytes = 0;
  static uint64_t numCPUToGPUCopies = 0;
  static uint64_t numCPUToGPUCopiedBytes = 0;
  static uint64_t numTotalCopies = 0;

  if (is_gpu_to_cpu) {
    ++numGPUToCPUCopies;
    numGPUToCPUCopiedBytes += tensor_bytes;
  } else {
    ++numCPUToGPUCopies;
    numCPUToGPUCopiedBytes += tensor_bytes;
  }
  if ((++numTotalCopies % 0x400) == 0) {
    RDMA_LOG(0) << "Tensor copies:"
                << " GPU to CPU: " << numGPUToCPUCopies << " ("
                << numGPUToCPUCopiedBytes << " Bytes)"
                << " CPU to GPU: " << numCPUToGPUCopies << " ("
                << numCPUToGPUCopiedBytes << " Bytes)";
  }
  RDMA_LOG(2) << "Copying tensor " << key << " From: " << src_addr
              << " To: " << dst_addr;
#endif  // RDMA_COUNT_COPIES
}
#endif  // GOOGLE_CUDA

#ifdef RDMA_DATA_VALIDATION
static uint64_t Checksum(Device* device, const DeviceContext* device_context,
                         const Tensor& in) {
  uint64 checksum = 0;
  if (DataTypeCanUseMemcpy(in.dtype())) {
#if GOOGLE_CUDA
    if (in.TotalBytes() == 0) {
      return 0;
    }
    checksum = (device_context != nullptr)
                   ? GPUUtil::Checksum(device, device_context, in)
                   : GPUUtil::Checksum(in);
#endif  // GOOGLE_CUDA
  } else {
    string s = in.SummarizeValue(999999);
    checksum = Hash64(s.c_str(), s.size(), 0);
  }
  return checksum;
}

static void ValidateChecksum(uint64_t expected, uint64_t actual,
                             const Tensor& in, uint32_t request_index,
                             const std::string& key, const std::string& msg) {
  RDMA_LOG(2) << "Request #" << request_index << ": " << key
              << ": Checksum: " << std::hex << " Expected = 0x" << expected
              << ". Actual = 0x" << actual << ".";

  if (expected != actual) {
    // Checksum failed. There is one case where this is allowed - if the
    // tensor is an AssignAdd of the global step. Since the data-validation
    // always postpones the Tensor response in order to send a checksum message,
    // it is possible that the global-step was updated while the response was
    // still in queue.
    if ((in.TotalBytes() == 8) && (in.dtype() == DT_INT64)) {
      int64_t prev_val = *(int64_t*)DMAHelper::base(&in) - 1;
      actual = Hash64((const char*)&prev_val, 8, 0);
    }
    if (expected != actual) {
      LOG(FATAL) << "[" << msg << "]: Checksum validation failed for request #"
                 << request_index << ": " << key << std::hex << " "
                 << DataTypeString(in.dtype()) << " "
                 << in.shape().DebugString() << " (0x" << in.TotalBytes()
                 << " bytes): "
                 << " Expected 0x" << expected << ". Got 0x" << actual << ".";
    }
  }
}
#endif  // RDMA_DATA_VALIDATION

#if GOOGLE_CUDA
// Sync the 'done' operation on the GPU stream, but without all the data
// copying.
static void StreamGPUOp(Device* gpu_device, const DeviceContext* device_context,
                        StatusCallback done) {
  Tensor dummy1, dummy2;
  GPUUtil::CopyGPUTensorToCPU(gpu_device, device_context, &dummy1, &dummy2,
                              done);
}
#endif  // GOOGLE_CUDA

RdmaTensorResponse* RdmaChannel::AddTensorResponse(const RdmaMessage& rm) {
  mutex_lock lock{mu_};
  auto it =
      responses_table_.emplace(rm.request_index_, RdmaTensorResponse(this, rm));
  CHECK(it.second) << "Response with the ID " << rm.request_index_
                   << " already exists.";
  return &it.first->second;
}

RdmaTensorResponse* RdmaChannel::UpdateTensorResponse(const RdmaMessage& rm) {
  mutex_lock lock{mu_};
  auto it = responses_table_.find(rm.request_index_);
  CHECK(it != responses_table_.end()) << "No response found.";
  RdmaTensorResponse* response = &it->second;
  response->Update(rm);
  return response;
}

void RdmaChannel::RemoveTensorResponse(uint32_t request_index) {
  mutex_lock lock{mu_};
  responses_table_.erase(request_index);
}

void RdmaTensorResponse::Start() {
  Rendezvous::ParsedKey parsed;
  Status s = Rendezvous::ParseKey(rm_.name_, &parsed);
  if (!s.ok()) {
    SendErrorStatus(s);
    return;
  }

  channel_->adapter_->worker_env_->rendezvous_mgr->RecvLocalAsync(
      rm_.step_id_, parsed,
      [this, parsed](const Status& status, const Rendezvous::Args& send_args,
                     const Rendezvous::Args& recv_args, const Tensor& in,
                     bool is_dead) {
        CHECK(status.ok()) << "RecvLocalAsync was not ok."
                           << " error message: " << status.error_message();
        RecvHandler(parsed, send_args, recv_args, in, is_dead);
      });
}

void RdmaTensorResponse::Resume() { SendContent(*tensor_, *proto_, is_dead_); }

// Helper for RecvTensor. Validates "key" and returns the source
// device in "*src_dev".
Status RdmaTensorResponse::PrepareRecvTensor(
    const Rendezvous::ParsedKey& parsed, Device** src_dev) {
  // Figures out which device the tensor is hosted on.
  string local_name = DeviceNameUtils::LocalName(parsed.src_device);
  TF_RETURN_IF_ERROR(channel_->adapter_->worker_env_->device_mgr->LookupDevice(
      local_name, src_dev));

  // Does the device have the right incarnation number we expect?
  if ((*src_dev)->attributes().incarnation() != parsed.src_incarnation) {
    return errors::Aborted(
        "RecvTensor expects a different device incarnation: ",
        parsed.src_incarnation, " vs. ", (*src_dev)->attributes().incarnation(),
        ". Your worker job (\"",
        channel_->adapter_->worker_env_->session_mgr->LegacySession()
            ->worker_name,
        "\") was probably restarted. Check your "
        "worker job for the reason why it was restarted.");
  }

  return Status::OK();
}

void RdmaTensorResponse::RecvHandler(Rendezvous::ParsedKey parsed,
                                     const Rendezvous::Args& send_args,
                                     const Rendezvous::Args& recv_args,
                                     const Tensor& in, bool is_dead) {
  Status s = PrepareRecvTensor(parsed, &src_dev_);
  if (!s.ok()) {
    SendErrorStatus(s);
    return;
  }

  meta_data_changed_ = TensorMetaDataChanged(in, is_dead);
#ifdef RDMA_DATA_VALIDATION
  // Always send a meta data message with the source checksum
  meta_data_changed_ = rm_.type_ == RDMA_MESSAGE_TENSOR_REQUEST;
  checksum_ = Checksum(src_dev_, send_args.device_context, in);
#endif
  bool can_memcpy = DataTypeCanUseMemcpy(in.dtype());
  // string tensor needs to be serialized
  Tensor copy;
  TensorProto proto;
  const bool on_host = send_args.alloc_attrs.on_host();
  if (src_dev_->tensorflow_gpu_device_info() && !on_host) {
#if GOOGLE_CUDA
    DeviceContext* send_dev_context = send_args.device_context;
    CHECK(send_dev_context)
        << "send dev name: " << src_dev_->name()
        << " gpu_info: " << src_dev_->tensorflow_gpu_device_info();

    if (can_memcpy) {
      // If the tensor is located on a GDR compatible GPU, there is no need to
      // copy it. We can send directly from the source, just need to make sure
      // we are in sync with the GPU stream.
      // If the tensor's meta-data changed however, we will need to clone it,
      // so anyway we'll have to copy it from GPU to CPU first. If at some
      // point in time Clone() is changed to only save a shallow copy, we can
      // skip the copy here as well.
      if ((in.TotalBytes() > 0) && !meta_data_changed_ &&
          (RdmaMemoryMgr::Singleton().FindMemoryRegion(
               (void*)DMAHelper::base(&in), in.TotalBytes()) != nullptr)) {
        StreamGPUOp(src_dev_, send_dev_context,
                    [this, in, proto, is_dead](const Status& s) {
                      Send(in, proto, is_dead, s);
                    });
        return;
      }

      // The tensor must be copied from GPU to CPU, because either:
      // 1. The tensor is located on a non GDR compatible GPU.
      // 2. The tensor's meta-data has changed.
      Allocator* alloc = GPUProcessState::singleton()->GetCUDAHostAllocator(0);
      copy = Tensor(alloc, in.dtype(), in.shape());
      CountCopies(rm_.name_, (void*)DMAHelper::base(&in),
                  (void*)DMAHelper::base(&copy), in.TotalBytes(), true);
      GPUUtil::CopyGPUTensorToCPU(
          src_dev_, send_dev_context, &in, &copy,
          [this, copy, proto, is_dead](const Status& s) {
            Send(copy, proto, is_dead, s);
          });
    } else {
      GPUUtil::SetProtoFromGPU(
          in, src_dev_, send_args.device_context, &proto, is_dead,
          [this, in, proto, is_dead](const Status& s) mutable {
            Send(in, proto, is_dead, s);
          });
    }
#else
    SendErrorStatus(errors::Internal("No GPU device in process"));
#endif  // GOOGLE_CUDA
  } else {
    // tensor is in CPU memory.
    if (!can_memcpy) {
      in.AsProtoTensorContent(&proto);
    }
    Send(in, proto, is_dead, Status::OK());
  }
}

void RdmaTensorResponse::Send(const Tensor& in, const TensorProto& proto,
                              bool is_dead, const Status& status) {
  if (!status.ok()) {
    SendErrorStatus(status);
    return;
  }
  bool can_memcpy = DataTypeCanUseMemcpy(in.dtype());
  bool proto_size_changed =
      (!can_memcpy) && (proto.ByteSize() != rm_.tensor_bytes_);
  if (meta_data_changed_ || proto_size_changed) {
    Clone(in, proto, is_dead);
    SendMetaData(in, proto, is_dead);
  } else {
    SendContent(in, proto, is_dead);
  }
}

bool RdmaTensorResponse::TensorMetaDataChanged(const Tensor& in, bool is_dead) {
  return (rm_.data_type_ != in.dtype()) || (rm_.tensor_shape_ != in.shape()) ||
         (rm_.is_dead_ != is_dead);
}

void RdmaTensorResponse::Clone(const Tensor& in, const TensorProto& proto,
                               bool is_dead) {
  // Clone the data to be sent later. For simplicity, we clone the tensor's
  // data even if it is already a copy. Performance is less of a concern here
  // since the meta-data hardly ever changes. The reason we create a copy, is
  // that some tensors share their buffer between different step-ids, so the
  // tensor content may change before re-request was completed.
  bool can_memcpy = DataTypeCanUseMemcpy(in.dtype());
  if (can_memcpy && (in.TotalBytes() > 0)) {
    AllocatorAttributes host_alloc_attrs;
    host_alloc_attrs.set_nic_compatible(true);
    host_alloc_attrs.set_on_host(true);
    Allocator* allocator = src_dev_->GetAllocator(host_alloc_attrs);
    tensor_ = new Tensor(allocator, in.dtype(), in.shape());
    memcpy(DMAHelper::base(tensor_), DMAHelper::base(&in), in.TotalBytes());
  } else {
    tensor_ = new Tensor(in.dtype(), in.shape());
  }
  if (!can_memcpy) {
    proto_ = new TensorProto(proto);
  }
  is_dead_ = is_dead;
}

void RdmaTensorResponse::SendMetaData(const Tensor& in,
                                      const TensorProto& proto, bool is_dead) {
  RDMA_LOG(2) << "Request #" << rm_.request_index_
              << ": Meta data changed: " << rm_.name_;
  bool can_memcpy = DataTypeCanUseMemcpy(in.dtype());
  size_t tensor_bytes = (can_memcpy) ? in.TotalBytes() : proto.ByteSize();

  // Send meta-data update:
  RdmaMessage rm;
  rm.type_ = RDMA_MESSAGE_META_DATA_UPDATE;
  rm.name_size_ = rm_.name_.size();
  rm.name_ = rm_.name_;
  rm.tensor_shape_ = in.shape();
  rm.data_type_ = in.dtype();
  rm.step_id_ = rm_.step_id_;
  rm.is_dead_ = is_dead;
  rm.tensor_bytes_ = tensor_bytes;
  rm.request_index_ = rm_.request_index_;
#ifdef RDMA_DATA_VALIDATION
  rm.checksum_ = checksum_;
#endif
  RDMA_LOG(1) << "Step 0x" << std::hex << rm.step_id_ << std::dec
              << ": Sending RDMA_MESSAGE_META_DATA_UPDATE #"
              << rm.request_index_ << ": " << rm.name_
              << " (shape = " << rm.tensor_shape_.DebugString() << "."
              << " data-type = " << DataTypeString(rm.data_type_) << "."
              << " is-dead = " << rm.is_dead_ << ")";

  string message = RdmaMessage::CreateMessage(rm);
  channel_->tx_message_buffer_->EnqueueItem(message);
  channel_->tx_message_buffer_->SendNextItem();
}

void RdmaTensorResponse::SendContent(const Tensor& in, const TensorProto& proto,
                                     bool is_dead) {
  bool can_memcpy = DataTypeCanUseMemcpy(in.dtype());
  size_t tensor_bytes = (can_memcpy) ? in.TotalBytes() : proto.ByteSize();
  uint32_t imm_data = rm_.request_index_;
  if (!is_dead) {
    if (can_memcpy) {
      src_buffer_ = const_cast<TensorBuffer*>(DMAHelper::buffer(&in));
      if (src_buffer_ != nullptr) {
        src_buffer_->Ref();  // Keep buffer alive until write is complete
        src_addr_ = src_buffer_->data();
        mr_ = RdmaMemoryMgr::Singleton().FindMemoryRegion(src_addr_,
                                                          tensor_bytes);
      }
    } else {
      RDMA_LOG(2) << "Encoding proto: " << rm_.name_
                  << " (Size: " << tensor_bytes << ") " << in.DebugString();
      src_addr_ = malloc(tensor_bytes);
      mr_ = ibv_reg_mr(channel_->adapter_->pd_, src_addr_, tensor_bytes,
                       IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
      proto.SerializeToArray(src_addr_, tensor_bytes);
    }
  } else {
    tensor_bytes = 0;
  }

  uint32_t lkey = (mr_ == nullptr) ? 0 : mr_->lkey;
  RDMA_LOG(1) << "Step 0x" << std::hex << rm_.step_id_ << std::dec
              << ": Sending tensor content #" << rm_.request_index_ << " from "
              << std::hex << src_addr_ << " (0x" << lkey << ")"
              << " to " << rm_.remote_addr_ << " (0x" << rm_.rkey_
              << "): " << rm_.name_ << " (size: 0x" << std::hex << tensor_bytes
              << ")";

  RdmaMessageBuffer::Write(channel_, imm_data, tensor_bytes,
                           (uint64_t)src_addr_, lkey, rm_.remote_addr_,
                           rm_.rkey_, RDMA_WRITE_ID_TENSOR_WRITE, this);
}

void RdmaTensorResponse::SendErrorStatus(const Status& status) {
  RdmaMessage rm;
  rm.type_ = RDMA_MESSAGE_ERROR_STATUS;
  rm.name_size_ = rm_.name_.size();
  rm.name_ = rm_.name_;
  rm.step_id_ = rm_.step_id_;
  rm.request_index_ = rm_.request_index_;
  rm.status_ = status;
  LOG(ERROR) << "Step 0x" << std::hex << rm.step_id_ << std::dec
             << ": Sending RDMA_MESSAGE_ERROR_STATUS #" << rm.request_index_
             << ": " << rm.name_ << ". Status: " << status.ToString();

  string message = RdmaMessage::CreateMessage(rm);
  channel_->tx_message_buffer_->EnqueueItem(message);
  channel_->tx_message_buffer_->SendNextItem();

  // Destroy the response.
  Destroy();
}

void RdmaTensorResponse::Destroy() {
  if (src_buffer_ != nullptr) {
    src_buffer_->Unref();
  }
  if (tensor_ != nullptr) {
    delete tensor_;
  }
  if (proto_ != nullptr) {
    ibv_dereg_mr(mr_);
    free(src_addr_);
    delete proto_;
  }
  // Remove response from the pending list:
  channel_->RemoveTensorResponse(rm_.request_index_);
}

// Create a RdmaMessage according to the pre-defined format
// Args:
//   rm: the message structure
// Returns:
//   message in string format
string RdmaMessage::CreateMessage(const RdmaMessage& rm) {
  // Rdma Message format
  // type|name_size|name|step_id|request_index|remote_addr|rkey|is_dead|...
  //   1B|    2B   | 512|  8B   |     8B      |       8B  | 4B |    1B |...
  // ...|data_type|tensor_shape|tensor_bytes|error_status          |
  // ...|   XB    |    XB      |    8B      |size - 4B, proto - XB |
  //
  // ACK:             Imm-type: ACK
  // TENSOR_REQUEST:  Imm-type: MESSAGE
  //                  Fields: type, request_index, name, step_id, remote_addr,
  //                      rkey, is_dead, data_type, tensor_shape, tensor_bytes
  // META_DATA_UPDATE: Imm-type: MESSAGE
  //                  Fields: type, request_index, is_dead, data_type,
  //                      tensor_shape, tensor_bytes
  // TENSOR_RE_REQUST: Imm-type: MESSAGE
  //                  Fields: type, request_index, name, step_id, remote_addr,
  //                      rkey, is_dead, data_type, tensor_shape, tensor_bytes
  // ERROR_STATUS:    Imm-type: MESSAGE
  //                  Fields: type, request_index, name, step_id, error_status
  // Tensor content:  Imm-type: request_index
  size_t message_size = kMessageTotalBytes;
  char message[kMessageTotalBytes + kErrorStatusMaxSize];
  // type
  message[kTypeStartIndex] = static_cast<char>(rm.type_) & 0xff;
  // request index
  memcpy(&message[kRequestIndexStartIndex], &rm.request_index_,
         sizeof(rm.request_index_));
  // name, step_id, remote_addr, rkey
  if ((rm.type_ == RDMA_MESSAGE_TENSOR_REQUEST) ||
      (rm.type_ == RDMA_MESSAGE_TENSOR_RE_REQUEST)) {
    memcpy(&message[kNameSizeStartIndex], &rm.name_size_,
           sizeof(rm.name_size_));
    memcpy(&message[kNameStartIndex], rm.name_.data(), rm.name_.size());
    memcpy(&message[kRemoteAddrStartIndex], &rm.remote_addr_,
           sizeof(rm.remote_addr_));
    memcpy(&message[kRkeyStartIndex], &rm.rkey_, sizeof(rm.rkey_));
    memcpy(&message[kStepIdStartIndex], &rm.step_id_, sizeof(rm.step_id_));
  }
  // is_dead, data_type, tensor_shape, tensor_bytes
  if ((rm.type_ == RDMA_MESSAGE_TENSOR_REQUEST) ||
      (rm.type_ == RDMA_MESSAGE_META_DATA_UPDATE) ||
      (rm.type_ == RDMA_MESSAGE_TENSOR_RE_REQUEST)) {
    memcpy(&message[kIsDeadStartIndex], &rm.is_dead_, sizeof(rm.is_dead_));

    memcpy(&message[kDataTypeStartIndex], &rm.data_type_,
           sizeof(rm.data_type_));
    memcpy(&message[kTensorShapeStartIndex], &rm.tensor_shape_,
           sizeof(rm.tensor_shape_));
    memcpy(&message[kTensorBytesStartIndex], &rm.tensor_bytes_,
           sizeof(rm.tensor_bytes_));
  }
  // checksum
#ifdef RDMA_DATA_VALIDATION
  memcpy(&message[kChecksumStartIndex], &rm.checksum_, sizeof(rm.checksum_));
#endif
  // error status
  if (rm.type_ == RDMA_MESSAGE_ERROR_STATUS) {
    ::grpc::Status gs = ToGrpcStatus(rm.status_);
    ErrorStatusProto gsProto;
    gsProto.set_error_code(gs.error_code());
    gsProto.set_error_message(gs.error_message());
    gsProto.set_error_details(gs.error_details());
    uint32_t gsProtoSize = gsProto.ByteSize();
    if (gsProtoSize + 4 > kErrorStatusMaxSize) {
      LOG(ERROR) << "Error status (" << gsProtoSize + 4 << " bytes) "
                 << "is too big to fit in RDMA message (" << kErrorStatusMaxSize
                 << " bytes). Truncated.";
      gsProtoSize = kErrorStatusMaxSize - 4;
    }
    uint32_t* proto_size = (uint32_t*)&message[kErrorStatusStartIndex];
    *proto_size = gsProtoSize;
    gsProto.SerializeToArray(&message[kErrorStatusStartIndex + 4], gsProtoSize);
    message_size += gsProtoSize + 4;
  }
  return string(message, message_size);
}

// Parse a RdmaMessage according to the pre-defined format
// Args:
//   rm: the message structure where the parsed message will be saved
//   buffer: the place where the raw message is stored
// Returns:
//   None
void RdmaMessage::ParseMessage(RdmaMessage& rm, void* buffer) {
  char* message = static_cast<char*>(buffer);
  // type
  rm.type_ = static_cast<RdmaMessageType>(message[kTypeStartIndex]);
  // request index
  memcpy(&rm.request_index_, &message[kRequestIndexStartIndex],
         sizeof(rm.request_index_));
  // name, step_id, remote_addr, rkey
  if ((rm.type_ == RDMA_MESSAGE_TENSOR_REQUEST) ||
      (rm.type_ == RDMA_MESSAGE_TENSOR_RE_REQUEST)) {
    memcpy(&rm.name_size_, &message[kNameSizeStartIndex],
           sizeof(rm.name_size_));
    rm.name_ = string(&message[kNameStartIndex], rm.name_size_);
    memcpy(&rm.remote_addr_, &message[kRemoteAddrStartIndex],
           sizeof(rm.remote_addr_));
    memcpy(&rm.rkey_, &message[kRkeyStartIndex], sizeof(rm.rkey_));
    memcpy(&rm.step_id_, &message[kStepIdStartIndex], sizeof(rm.step_id_));
  }
  // data_type, tensor_bytes, tensor_shape, is_dead
  if ((rm.type_ == RDMA_MESSAGE_TENSOR_REQUEST) ||
      (rm.type_ == RDMA_MESSAGE_META_DATA_UPDATE) ||
      (rm.type_ == RDMA_MESSAGE_TENSOR_RE_REQUEST)) {
    memcpy(&rm.is_dead_, &message[kIsDeadStartIndex], sizeof(rm.is_dead_));
    memcpy(&rm.data_type_, &message[kDataTypeStartIndex],
           sizeof(rm.data_type_));
    memcpy(&rm.tensor_shape_, &message[kTensorShapeStartIndex],
           sizeof(rm.tensor_shape_));
    memcpy(&rm.tensor_bytes_, &message[kTensorBytesStartIndex],
           sizeof(rm.tensor_bytes_));
  }
  // checksum
#ifdef RDMA_DATA_VALIDATION
  memcpy(&rm.checksum_, &message[kChecksumStartIndex], sizeof(rm.checksum_));
#endif
  // error status
  if (rm.type_ == RDMA_MESSAGE_ERROR_STATUS) {
    ErrorStatusProto gsProto;
    uint32_t gsProtoSize = *(uint32_t*)&message[kErrorStatusStartIndex];
    CHECK(ParseProtoUnlimited(&gsProto, &message[kErrorStatusStartIndex + 4],
                              gsProtoSize))
        << "Failed to parse error status proto from message. Aborting.";
    ::grpc::Status gs((::grpc::StatusCode)gsProto.error_code(),
                      gsProto.error_message(), gsProto.error_details());
    rm.status_ = FromGrpcStatus(gs);
  }
}

//*****************************************************************************
// RdmaMemoryMgr
//*****************************************************************************

ibv_mr* RdmaMemoryMgr::FindMemoryRegion(void* addr, size_t length) {
  mutex_lock l(mrs_mu_);
  auto iter = std::upper_bound(mrs_.begin(), mrs_.end(), addr, &Comparator);
  if (iter == std::end(mrs_) || iter->get()->addr > addr) {
    return nullptr;
  } else {
    return iter->get();
  }
}

void RdmaMemoryMgr::InsertMemoryRegion(void* addr, size_t length,
                                       const std::string& allocator_name) {
  if (length == 0) return;
  ibv_mr* mr = ibv_reg_mr(pd_, addr, length,
                          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
  RDMA_LOG(1) << "Insert memory region 0x" << std::hex << mr->rkey << ". ["
              << addr << "-" << (void*)((uint64_t)addr + length - 1) << "]"
              << " SIZE: 0x" << length << " (" << allocator_name << ").";
  if (mr != nullptr) {
    mutex_lock l(mrs_mu_);
    auto iter = std::upper_bound(mrs_.begin(), mrs_.end(), addr, &Comparator);
    mrs_.insert(iter, {mr, &MRDeleter});
  } else {
    LOG(WARNING) << "Cannot register memory region";
  }
}

void RdmaMemoryMgr::EvictMemoryRegion(void* addr, size_t length) {
  if (length == 0) return;
  mutex_lock l(mrs_mu_);
  auto iter = std::upper_bound(mrs_.begin(), mrs_.end(), addr, &Comparator);
  if (iter != std::end(mrs_) && iter->get()->addr == addr) {
    mrs_.erase(iter);
    RDMA_LOG(1) << "Evict memory region 0x" << std::hex << iter->get()->rkey;

  } else {
    LOG(WARNING) << "Failed to de-register memory region";
  }
}

const TensorMetaData* RdmaMemoryMgr::GetTensorMetaData(
    const std::string& tensor_name) {
  mutex_lock l(tensor_meta_data_mu_);
  auto it = tensors_meta_data_.find(tensor_name);
  if (it == tensors_meta_data_.end()) {
    return nullptr;
  }
  return &it->second;
}

const TensorMetaData* RdmaMemoryMgr::SetTensorMetaData(
    const std::string& tensor_name, DataType dtype, const TensorShape& shape,
    bool is_dead, size_t proto_size) {
  mutex_lock l(tensor_meta_data_mu_);
  TensorMetaData& meta_data = tensors_meta_data_[tensor_name];
  meta_data.data_type_ = dtype;
  meta_data.tensor_shape_ = shape;
  meta_data.proto_size_ = proto_size;
  meta_data.is_dead_ = is_dead;
  return &meta_data;
}

//*****************************************************************************
// RdmaTensorRequest
//*****************************************************************************

RdmaTensorRequest::RdmaTensorRequest(
    uint32_t index, const string& key, int64 step_id, RdmaChannel* channel,
    Device* dst_dev, const Rendezvous::Args recv_args,
    const RdmaTensorRequest::RecvDoneCallback& done)
    : index_(index),
      key_(key),
      step_id_(step_id),
      channel_(channel),
      dst_dev_(dst_dev),
      recv_args_(recv_args),
      meta_data_(RdmaMemoryMgr::Singleton().GetTensorMetaData(key)),
      result_tensor_(nullptr),
      proxy_tensor_(nullptr),
      rdma_addr_(nullptr),
      mr_(nullptr),
      done_(done) {}

RdmaTensorRequest::~RdmaTensorRequest() { DeallocateTensors(); }

void RdmaTensorRequest::Done(const Status& s) {
  Tensor val = std::move(*result_tensor_);

#ifdef RDMA_DATA_VALIDATION
  // Validate checksum
  // Unfortunately we can't always do a Checksum directly on the result tensor.
  // If the result tensor is on GPU, then we need to copy it back to CPU. If
  // we happen to be in the midst of a proxy callback, then the copying will
  // get stuck.
  uint64_t checksum = (proxy_tensor_ != nullptr)
                          ? Checksum(nullptr, nullptr, *proxy_tensor_)
                          : Checksum(dst_dev_, recv_args_.device_context, val);
  ValidateChecksum(checksum_, checksum, val, index_, key_, "RDMA");
#endif

  Rendezvous::Args recv_args = std::move(recv_args_);
  bool is_dead = (meta_data_ == nullptr) ? false : meta_data_->is_dead_;
  RecvDoneCallback done = done_;
  DeallocateTensors();
  channel_->RemoveTensorRequest(index_);
  done(s, Rendezvous::Args(), recv_args, val, is_dead);
}

void RdmaTensorRequest::DeallocateTensors() {
  if (result_tensor_ != nullptr) {
    delete result_tensor_;
    result_tensor_ = nullptr;
  }
  if (proxy_tensor_ != nullptr) {
    delete proxy_tensor_;
    proxy_tensor_ = nullptr;
  }
}

bool RdmaTensorRequest::AllocateTensors() {
  result_tensor_ =
      new Tensor(dst_dev_->GetAllocator(recv_args_.alloc_attrs),
                 meta_data_->data_type_, meta_data_->tensor_shape_);

  size_t tensor_size = result_tensor_->TotalBytes();
  bool can_memcpy = DataTypeCanUseMemcpy(result_tensor_->dtype());
  if (can_memcpy) {
    if (tensor_size == 0) {
      return true;
    }
    rdma_addr_ = DMAHelper::base(result_tensor_);
    mr_ = RdmaMemoryMgr::Singleton().FindMemoryRegion(rdma_addr_, tensor_size);
#if GOOGLE_CUDA
    if (mr_ == nullptr) {
      // Can't RDMA directly to result. Use a proxy.
      proxy_tensor_ =
          new Tensor(GPUProcessState::singleton()->GetCUDAHostAllocator(0),
                     result_tensor_->dtype(), result_tensor_->shape());
      rdma_addr_ = DMAHelper::base(proxy_tensor_);
      mr_ =
          RdmaMemoryMgr::Singleton().FindMemoryRegion(rdma_addr_, tensor_size);
    }
#endif
  } else {
    uint32_t proto_size = meta_data_->proto_size_;
    rdma_addr_ = malloc(proto_size);
    mr_ = ibv_reg_mr(RdmaMemoryMgr::Singleton().pd_, rdma_addr_, proto_size,
                     IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
  }
  CHECK(mr_ != nullptr) << " No memory region found for address " << rdma_addr_
                        << ": " << key_;
  return true;
}

void RdmaTensorRequest::AllocateTensorsAsync(StatusCallback done) {
  AllocateTensors();
  bool on_host = recv_args_.alloc_attrs.on_host();
  if (dst_dev_->tensorflow_gpu_device_info() && !on_host &&
      (proxy_tensor_ == nullptr)) {
#if GOOGLE_CUDA
    // We need to sync the memory allocation on the GPU:
    StreamGPUOp(dst_dev_, recv_args_.device_context, done);
#endif
  } else {
    done(Status::OK());
  }
}

void RdmaTensorRequest::Send(RdmaMessageType message_type) {
  RdmaMessageBuffer* rb = channel_->tx_message_buffer_;
  RdmaMessage rm;
  rm.type_ = message_type;
  rm.request_index_ = index_;
  rm.name_size_ = key_.size();
  rm.name_ = key_;
  rm.step_id_ = step_id_;
  rm.remote_addr_ = (uint64_t)rdma_addr_;
  if (meta_data_ != nullptr) {
    rm.data_type_ = meta_data_->data_type_;
    rm.tensor_shape_ = meta_data_->tensor_shape_;
    rm.is_dead_ = meta_data_->is_dead_;
    rm.tensor_bytes_ = meta_data_->proto_size_;
  } else {
    rm.data_type_ = DT_INVALID;
  }
  rm.rkey_ = (mr_ == nullptr) ? 0 : mr_->rkey;

  RDMA_LOG(1) << "Step 0x" << std::hex << rm.step_id_ << std::dec
              << ": Sending  " << MessageTypeToString(message_type) << " #"
              << index_ << ": " << rm.name_ << " on " << rdma_addr_
              << " (rkey: 0x" << std::hex << rm.rkey_ << ")";

  string message = RdmaMessage::CreateMessage(rm);
  rb->EnqueueItem(message);
  rb->SendNextItem();
}

void RdmaTensorRequest::RecvTensorMetaData(DataType dtype, TensorShape shape,
                                           bool is_dead, size_t proto_size) {
  meta_data_ = RdmaMemoryMgr::Singleton().SetTensorMetaData(
      key_, dtype, shape, is_dead, proto_size);

  DeallocateTensors();
  AllocateTensorsAsync(
      [this](const Status& s) { Send(RDMA_MESSAGE_TENSOR_RE_REQUEST); });
}

void RdmaTensorRequest::RecvTensorContent() {
  bool can_memcpy = DataTypeCanUseMemcpy(meta_data_->data_type_);
  size_t message_size =
      can_memcpy ? result_tensor_->TotalBytes() : meta_data_->proto_size_;
  RDMA_LOG(1) << "Step 0x" << std::hex << step_id_ << std::dec
              << ": Received tensor content #" << index_ << ": " << key_
              << " (Size: 0x" << std::hex << message_size << ")";

  Tensor val;

#if GOOGLE_CUDA
  if (proxy_tensor_ != nullptr) {
    CountCopies(key_, (void*)DMAHelper::base(proxy_tensor_),
                (void*)DMAHelper::base(result_tensor_),
                result_tensor_->TotalBytes(), false);
    GPUUtil::CopyCPUTensorToGPU(proxy_tensor_, recv_args_.device_context,
                                dst_dev_, result_tensor_,
                                [this](const Status& s) {
                                  CHECK(s.ok()) << "copy tensor to gpu sync";
                                  Done(s);
                                });
    return;
  }
#endif

  if (can_memcpy) {
    Done(Status::OK());
  } else {
    RDMA_LOG(2) << "Decoding proto: " << key_
                << " (Size: " << meta_data_->proto_size_ << ")";
    TensorProto proto;
    CHECK(ParseProtoUnlimited(&proto, rdma_addr_, meta_data_->proto_size_))
        << "fail to parse proto from array";
    ibv_dereg_mr(mr_);
    free(rdma_addr_);
    Status s = dst_dev_->MakeTensorFromProto(proto, recv_args_.alloc_attrs,
                                             result_tensor_);
    Done(s);
  }
}

void RdmaTensorRequest::RecvErrorStatus(const Status& status) {
  if (result_tensor_ == nullptr) {
    result_tensor_ = new Tensor();
  }
  LOG(ERROR) << "Received RDMA_MESSAGE_ERROR_STATUS: " << status.ToString();
  Done(status);
}

void RdmaTensorRequest::Start() {
  meta_data_ = RdmaMemoryMgr::Singleton().GetTensorMetaData(key_);
  if (meta_data_ != nullptr) {
    AllocateTensorsAsync(
        [this](const Status& s) { Send(RDMA_MESSAGE_TENSOR_REQUEST); });
  } else {
    Send(RDMA_MESSAGE_TENSOR_REQUEST);
  }
}

}  // end namespace tensorflow

#endif
