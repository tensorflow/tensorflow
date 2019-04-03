#include "tensorflow/core/common_runtime/dma_helper.h"
#if GOOGLE_CUDA
#include "tensorflow/core/common_runtime/gpu/gpu_util.h"
#endif // GOOGLE_CUDA
#include "tensorflow/core/common_runtime/gpu/process_state.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/logging.h"

#include "tensorflow/contrib/seastar/seastar_client_tag.h"
#include "tensorflow/contrib/seastar/seastar_message.h"
#include "tensorflow/contrib/seastar/seastar_tensor_coding.h"

namespace tensorflow {
namespace {
void ProcessCallOptions(SeastarClientTag* tag) {
  if (tag->call_opts_ != nullptr) {
    if (!tag->call_opts_->UseWaitForReady()) {
      tag->fail_fast_ = true;
    }
    
    if (tag->call_opts_->GetTimeout() > 0) {
      tag->timeout_in_ms_ = tag->call_opts_->GetTimeout();
    }
  }
}
} // namespace

void InitSeastarClientTag(protobuf::Message* request,
			  protobuf::Message* response,
			  StatusCallback done,
			  SeastarClientTag* tag,
        CallOptions* call_opts) {
  tag->req_body_buf_.len_ = request->ByteSize();
  tag->req_body_buf_.data_ = new char[tag->req_body_buf_.len_]();
  request->SerializeToArray(tag->req_body_buf_.data_, tag->req_body_buf_.len_);

  tag->req_header_buf_.len_ = 32;
  tag->req_header_buf_.data_ = new char[32]();
  
  memcpy(tag->req_header_buf_.data_, "AAAAAAAA", 8);
  memcpy(tag->req_header_buf_.data_ + 8, &tag, 8);
  memcpy(tag->req_header_buf_.data_ + 16, &tag->method_, 4);
  // memcpy(tag->req_header_buf_.data_ + 20, &tag->status_, 2);
  memcpy(tag->req_header_buf_.data_ + 24, &tag->req_body_buf_.len_, 8);

  StatusCallback wrapper_done
    = std::bind([response, tag](StatusCallback done,
                                const Status& s) {
                  // internal error, that we dont need to parse the response,
                  // reponse is nullptr
                  if (s.code() != error::INTERNAL) {
                    response->ParseFromArray(tag->resp_body_buf_.data_,
                                            tag->resp_body_buf_.len_);
                  }
                  if (!s.ok()) {
                    if (tag->method_ == SeastarWorkerServiceMethod::kLogging ||
                        tag->method_ == SeastarWorkerServiceMethod::kTracing) {
                      // Logging & Tracing in worker.cc is UNIMPLEMENTED, ignore the error
                    } else {
                      // Debugging info
                      LOG(INFO) << "RPC's status is not ok. status code=" << s.code()
                                << ", err msg=" << s.error_message().c_str();
                    }
                  }
                  done(s);
                  delete tag;
                },
                std::move(done),
                std::placeholders::_1);

  tag->done_ = std::move(wrapper_done);
  tag->call_opts_ = call_opts;
  ProcessCallOptions(tag);
}

void InitSeastarClientTag(protobuf::Message* request,
			  SeastarTensorResponse* response,
			  StatusCallback done,
			  SeastarClientTag* tag,
        CallOptions* call_opts) {
  // LOG(INFO) << "InitSeastarClientTag for no fuse tensor recv";
  tag->req_body_buf_.len_ = request->ByteSize();
  tag->req_body_buf_.data_ = new char[tag->req_body_buf_.len_]();
  request->SerializeToArray(tag->req_body_buf_.data_,
                            tag->req_body_buf_.len_);

  tag->req_header_buf_.len_ = 32;
  tag->req_header_buf_.data_ = new char[32]();
  
  memcpy(tag->req_header_buf_.data_, "AAAAAAAA", 8);
  memcpy(tag->req_header_buf_.data_ + 8, &tag, 8);
  memcpy(tag->req_header_buf_.data_ + 16, &tag->method_, 4);
  // Ignore the status segment in request
  // memcpy(tag->req_header_buf_.data_ + 20, &tag->status_, 2);
  memcpy(tag->req_header_buf_.data_ + 24, &tag->req_body_buf_.len_, 8);

  // LOG(INFO) << "tensor request: " << request->DebugString();

 ParseMessageCallback wrapper_parse_message
    = [request, response, tag] (int idx) {
      SeastarMessage sm;
      SeastarMessage::DeserializeMessage(&sm, tag->resp_message_bufs_[idx].data_);

      response->SetIsDead(sm.is_dead_);
      response->SetDataType(sm.data_type_);
      bool can_memcpy = DataTypeCanUseMemcpy(sm.data_type_);

      if (can_memcpy) {
        if (response->GetDevice()->tensorflow_gpu_device_info() &&
            (!response->GetOnHost())) {
 #if GOOGLE_CUDA
          // LOG(INFO) << "parse msg, can memcpy and on GPU";
          // dst tensor on gpu
          Allocator* alloc = ProcessState::singleton()->GetCUDAHostAllocator(0);
          Tensor cpu_copy(alloc, sm.data_type_, sm.tensor_shape_);

          tag->resp_tensor_bufs_[idx].data_ = reinterpret_cast<char*>(DMAHelper::base(&cpu_copy));
          tag->resp_tensor_bufs_[idx].len_ =  sm.tensor_bytes_;
          tag->resp_tensor_bufs_[idx].owned_ = false;
          
          response->SetTensor(cpu_copy);
#else
          return errors::Internal("No GPU device in process");
#endif

        } else { 
          // LOG(INFO) << "parse msg for no fuse, can memcpy and on cpu"
          //          << ",request:" << request->DebugString();
          Tensor val(response->GetAlloc(), sm.data_type_, sm.tensor_shape_);
          tag->resp_tensor_bufs_[idx].data_ = reinterpret_cast<char*>(DMAHelper::base(&val));
          tag->resp_tensor_bufs_[idx].len_ =  sm.tensor_bytes_;
          tag->resp_tensor_bufs_[idx].owned_ = false;
          
          response->SetTensor(val);
        }
      } else {
        // LOG(INFO) << "parse msg, could not memcpy, tensor bytes: " << sm.tensor_bytes_
        //          << ",request:" << request->DebugString();
        tag->resp_tensor_bufs_[idx].len_ = sm.tensor_bytes_;
        tag->resp_tensor_bufs_[idx].data_ = new char[tag->resp_tensor_bufs_[idx].len_]();
      }

      return Status();
  };
  tag->parse_message_ = std::move(wrapper_parse_message);

  StatusCallback wrapper_done
    = std::bind([response, tag](StatusCallback done,
                                const Status& s) {
                  if (!s.ok()) {
                    LOG(ERROR) << "wrapper_done, status not ok. status code=" << s.code()
                               << ", err msg=" << s.error_message().c_str();
                    done(s);
                    delete tag;
                    return;
                  }
                  
                  bool can_memcpy = DataTypeCanUseMemcpy(response->GetDataType());
                  if (can_memcpy) {
                    if (response->GetDevice()->tensorflow_gpu_device_info() &&
                        (!response->GetOnHost())) {
#if GOOGLE_CUDA
                      Tensor* gpu_copy = new Tensor(response->GetAlloc(), response->GetTensor().dtype(), response->GetTensor().shape());
                      GPUUtil::CopyCPUTensorToGPU(&response->GetTensor(),
                                                  response->GetDevice()->tensorflow_gpu_device_info()->default_context,
                                                  response->GetDevice(),
                                                  gpu_copy,
                                                  [gpu_copy, response, done, tag](const Status& s) {
                                                    CHECK(s.ok()) << "copy tensor to gpu sync";
                                                    response->SetTensor(*gpu_copy);
                                                    done(s);
                                                    delete gpu_copy;
                                                    delete tag;
                                                  });
#else
                      done(errors::Internal("No GPU device in process"));
                      delete tag;
#endif
                    } else {
                      // LOG(INFO) << "wrapper_done for no fuse, nothon to do, in the case that tensor on cpu and can memcpy";
                      done(s);
                      delete tag;
                    }
                  } else {
                    // could not memcoy
                    // LOG(INFO) << "wrapper_done, could not memcpy, recv bytes: "
                    // << tag->resp_tensor_bufs_[0].len_
                    // << ", DataType: " << response->GetDataType();
                    ParseProtoUnlimited(&response->GetTensorProto(),
                                        tag->resp_tensor_bufs_[0].data_,
                                        tag->resp_tensor_bufs_[0].len_);
                    Tensor val;
                    Status status = response->GetDevice()->MakeTensorFromProto(
                        response->GetTensorProto(),
                        response->GetAllocAttributes(),
                        &val);
                    //LOG(INFO) << "parse msg status: " << status.error_message();
                    CHECK(status.ok()) << "make cpu tensor from proto.";
                    response->SetTensor(val);
                    done(status);
                    delete tag;
                  }
                },
                std::move(done),
                std::placeholders::_1);

  tag->done_ = std::move(wrapper_done);
  tag->call_opts_ = call_opts;
  ProcessCallOptions(tag);
}

void InitSeastarClientTag(protobuf::Message* request,
			  SeastarFuseTensorResponse* response,
			  StatusCallback done,
			  SeastarClientTag* tag,
        CallOptions* call_opts) {
  tag->req_body_buf_.len_ = request->ByteSize();
  tag->req_body_buf_.data_ = new char[tag->req_body_buf_.len_]();
  request->SerializeToArray(tag->req_body_buf_.data_, tag->req_body_buf_.len_);

  tag->req_header_buf_.len_ = 32;
  tag->req_header_buf_.data_ = new char[32]();
  
  memcpy(tag->req_header_buf_.data_, "AAAAAAAA", 8);
  memcpy(tag->req_header_buf_.data_ + 8, &tag, 8);
  memcpy(tag->req_header_buf_.data_ + 16, &tag->method_, 4);
  // Ignore the status segment in request
  // memcpy(tag->req_header_buf_.data_ + 20, &tag->status_, 2);
  memcpy(tag->req_header_buf_.data_ + 24, &tag->req_body_buf_.len_, 8);

  // LOG(INFO) << "tensor request: " << request->DebugString();

 ParseMessageCallback wrapper_parse_message
    = [request, response, tag] (int idx) {
      SeastarMessage sm;
      SeastarMessage::DeserializeMessage(&sm, tag->resp_message_bufs_[idx].data_);

      response->SetIsDeadByIndex(idx, sm.is_dead_);
      response->SetDataTypeByIndex(idx, sm.data_type_);
      bool can_memcpy = DataTypeCanUseMemcpy(sm.data_type_);

      if (can_memcpy) {
        if (response->GetDevice()->tensorflow_gpu_device_info() &&
            (!response->GetOnHost())) {
 #if GOOGLE_CUDA
          // LOG(INFO) << "parse msg, can memcpy and on GPU";
          // dst tensor on gpu
          Allocator* alloc = ProcessState::singleton()->GetCUDAHostAllocator(0);
          Tensor cpu_copy(alloc, sm.data_type_, sm.tensor_shape_);

          tag->resp_tensor_bufs_[idx].data_ = reinterpret_cast<char*>(DMAHelper::base(&cpu_copy));
          tag->resp_tensor_bufs_[idx].len_ =  sm.tensor_bytes_;
          tag->resp_tensor_bufs_[idx].owned_ = false;
          
          response->SetTensorByIndex(idx, cpu_copy);
#else
          return errors::Internal("No GPU device in process");
#endif

        } else { 
          // LOG(INFO) << "parse msg for fuse, can memcpy and on cpu"
          //          << "idx is: " << idx
          //          << ", request is:" << request->DebugString();
          Tensor val(response->GetAlloc(), sm.data_type_, sm.tensor_shape_);
          tag->resp_tensor_bufs_[idx].data_ = reinterpret_cast<char*>(DMAHelper::base(&val));
          tag->resp_tensor_bufs_[idx].len_ =  sm.tensor_bytes_;
          tag->resp_tensor_bufs_[idx].owned_ = false;
          
          response->SetTensorByIndex(idx, val);
        }
      } else {
        // LOG(INFO) << "parse msg, could not memcpy, tensor bytes: " << sm.tensor_bytes_
        //          << ", idx is: " << idx
        //          << ", request is:" << request->DebugString();
        tag->resp_tensor_bufs_[idx].len_ = sm.tensor_bytes_;
        tag->resp_tensor_bufs_[idx].data_ = new char[tag->resp_tensor_bufs_[idx].len_]();
      }

      return Status();
  };
  tag->parse_message_ = std::move(wrapper_parse_message);

  StatusCallback wrapper_done
    = std::bind([response, tag](StatusCallback done,
                                const Status& s) {
                  if (!s.ok()) {
                    LOG(ERROR) << "wrapper_done, status not ok. status code=" << s.code()
                               << ", err msg=" << s.error_message().c_str();
                    done(s);
                    delete tag;
                    return;
                  }

                  int fuse_count = tag->fuse_count_;
                  int *fuse_counter = new int(fuse_count);

                  for (int idx = 0; idx < fuse_count; ++idx) {
                    bool can_memcpy = DataTypeCanUseMemcpy(response->GetDataTypeByIndex(idx));
                    // LOG(INFO) << "wrapper_done for fuse recv, fuse count is: " << tag->fuse_count_
                    //          <<", idx is: " << idx << ", data type is: " << response->GetDataTypeByIndex(idx);
                    if (can_memcpy) {
                      if (response->GetDevice()->tensorflow_gpu_device_info() &&
                          (!response->GetOnHost())) {
#if GOOGLE_CUDA
                        Tensor* gpu_copy = new Tensor(response->GetAlloc(),
                                                      response->GetTensorByIndex(idx).dtype(),
                                                      response->GetTensorByIndex(idx).shape());
                        GPUUtil::CopyCPUTensorToGPU(&response->GetTensorByIndex(idx),
                                                    response->GetDevice()->tensorflow_gpu_device_info()->default_context,
                                                    response->GetDevice(),
                                                    gpu_copy,
                                                    [gpu_copy, response, done, tag, fuse_counter, idx](const Status& s) {
                                                      CHECK(s.ok()) << "copy tensor to gpu sync";
                                                      response->SetTensorByIndex(idx, *gpu_copy);
                                                      delete gpu_copy;
                                                      if (__sync_sub_and_fetch(fuse_counter, 1) == 0) {
                                                        delete fuse_counter;
                                                        done(s);
                                                        delete tag;
                                                      }
                                                    });
#else
                        done(errors::Internal("No GPU device in process"));
                        // delete tag;
                        // It may be not safe to delete tag here, just abort here.
                        abort();
#endif
                      } else {
                        // LOG(INFO) << "wrapper_done for fuse recv, nothon to do, in the case that tensor on cpu and can memcpy";
                        if (__sync_sub_and_fetch(fuse_counter, 1) == 0) {
                          delete fuse_counter;
                          done(s);
                          delete tag;
                        }
                      }
                    } else { // could not memcoy
                      // LOG(INFO) << "wrapper_done for fuse recv, could not memcpy, recv bytes: "
                      //        << tag->resp_tensor_bufs_[idx].len_
                      //        << ", DataType: " << response->GetDataTypeByIndex(idx);
                      ParseProtoUnlimited(&response->GetTensorProtoByIndex(idx),
                                          tag->resp_tensor_bufs_[idx].data_,
                                          tag->resp_tensor_bufs_[idx].len_);
                      Tensor val;
                      Status status = response->GetDevice()->MakeTensorFromProto(
                          response->GetTensorProtoByIndex(idx),
                          response->GetAllocAttributes(), &val);
                      CHECK(status.ok()) << "make cpu tensor from proto.";
                      response->SetTensorByIndex(idx, val);
                      if (__sync_sub_and_fetch(fuse_counter, 1) == 0) {
                        delete fuse_counter;
                        done(status);
                        delete tag;
                      }
                    }
                  } // end for cycle of the fuse count
                },
                std::move(done),
                std::placeholders::_1);

  tag->done_ = std::move(wrapper_done);
  tag->call_opts_ = call_opts;
  ProcessCallOptions(tag);
}

SeastarClientTag::SeastarClientTag(tensorflow::SeastarWorkerServiceMethod method,
                                   WorkerEnv* env,
                                   int fuse_count)
  : method_(method),
    env_(env),
    resp_err_msg_len_(0),
    fuse_count_(fuse_count),
    resp_message_bufs_(fuse_count),
    resp_tensor_bufs_(fuse_count),
    fail_fast_(false),
    timeout_in_ms_(0) {

  for (int idx = 0; idx < fuse_count_; ++idx) {
    resp_message_bufs_[idx].len_ = SeastarMessage::kMessageTotalBytes;
    resp_message_bufs_[idx].data_ = new char[resp_message_bufs_[idx].len_];
  }  
}

SeastarClientTag::~SeastarClientTag() {
  delete [] req_header_buf_.data_;
  delete [] req_body_buf_.data_;
  delete [] resp_body_buf_.data_;

  for (int i = 0; i < fuse_count_; ++i) {
    delete [] resp_message_bufs_[i].data_;

    if (resp_tensor_bufs_[i].owned_) {
      delete [] resp_tensor_bufs_[i].data_;
    }
  }
}

void SeastarClientTag::StartReq(seastar::channel* seastar_channel) {
  seastar_channel->put(ToUserPacket());
}

bool SeastarClientTag::IsRecvTensor() {
  return method_ == SeastarWorkerServiceMethod::kRecvTensor
    || method_ == SeastarWorkerServiceMethod::kFuseRecvTensor;
}

Status SeastarClientTag::ParseMessage(int idx) {
  return parse_message_(idx);
}

void SeastarClientTag::Schedule(std::function<void()> f) {
  env_->compute_pool->Schedule(std::move(f));
}

void SeastarClientTag::RecvRespDone(Status s) {
  Schedule([this, s]() {
    done_(s);
  });
}

// payload size for non-tensor response
uint64_t SeastarClientTag::GetResponseBodySize() {
  return resp_body_buf_.len_;
}

// payload buffer for non-tensor response
char* SeastarClientTag::GetResponseBodyBuffer() {
  return resp_body_buf_.data_;
}

// message size
uint64_t SeastarClientTag::GetResponseMessageSize(int idx) {
  return resp_message_bufs_[idx].len_;
}

// message buffer 
char* SeastarClientTag::GetResponseMessageBuffer(int idx) {
  return resp_message_bufs_[idx].data_;
}

// tensor size
uint64_t SeastarClientTag::GetResponseTensorSize(int idx) {
  return resp_tensor_bufs_[idx].len_;
}

// tensor buffer
char* SeastarClientTag::GetResponseTensorBuffer(int idx) {
  return resp_tensor_bufs_[idx].data_;
}

seastar::user_packet* SeastarClientTag::ToUserPacket() {
  seastar::net::fragment reqHeader {req_header_buf_.data_, req_header_buf_.len_};
  seastar::net::fragment reqBody {req_body_buf_.data_, req_body_buf_.len_};

  std::vector<seastar::net::fragment> frags = { reqHeader, reqBody };
  seastar::user_packet* up = new seastar::user_packet;
  up->_fragments = frags;
  up->_done = []{};
  return up;
}

} // namespace tensorflow
