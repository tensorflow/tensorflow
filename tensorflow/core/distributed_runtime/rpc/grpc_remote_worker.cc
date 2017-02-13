/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/distributed_runtime/rpc/grpc_remote_worker.h"

#include "grpc++/grpc++.h"

#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_client_cq_tag.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_service_impl.h"
#include "tensorflow/core/distributed_runtime/tensor_coding.h"
#include "tensorflow/core/distributed_runtime/worker_cache_logger.h"
#include "tensorflow/core/distributed_runtime/worker_interface.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/protobuf/worker.pb.h"

#if 1

#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/lib/hash/hash.h"
#include <mpi.h>

#define MPICheck(cmd) do {                                 \
   int mpi_errno = cmd;                                          \
   if (MPI_SUCCESS != mpi_errno) {                               \
       fprintf(stderr, "[%s:%d] MPI call failed with %d \n",     \
        __FILE__, __LINE__,mpi_errno);                           \
       exit(EXIT_FAILURE);                                       \
   }                                                             \
   assert(MPI_SUCCESS == mpi_errno);                             \
   } while(false)

#endif





namespace tensorflow {

class GrpcRemoteWorker : public WorkerInterface {
 public:
  explicit GrpcRemoteWorker(SharedGrpcChannelPtr channel,
                            ::grpc::CompletionQueue* completion_queue,
                            WorkerCacheLogger* logger)
      : channel_(channel),
        cq_(completion_queue),
        getstatus_(Method(GrpcWorkerMethod::kGetStatus)),
        registergraph_(Method(GrpcWorkerMethod::kRegisterGraph)),
        deregistergraph_(Method(GrpcWorkerMethod::kDeregisterGraph)),
        rungraph_(Method(GrpcWorkerMethod::kRunGraph)),
        cleanupgraph_(Method(GrpcWorkerMethod::kCleanupGraph)),
        cleanupall_(Method(GrpcWorkerMethod::kCleanupAll)),
        recvtensor_(Method(GrpcWorkerMethod::kRecvTensor)),
        logging_(Method(GrpcWorkerMethod::kLogging)),
        tracing_(Method(GrpcWorkerMethod::kTracing)),
        logger_(logger) {}

  ~GrpcRemoteWorker() override {}

  void GetStatusAsync(const GetStatusRequest* request,
                      GetStatusResponse* response,
                      StatusCallback done) override {
    IssueRequest(request, response, getstatus_, std::move(done));
  }

  void RegisterGraphAsync(const RegisterGraphRequest* request,
                          RegisterGraphResponse* response,
                          StatusCallback done) override {
    IssueRequest(request, response, registergraph_, std::move(done));
  }

  void DeregisterGraphAsync(const DeregisterGraphRequest* request,
                            DeregisterGraphResponse* response,
                            StatusCallback done) override {
    IssueRequest(request, response, deregistergraph_, std::move(done));
  }

  void RunGraphAsync(CallOptions* call_opts, const RunGraphRequest* request,
                     RunGraphResponse* response, StatusCallback done) override {
    IssueRequest(request, response, rungraph_, std::move(done), call_opts);
  }
  void RunGraphAsync(CallOptions* call_opts, RunGraphRequestWrapper* request,
                     MutableRunGraphResponseWrapper* response,
                     StatusCallback done) override {
    IssueRequest(&request->ToProto(), get_proto_from_wrapper(response),
                 rungraph_, std::move(done), call_opts);
  }

  void CleanupGraphAsync(const CleanupGraphRequest* request,
                         CleanupGraphResponse* response,
                         StatusCallback done) override {
    IssueRequest(request, response, cleanupgraph_, std::move(done));
  }

  void CleanupAllAsync(const CleanupAllRequest* request,
                       CleanupAllResponse* response,
                       StatusCallback done) override {
    IssueRequest(request, response, cleanupall_, std::move(done));
  }



  void SendTensorSync(const WorkerEnv* env,
                      const Rendezvous::ParsedKey& key,
                      const Rendezvous::Args &args,
                      const Tensor& val,
                      const bool is_dead,
                      Status &s)  {
    s =  Status(tensorflow::error::UNIMPLEMENTED, "SendTensorSync()");
  }



  void RecvTensorAsync(WorkerEnv* env, CallOptions* call_opts, const RecvTensorRequest* request,
                       TensorResponse* response, StatusCallback done) override {
    VLOG(1) << "RecvTensorAsync req: " << request->DebugString();
    int64 start_usec = Env::Default()->NowMicros();
    // Don't propagate dma_ok over gRPC.
    RecvTensorRequest* req_copy = nullptr;
    if (request->dma_ok()) {
      req_copy = new RecvTensorRequest;
      *req_copy = *request;
      req_copy->set_dma_ok(false);
    }
    // Type-specialized logging for this method.
    bool logging_active = logger_->LoggingActive() || VLOG_IS_ON(2);
    StatusCallback wrapper_done;
    const StatusCallback* cb_to_use;
    if (!logging_active && req_copy == nullptr) {
      cb_to_use = &done;  // No additional work to do, so just use done directly
    } else if (!logging_active) {
      wrapper_done = [req_copy, done](Status s) {
        delete req_copy;
        done(s);
      };
      cb_to_use = &wrapper_done;
    } else {
      wrapper_done = [this, request, req_copy, response, done,
                      start_usec](Status s) {
        if (logger_->LoggingActive()) {
          int64 end_usec = Env::Default()->NowMicros();
          int64 step_id = request->step_id();
          int64 bytes = response->tensor().TotalBytes();
          int64 send_start_usec = start_usec;
          // If a send start time was reported by the other side, use
          // that instead.  Maybe we should mark the display if we're using
          // our local time instead of the remote start time?
          if (response->metadata().send_start_micros()) {
            // send_start_micros is the timestamp taken when the
            // remote machine began to send the RecvTensor response.
            // Due to clock skew between source and dest machines, it
            // is possible that send_start_micros can be larger than
            // end_usec or less than start_usec.
            //
            // To respect causality, we enforce the invariants that
            // the RecvTensor response can not have been sent before
            // the RecvTensor request, and must have been sent before
            // it was received.
            send_start_usec = std::max(start_usec, static_cast<int64>(
                response->metadata().send_start_micros()));
            send_start_usec = std::min(send_start_usec, end_usec - 1);
          }
          const string& key = request->rendezvous_key();
          std::vector<string> key_parts = str_util::Split(key, ';');
          if (key_parts.size() != 5) {
            LOG(WARNING) << "Bad key: " << key;
          } else {
            logger_->RecordRecvTensor(step_id, send_start_usec, end_usec,
                                      key_parts[3],  // tensor name
                                      key_parts[0],  // src_device
                                      key_parts[2],  // dst_device
                                      bytes);
          }
        }
        VLOG(2) << "done callback, req: " << request->DebugString()
                << " response " << response->metadata().DebugString();
        delete req_copy;
        done(s);
      };
      cb_to_use = &wrapper_done;
    }

    IssueRequest(req_copy ? req_copy : request, response, recvtensor_,
                 std::move(*cb_to_use), call_opts);
  }



  void LoggingAsync(const LoggingRequest* request, LoggingResponse* response,
                    StatusCallback done) override {
    IssueRequest(request, response, logging_, done);
  }

  void TracingAsync(const TracingRequest* request, TracingResponse* response,
                    StatusCallback done) override {
    IssueRequest(request, response, tracing_, done);
  }

 private:
  // Object allocated per active RPC.
  template <class RequestMessage, class ResponseMessage>
  class RPCState final : public GrpcClientCQTag {
   public:
    RPCState(::grpc::ChannelInterface* channel, ::grpc::CompletionQueue* cq,
             const ::grpc::RpcMethod& method, const RequestMessage& request,
             StatusCallback done, CallOptions* call_opts)
        : call_opts_(call_opts),
          reader_(channel, cq, method, InitContext(call_opts), request),
          done_(std::move(done)) {}

    ~RPCState() override {}

    void StartRPC(ResponseMessage* response) {
      reader_.Finish(response, &status_, this);
    }

    void OnCompleted(bool ok) override {
      if (!ok) {
        VLOG(2) << "Call returned with non-ok status: "
                << status_.error_message();
      }
      if (call_opts_) {
        call_opts_->ClearCancelCallback();
      }
      done_(FromGrpcStatus(status_));
      delete this;
    }

   private:
    CallOptions* call_opts_;
    ::grpc::ClientContext context_;
    ::grpc::ClientAsyncResponseReader<ResponseMessage> reader_;
    ::grpc::Status status_;
    StatusCallback done_;

    ::grpc::ClientContext* InitContext(CallOptions* call_opts) {
      // The initialization and recovery protocols rely on blocking
      // until we get a response.
      context_.set_fail_fast(false);
      if (call_opts) {
        call_opts->SetCancelCallback([this]() { context_.TryCancel(); });
      }
      return &context_;
    }
  };

  // Utility method for issuing a generic asynchronous request. The
  // given callback, `done`, will be called when the RPC completes.
  template <class RequestMessage, class ResponseMessage>
  void IssueRequest(const RequestMessage* request, ResponseMessage* response,
                    const ::grpc::RpcMethod& method, StatusCallback done,
                    CallOptions* call_opts = nullptr) {
    auto state = new RPCState<RequestMessage, ResponseMessage>(
        channel_.get(), cq_, method, *request, std::move(done), call_opts);
    state->StartRPC(response);
  }

  // Helper function for initializing the RpcMethod objects below.
  ::grpc::RpcMethod Method(GrpcWorkerMethod id) {
    return ::grpc::RpcMethod(GrpcWorkerMethodName(id),
                             ::grpc::RpcMethod::NORMAL_RPC, channel_);
  }

  SharedGrpcChannelPtr channel_;
  ::grpc::CompletionQueue* cq_;

  const ::grpc::RpcMethod getstatus_;
  const ::grpc::RpcMethod registergraph_;
  const ::grpc::RpcMethod deregistergraph_;
  const ::grpc::RpcMethod rungraph_;
  const ::grpc::RpcMethod cleanupgraph_;
  const ::grpc::RpcMethod cleanupall_;
  const ::grpc::RpcMethod recvtensor_;
  const ::grpc::RpcMethod logging_;
  const ::grpc::RpcMethod tracing_;

  // Support for logging.
  WorkerCacheLogger* logger_;

  TF_DISALLOW_COPY_AND_ASSIGN(GrpcRemoteWorker);
};

#if 0
WorkerInterface* NewGrpcRemoteWorker(SharedGrpcChannelPtr channel,
                                     ::grpc::CompletionQueue* completion_queue,
                                     WorkerCacheLogger* logger) {
  return new GrpcRemoteWorker(channel, completion_queue, logger);
}
#endif

class RDMARemoteWorker : public GrpcRemoteWorker {

private:
  
  //Returns the name of the desitnation specified in a rendezvous key
  //For idx=0 it is the source, for idx=2 it is the destination
  string GetWorkerName(const std::string &key, const int idx)
  {
      //Convert the key back to the subpieces
      const std::vector<string> num_strings = str_util::Split(key, ';');
      //Sanity check, should be 5 src;id;dst;name;frame_iter
      assert(num_strings.size() == 5); 
      //Strip the device eg /cpu:0 to get the worker name
      return num_strings[idx].substr(0, num_strings[idx].find_last_of('/'));
  }
  string SourceWorker     (const std::string &key) { return GetWorkerName(key,0); }
  string DestinationWorker(const std::string &key) { return GetWorkerName(key,2); }


  const int getMPIPartnerID(const bool targetIsSource, 
                            std::string key,
                            const WorkerEnv* env)
  {
    //Convert the grpc-name to MPI process ID
    std::string name;
    if(targetIsSource)  name = SourceWorker     (key);
    else                name = DestinationWorker(key);
    auto       it       = env->worker_name_MPI_idx.find(name);
    if(it == env->worker_name_MPI_idx.end())
    {   
        LOG(FATAL) << "Failed to convert worker name to MPI index: " << name;
    }   
    return it->second;
  }

  const int tensorHash(const std::string key)
  {
    const uint32  hash32    = Hash32(key.data(), key.size(), 20161211);
    const int tensorKeyHash = std::abs(static_cast<int>(hash32));
    return (tensorKeyHash % maxMessageTag);
  }

  void MPISendTensor(const int dst,      const int hash, 
                     const bool is_dead, const Tensor &val)
  {
#if 1
    //Send a header using the 'hash', followed by a message identified by
    //the followUpTag, which is unique to this process
    const int followUpTag = std::abs(static_cast<int>(pthread_self()));

    //Encode the properties of the tensor and send this to the destination
    RecvTensorResponse response;
    response.set_is_dead(is_dead);
    response.set_send_start_micros(Env::Default()->NowMicros());
    response.mutable_tensor()->set_dtype(val.dtype());
    val.shape().AsProto(response.mutable_tensor()->mutable_tensor_shape());
    std::vector<char> respBuff(response.ByteSize() + sizeof(int));
    response.SerializeToArray(&respBuff[4], respBuff.size() - sizeof(int));

    //fprintf(stderr, "JBDBG Going to send the following message hash: %d tag: %d\n", hash, followUpTag);

    ((int*)(&respBuff[0]))[0] = followUpTag;

    MPICheck(MPI_Send(&respBuff[0], respBuff.size(), MPI_BYTE, dst, hash, MPI_COMM_WORLD));

    //Next transfer the actual Tensor data
    const size_t nBytes = val.TotalBytes();
    const char    *data = val.tensor_data().data();

    //frintf(stderr, "JBDBG Going to send this data: %p bytes: %d hash: %d tag: %d\n", data, nBytes, hash, followUpTag);

    //Transfer the Tensor content, in maxSize chuncks
    for(size_t i=0; i <= nBytes; i+= maxSize)
    {
      //Use MPI_Ssend to prevent that the function returns before the message has arrived. This would cause inconsistencies in the execution and crashes.
      const int toSend = std::min(maxSize, nBytes - i);
      MPICheck(MPI_Ssend(&data[i], toSend, MPI_BYTE, dst, followUpTag, MPI_COMM_WORLD));
    }
#endif
  } //MPISendTensor



  void MPIRecvTensor(const int src, const int hash, TensorResponse *response)
  {
#if 1
    MPI_Status status;
    MPI_Message msg;

    //Receive the header message, probe as size is variable
    int incSize;
    MPICheck(MPI_Mprobe(src, hash, MPI_COMM_WORLD, &msg, &status));
    MPICheck(MPI_Get_count(&status, MPI_CHAR, &incSize));
    std::vector<char> sta(incSize);
    MPICheck(MPI_Mrecv(&sta[0], incSize, MPI_CHAR, &msg, &status));

    const int followUpTag = *((int*)(&sta[0]));
    RecvTensorResponse RTresponse;
    //std::cerr << response->DebugString() << std::endl;
    RTresponse.ParseFromArray(&sta[4], sta.size()-sizeof(int));
    //std::cerr << RTresponse.DebugString() << std::endl;

    //Initialize the destination tensor
    //TODO is it possible that this tensor already exists/reuse of memory location?
    response->InitPartial(RTresponse);
    const size_t nBytes = response->tensor().TotalBytes();

    //Receive the Tensor content
    char *data = const_cast<char*>(response->tensor().tensor_data().data());
    for(size_t i=0; i <= nBytes; i+= maxSize)
    {
        const int toRecv = std::min(maxSize, nBytes - i);
        MPICheck(MPI_Recv(&data[i], toRecv, MPI_BYTE, src,
                          followUpTag, MPI_COMM_WORLD, &status));
    }
#endif
  } //MPIRecvTensor


  //Different MPI implementations use different values for allowed tag ranges
  //so retrieve the max value to generate valid tags
  int maxMessageTag;

  //Max size of the data chuncks, 512MB
  const size_t maxSize = 1024*1024*512;


 public:
  explicit RDMARemoteWorker(SharedGrpcChannelPtr channel,
                            ::grpc::CompletionQueue* completion_queue,
                            WorkerCacheLogger* logger) : GrpcRemoteWorker(channel, completion_queue, logger)
  {
       //Determine the maximum allowed message tag, used to determine hash
       void *v;
       int flag;
       MPICheck(MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &v, &flag));
       maxMessageTag =  *(int*)v;
  }


  void SendTensorSync(const WorkerEnv* env,
                      const Rendezvous::ParsedKey& key,
                      const Rendezvous::Args &args,
                      const Tensor& val,
                      const bool is_dead,
                      Status &s) override
  {
    const int dst  = getMPIPartnerID(false, key.FullKey().ToString(), env);
    const int hash = tensorHash(key.FullKey().ToString());
    //fprintf(stderr, "JBDBG Going to send data to: %s  : %d  || %d  devContext: %d\n", key.FullKey().ToString().c_str(), dst, hash, args.device_context); 
    MPISendTensor(dst, hash, is_dead, val);

    s = Status::OK();

  }



  void RecvTensorAsync(WorkerEnv* env, CallOptions* call_opts, const RecvTensorRequest* request,
                       TensorResponse* response, StatusCallback done) override {
    #if 0
                  GrpcRemoteWorker::RecvTensorAsync(env, call_opts, request, response, done);
    #else
    //TODO: Figure out how to get the size of the requested Tensor
    //Is this known by looking up the key/request?
    //TODO is it possible that a tensor has been pre-allocated?
    //so we can reuse the same (mapped) pointer?
    //fprintf(stderr, "JBDBG TensorResponse numElem: %ld ", response->tensor().NumElements());

    const int src  = getMPIPartnerID(true, request->rendezvous_key(), env);
    //fprintf(stderr, "JBDBG Going to receive data from: %s  : %d \n", request->rendezvous_key().c_str(), src);
    MPIRecvTensor(src, tensorHash(request->rendezvous_key()), response);

    //   response->ClearTensor();  //Reset the receive tensor, invalidates all the above received data :)
    done(Status::OK());
#endif
 } //RecvTensorAsync


}; //class RDMARemoteWorker

WorkerInterface* NewGrpcRemoteWorker(SharedGrpcChannelPtr channel,
                                     ::grpc::CompletionQueue* completion_queue,
                                     WorkerCacheLogger* logger) {
  //return new GrpcRemoteWorker(channel, completion_queue, logger);
  return new RDMARemoteWorker(channel, completion_queue, logger);
}





}  // namespace tensorflow
