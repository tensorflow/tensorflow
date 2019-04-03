#ifndef TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_WOKRER_INTERFACE_H_
#define TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_WOKRER_INTERFACE_H_

namespace tensorflow {
class CallOptions;
class RecvTensorRequest;
class SeastarTensorResponse;
class FuseRecvTensorRequest;
class SeastarFuseTensorResponse;

class SeastarWorkerInterface {
public:
  virtual void RecvTensorAsync(CallOptions* call_opts,
                               const RecvTensorRequest* request,
                               SeastarTensorResponse* response,
                               StatusCallback done) = 0;

  virtual void FuseRecvTensorAsync(CallOptions* call_opts,
                                   const FuseRecvTensorRequest* request,
                                   SeastarFuseTensorResponse* response,
                                   StatusCallback done) = 0;

};
} // namespace tensorflow

#endif //TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_WOKRER_INTERFACE_H_
