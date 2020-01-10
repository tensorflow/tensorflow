#ifndef TENSORFLOW_FRAMEWORK_OP_SEGMENT_H_
#define TENSORFLOW_FRAMEWORK_OP_SEGMENT_H_

#include <string>
#include <unordered_map>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/public/status.h"

namespace tensorflow {

// OpSegment keeps track of OpKernels registered for sessions running
// on a device.
//
// The implementation maintains a two-level map. The 1st level maps
// session handle to the map of registered OpKernels. The 2nd level
// map maps node names to instantiated OpKernel objects.
//
// Each 2-nd level map is reference-counted and the caller can call
// AddHold to obtain a reference on all kernels of a session and
// ensure these kernels are alive until a corresponding RemoveHold is
// called on the same session.
class OpSegment {
 public:
  OpSegment();
  ~OpSegment();

  // A hold can be placed on a session, preventing all its kernels
  // from being deleted.
  void AddHold(const string& session_handle);
  void RemoveHold(const string& session_handle);

  // If the kernel for "node_name" has been created in the
  // "session_handle", returns the existing op kernel in "*kernel".
  // Otherwise, creates the kernel by calling create_fn(), cache it,
  // and returns it in "*kernel". If create_fn() fails, returns the
  // error.
  //
  // OpSegment keeps the ownership of the returned "*kernel".
  typedef std::function<Status(OpKernel**)> CreateKernelFn;
  Status FindOrCreate(const string& session_handle, const string& node_name,
                      OpKernel** kernel, CreateKernelFn create_fn);

 private:
  // op name -> OpKernel
  typedef std::unordered_map<string, OpKernel*> KernelMap;
  struct Item {
    int num_holds = 1;      // Num of holds put on the session.
    KernelMap name_kernel;  // op name -> kernel.
    ~Item();
  };

  // session handle -> item.
  // Session handles are produced by strings::FpToString()
  typedef std::unordered_map<string, Item*> SessionMap;

  mutable mutex mu_;
  SessionMap sessions_ GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(OpSegment);
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_OP_SEGMENT_H_
