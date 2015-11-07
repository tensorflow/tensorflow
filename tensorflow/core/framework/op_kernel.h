#ifndef TENSORFLOW_FRAMEWORK_OP_KERNEL_H_
#define TENSORFLOW_FRAMEWORK_OP_KERNEL_H_

#include <functional>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/control_flow.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/kernel_def.pb.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/tracking_allocator.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/public/env.h"
#include "tensorflow/core/public/status.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/public/tensor_shape.h"

namespace Eigen {
class ThreadPoolDevice;
class GpuDevice;
}  // end namespace Eigen

namespace tensorflow {

namespace checkpoint {
class TensorSliceReaderCacheWrapper;
}  // namespace checkpoint

class AsyncOpKernel;
class OpKernelConstruction;  // declared below
class OpKernelContext;       // declared below
class ResourceMgr;

// TODO(josh11b): Make reference-counted if needed.
class OpKernel {
 public:
  // OpKernel won't be instantiated by the scheduler, so you may perform
  // expensive initialization in the descendant's constructor.
  explicit OpKernel(OpKernelConstruction* context);
  virtual ~OpKernel() {}

  // An OpKernel's computation can be either synchronous or
  // asynchronous.
  //
  // Most OpKernels should compute synchronously.  They should
  // subclass OpKernel and override the Compute() method and have it
  // return after completing the supplied work.
  //
  // A few special kernels might need to be asynchronous to bound the
  // number of threads (e.g., network receive operations). These
  // kernels must subclass AsyncOpKernel and override
  // AsyncOpKernel::ComputeAsync().
  //
  // In both cases, implementations of Compute() and ComputeAsync()
  // get inputs and write outputs through the given OpKernelContext
  // and returns a status via context->SetStatus(). They must be
  // thread-safe.

  // Synchronous compute.
  //
  // "context" is guaranteed to be alive until Compute() returns.
  virtual void Compute(OpKernelContext* context) = 0;

  // Returns nullptr iff this op kernel is synchronous.
  virtual AsyncOpKernel* AsAsync() { return nullptr; }

  // Returns true iff this op kernel is considered "expensive". The
  // runtime may use this flag to optimize graph execution for example
  // to "inline" inexpensive kernels.
  virtual bool IsExpensive() { return true; }

  // Accessors.
  const NodeDef& def() const { return def_; }
  const string& name() const { return def_.name(); }
  const string& type_string() const { return def_.op(); }

  int num_inputs() const { return input_types_.size(); }
  DataType input_type(int i) const { return input_types_[i]; }
  const DataTypeVector& input_types() const { return input_types_; }
  const MemoryTypeVector& input_memory_types() const {
    return input_memory_types_;
  }

  int num_outputs() const { return output_types_.size(); }
  DataType output_type(int o) const { return output_types_[o]; }
  const DataTypeVector& output_types() const { return output_types_; }
  const MemoryTypeVector& output_memory_types() const {
    return output_memory_types_;
  }

  Status InputRange(const string& input_name, int* start, int* stop) const;
  Status OutputRange(const string& output_name, int* start, int* stop) const;

 private:
  const NodeDef def_;
  const DataTypeVector input_types_;
  const DataTypeVector output_types_;
  NameRangeMap input_name_map_;
  NameRangeMap output_name_map_;
  MemoryTypeVector input_memory_types_;
  MemoryTypeVector output_memory_types_;

  TF_DISALLOW_COPY_AND_ASSIGN(OpKernel);
};

class AsyncOpKernel : public OpKernel {
 public:
  using OpKernel::OpKernel;  // Lift OpKernel constructors.

  // Asynchronous compute.
  //
  // Implementations of ComputeAsync() must run "done" to signal the
  // completion of the computation. "context" is guaranteed to be
  // alive until the "done" callback starts.
  typedef std::function<void()> DoneCallback;
  virtual void ComputeAsync(OpKernelContext* context, DoneCallback done) = 0;

  AsyncOpKernel* AsAsync() final { return this; }

  void Compute(OpKernelContext* context) final;
};

// Wraps a tensor that is held by an Op across calls to Compute(). For
// memory safety when using asynchronous devices like GPUs, the system
// must be notified when a Tensor is used inside an Op execution. The
// wrapper ensures that all uses of the Tensor are tracked, because in
// order to retrieve the Tensor the caller must use AccessTensor which
// notifies the context.
class PersistentTensor {
 public:
  PersistentTensor() {}
  explicit PersistentTensor(const Tensor& tensor) : tensor_(tensor) {}

  // Caller does not own the returned Tensor*.
  Tensor* AccessTensor(OpKernelConstruction* context);
  // Caller does not own the returned Tensor*.
  Tensor* AccessTensor(OpKernelContext* context);

  // The check for initialization does not need to access the
  // underlying tensor buffer.
  bool IsInitialized() { return tensor_.IsInitialized(); }

 private:
  Tensor tensor_;
};

class OpKernelConstruction {
 public:
  // TODO(yuanbyu): Probably reduce the number of arguments.
  OpKernelConstruction(DeviceType device_type, DeviceBase* device,
                       Allocator* allocator, const NodeDef* node_def,
                       const OpDef* op_def, FunctionLibraryRuntime* flib,
                       const DataTypeSlice& input_types,
                       const DataTypeSlice& output_types, Status* status)
      : device_type_(device_type),
        device_(device),
        allocator_(allocator),
        def_(node_def),
        op_def_(op_def),
        flib_(flib),
        input_types_(input_types),
        output_types_(output_types),
        status_(status) {}

  Env* env() const { return device_->env(); }

  // Allocation of tensors during kernel construction:
  //
  // It is legal to temporarily allocate scratch tensor storage during
  // Op kernel construction. Scratch tensors should be allocated using
  // allocate_temp below. Some kernels need to keep tensors in between
  // invocations. If such a Tensor is allocated during kernel
  // construction this must be done using allocate_persistent, and the
  // Op may only store the returned PersistentTensor object. When the
  // Tensor is needed in a subsequent invocation, it can be retrieved
  // from the PersistentTensor using the AccessTensor method. This
  // ensures that the system is made aware of any use of the tensor's
  // allocated memory, which is needed for correctness on asynchronous
  // devices such as GPUs.

  // Allocates a temporary Tensor of the specified type and shape. The
  // Tensor must not be used after kernel construction is
  // complete. See comment above.
  Status allocate_temp(DataType type, const TensorShape& shape,
                       Tensor* out_temp);

  // Allocates a Tensor of the specified type and shape which the Op
  // plans to maintain as persistent state. out_persistent holds the
  // PersistentTensor which is the object the caller should store. For
  // convenience, if out_tensor is non-null then it will be filled in
  // with a Tensor* pointing to the newly-allocated tensor which the
  // caller can use instead of calling
  // out_persistent->AccessTensor. The caller does not own out_tensor
  // and should not keep a copy of it. See comment above.
  Status allocate_persistent(DataType type, const TensorShape& shape,
                             PersistentTensor* out_persistent,
                             Tensor** out_tensor);

  // User-supplied configuration of this operation.
  const NodeDef& def() const { return *def_; }

  // Op registered for this op type.
  const OpDef& op_def() const { return *op_def_; }

  // For inspecting the inputs to this operation.
  int num_inputs() const { return input_types_.size(); }
  DataType input_type(int i) const { return input_types_[i]; }
  const DataTypeSlice& input_types() const { return input_types_; }

  // For inspecting the outputs expected from this operation.
  int num_outputs() const { return output_types_.size(); }
  DataType output_type(int i) const { return output_types_[i]; }
  const DataTypeSlice& output_types() const { return output_types_; }

  // If expected_inputs == inputs() and expected_outputs == output_types(),
  // returns OK, else returns INVALID_ARGUMENT with an error message.
  // Recommended for Ops with dynamic signatures.
  Status MatchSignature(const DataTypeSlice expected_inputs,
                        const DataTypeSlice expected_outputs);

  // For recording configuration errors during construction.
  void SetStatus(const Status& status) { status_->Update(status); }
  const Status& status() const { return *status_; }

  // Look up the attr with name attr_name and set *value to its value.  If no
  // attr with attr_name is found in def(), or the attr does not have
  // a matching type, a non-ok status will be returned.
  template <class T>
  Status GetAttr(const string& attr_name, T* value) const {
    return GetNodeAttr(def(), attr_name, value);
  }

  // May be used, e.g., to get GPU handles, etc.
  // TODO(tucker): Add example usage.
  DeviceBase* device() const { return device_; }

  // Return the device type.
  const DeviceType& device_type() const { return device_type_; }

  // If not nullptr, the kernel can instantiate functions defined in
  // the library. E.g.,
  // CHECK_NOTNULL(function_library())->Instantiate("Foo", ...).
  FunctionLibraryRuntime* function_library() const { return flib_; }

 private:
  const DeviceType device_type_;
  DeviceBase* const device_;
  Allocator* allocator_;
  const NodeDef* def_;
  const OpDef* op_def_;
  FunctionLibraryRuntime* flib_;
  DataTypeSlice input_types_;
  DataTypeSlice output_types_;
  Status* status_;

  TF_DISALLOW_COPY_AND_ASSIGN(OpKernelConstruction);
};

// TODO(mrry): Consider converting to a random_access_iterator, and upgrading
// tensorflow::gtl::iterator_range to make the below container classes
// unnecessary.
template <typename ListType, typename ElementType>
class OpArgIterator {
 public:
  typedef OpArgIterator<ListType, ElementType> ME;
  OpArgIterator(const ListType* list, int i) : list_(list), i_(i) {}
  bool operator==(const ME& rhs) {
    DCHECK(list_ == rhs.list_);
    return i_ == rhs.i_;
  }
  bool operator!=(const ME& rhs) {
    DCHECK(list_ == rhs.list_);
    return i_ != rhs.i_;
  }
  void operator++() { ++i_; }
  ElementType& operator*() { return (*list_)[i_]; }

 private:
  const ListType* const list_;
  int i_;
};

// Utility class for representing a list of immutable input tensors
// that are passed to the op as a single named argument.
class OpInputList {
 public:
  typedef OpArgIterator<OpInputList, const Tensor&> Iterator;
  OpInputList() : ctx_(nullptr), start_(0), stop_(0) {}
  OpInputList(const OpKernelContext* ctx, int start, int stop)
      : ctx_(ctx), start_(start), stop_(stop) {}
  OpInputList& operator=(const OpInputList& other) = default;
  const Tensor& operator[](int i) const;
  int size() const { return stop_ - start_; }
  Iterator begin() const { return Iterator(this, 0); }
  Iterator end() const { return Iterator(this, size()); }

 private:
  const OpKernelContext* ctx_;  // not owned
  int start_;
  int stop_;
};

// Utility class for representing a list of mutable ("ref") input tensors
// that are passed to the op as a single named argument.
class OpMutableInputList {
 public:
  typedef OpArgIterator<OpMutableInputList, Tensor*> Iterator;
  OpMutableInputList(OpKernelContext* ctx, int start, int stop)
      : ctx_(ctx), start_(start), stop_(stop) {}
  OpMutableInputList() : ctx_(nullptr), start_(0), stop_(0) {}
  OpMutableInputList& operator=(const OpMutableInputList& other) = default;
  Tensor at(int i, bool lock_held);
  mutex* ref_mutex(int i);
  int size() const { return stop_ - start_; }
  Iterator begin() const { return Iterator(this, 0); }
  Iterator end() const { return Iterator(this, size()); }

 private:
  OpKernelContext* ctx_;  // not owned
  int start_;
  int stop_;
};

// Utility class for representing a list of output tensors that are
// grouped as a single named output.
class OpOutputList {
 public:
  typedef OpArgIterator<OpOutputList, const Tensor*> Iterator;
  OpOutputList() : ctx_(nullptr), start_(0), stop_(0) {}
  OpOutputList(OpKernelContext* ctx, int start, int stop)
      : ctx_(ctx), start_(start), stop_(stop) {}
  OpOutputList& operator=(const OpOutputList& other) = default;
  Tensor* operator[](int i);
  bool required(int i) const;
  Status allocate(int i, const TensorShape& shape, Tensor** output);
  void set(int i, const Tensor& tensor);
  void set_ref(int i, mutex* mu, Tensor* tensor_for_ref);
  int size() const { return stop_ - start_; }
  Iterator begin() const { return Iterator(this, 0); }
  Iterator end() const { return Iterator(this, size()); }

 private:
  OpKernelContext* ctx_;  // not owned
  int start_;
  int stop_;
};

// Holds a tensor or tensor reference. For tensor references, we need
// a mutex to prevent concurrent access to the tensor.
struct TensorValue {
  TensorValue() : mutex_if_ref(nullptr), tensor(nullptr) {}
  TensorValue(Tensor* t)  // NOLINT(runtime/explicit)
      : mutex_if_ref(nullptr),
        tensor(t) {}
  TensorValue(mutex* mu, Tensor* t) : mutex_if_ref(mu), tensor(t) {}
  Tensor* operator->() const { return tensor; }
  bool is_ref() const { return mutex_if_ref != nullptr; }

  mutex* mutex_if_ref;  // nullptr if not a ref, != nullptr if a ref
  Tensor* tensor;
};

class OpKernelContext {
 public:
  // The first element of a WrappedAllocator is a "base" Allocator and
  // the second element is that Allocator wrapped by a
  // TrackingAllocator
  typedef std::pair<Allocator*, TrackingAllocator*> WrappedAllocator;

  // TODO(zhifengc): Do some cleanup of Params.
  struct Params {
    // The op kernel being computed.
    OpKernel* op_kernel = nullptr;

    // The device on which the kernel is running.
    DeviceBase* device = nullptr;

    bool track_allocations = false;
    std::function<AllocatorAttributes(int index)> output_alloc_attr = nullptr;

    // Shared resources accessible by this op kernel invocation.
    ResourceMgr* resource_manager = nullptr;

    // Per-step resources accessible by this op kernel invocation.
    ResourceMgr* step_resource_manager = nullptr;

    // Mechanism used by this op kernel invocation to communicate with
    // computations running on other devices.
    Rendezvous* rendezvous = nullptr;

    // Mechanism used by this op kernel invocation to register a callback
    // for its cancellation.
    CancellationManager* cancellation_manager = nullptr;

    // Inputs to this op kernel.
    const gtl::InlinedVector<TensorValue, 4>* inputs = nullptr;
    bool is_input_dead = false;

    const gtl::InlinedVector<AllocatorAttributes, 4>* input_alloc_attrs =
        nullptr;

    // Device contexts.
    const gtl::InlinedVector<DeviceContext*, 4>* input_device_contexts =
        nullptr;
    DeviceContext* op_device_context = nullptr;

    // Control-flow op supports.
    FrameAndIter frame_iter;

    // Function call supports.
    FunctionCallFrame* call_frame = nullptr;
    FunctionLibraryRuntime* function_library = nullptr;

    // TensorSliceReaderCache support.
    checkpoint::TensorSliceReaderCacheWrapper* slice_reader_cache = nullptr;
  };
  explicit OpKernelContext(const Params& params);
  ~OpKernelContext();

  Env* env() const { return params_.device->env(); }

  // Input/output signature.

  int num_inputs() const { return params_.inputs->size(); }
  DataType input_dtype(int index) const;
  int num_outputs() const { return outputs_.size(); }
  DataType expected_output_dtype(int index) const;

  // Input

  // Returns an immutable input tensor. May only be used for non-Ref
  // inputs. For Ref inputs use mutable_input below.
  // REQUIRES: !IsRefType(input_dtype(index))
  // TODO(mrry): Convert this to return Status.
  const Tensor& input(int index) const;

  // Returns the named immutable input tensor in "tensor", as defined
  // in the OpDef. May only be used for non-Ref inputs. For Ref inputs
  // use mutable_input below.
  // REQUIRES: !IsRefType(input_dtype(index))
  // REQUIRES: the named input must not be a list.
  Status input(const string& name, const Tensor** tensor) const;

  // Returns the named list-valued immutable input in "list", as
  // defined in the OpDef.  If the named output is not list-valued,
  // returns a one-element list. May only be used for non-Ref
  // inputs. For Ref inputs use mutable_input below.
  // REQUIRES: !IsRefType(input_dtype(index))
  Status input_list(const string& name, OpInputList* list) const;

  // For mutable inputs, use the following together to make sure there
  // is no concurrent access to mutable_input(), e.g.:
  // {
  //   Tensor& t = context->mutable_input(index);
  //   mutex_lock lock(*context->input_ref_mutex(index));
  //   // modify the values in t
  // }
  // REQUIRES: IsRefType(input_dtype(index))
  // TODO(mrry): Convert this to return Status.
  mutex* input_ref_mutex(int index);
  Status input_ref_mutex(const string& name, mutex** out_mutex);

  // Returns a mutable input tensor. Must be used to access Ref
  // inputs.  REQUIRES: IsRefType(input_dtype(index)). The caller may
  // modify the values stored in the Tensor buffer, and modifications
  // will be visible to other Ops reading the same ref tensor. If
  // !lock_held the input mutex will be acquired before returning the
  // Tensor.
  // TODO(mrry):
  // Convert this to return Status.
  Tensor mutable_input(int index, bool lock_held);

  // Returns the named mutable input tensor in "tensor", as defined in
  // the OpDef. Must be used to access Ref inputs. The values stored
  // in the Tensor buffer may be modified, and modifications will be
  // visible to other Ops reading the same ref tensor. If !lock_held
  // the input mutex will be acquired before returning the Tensor.
  // REQUIRES: the named input must not be a list.
  // REQUIRES: the named input must be a ref tensor.
  Status mutable_input(const string& name, Tensor* tensor, bool lock_held);

  // Returns the named list-valued mutable input in "list", as defined
  // in the OpDef.  If the named intput is not list-valued, returns a
  // one-element list. Must be used to access Ref inputs. The values
  // stored in the Tensor buffer may be modified, and modifications
  // will be visible to other Ops reading the same ref tensor.
  // REQUIRES: the named input must be a ref tensor.
  Status mutable_input_list(const string& name, OpMutableInputList* list);

  // Replace the corresponding Ref Input to use the storage buffer
  // used by tensor. If !lock_held the input mutex will be acquired
  // before returning the Tensor.
  // REQUIRES: IsRefType(input_dtype(index)).
  void replace_ref_input(int index, const Tensor& tensor, bool lock_held);

  // Replace the corresponding named Ref Input to use the storage
  // buffer used by tensor. If !lock_held the input mutex will be
  // acquired before returning the Tensor.
  // REQUIRES: IsRefType(input_dtype(index)).
  Status replace_ref_input(const string& name, const Tensor& tensor,
                           bool lock_held);

  // Set the output Ref Tensor at output_index to be an alias of the
  // input Ref Tensor at input_index.
  // REQUIRES: IsRefType(input_dtype(input_index)).
  // REQUIRES: IsRefType(output_dtype(output_index)).
  void forward_ref_input_to_ref_output(int input_index, int output_index);

  // Deletes the Tensor object used as the Ref Input at
  // input_index. This is not usually necessary and should be used
  // with caution. If !lock_held the input mutex will be acquired
  // before returning the Tensor.
  // REQUIRES: IsRefType(input_dtype(input_index)).
  void delete_ref_input(int input_index, bool lock_held);

  // Return true if there is input at the given index. An operator has no
  // input at index if its tensor is null. This is primarily used by the
  // merge operator.
  // TODO(mrry): Convert this to return Status.
  bool has_input(int index) const;

  // Returns true if all inputs are the same shape, otherwise sets the
  // status to a non-OK value and returns false.
  // Usage: if (!context->ValidateInputsAreSameShape(this)) return;
  bool ValidateInputsAreSameShape(OpKernel* op);

  // Output

  // Returns the named list-valued output in "list", as defined in the OpDef.
  // If the named output is not list-valued, returns a one-element list.
  Status output_list(const string& name, OpOutputList* list);

  // If output_required(index) returns true, the OpKernel's Compute() method
  // should call allocate_output(index, ...), set_output(index, ...),
  // set_output_ref(index, ...), or set the status to a non-ok value.
  // If it returns false, it may output, but is not required to do so.
  // TODO(mrry): Convert this to return Status, and implement a string
  // name version.
  bool output_required(int index) const {
    return true;  // TODO(josh11b): implement
  }

  // Allocation of tensors during kernel execution inside the Compute
  // method:
  //
  // There are three methods to allocate Tensors when an Op kernel
  // executes.
  //
  // 1) allocate_persistent. This is only needed for Tensors that will
  // be stored by the Op between invocations, and it *must* be used
  // for those Tensors. The call returns a PersistentTensor, and that
  // is the only object the Op is allowed to hold on to between
  // invocations. When the Tensor is needed in a subsequent
  // invocation, it can be retrieved from the PersistentTensor using
  // the AccessTensor method. This ensures that the system is made
  // aware of any use of the tensor's allocated memory, which is
  // needed for correctness on asynchronous devices such as GPUs.
  //
  // 2) allocate_output. This should be used to allocate any tensor
  // that is going to be used as an output from the Op at the end of
  // the current execution. The caller indicates which output the
  // Tensor will be assigned to, and the call returns the
  // newly-allocated Tensor. The Tensor can subsequently be assigned
  // to during kernel execution, and will be used as the designated
  // output when the kernel execution completes.
  //
  // 3) allocate_temp. This should be used to allocate any scratch
  // storage that is needed while the kernel is executing, and will
  // not be retained by the Op.
  //
  // In some cases a Tensor needs to be used as an output even though
  // it was previously allocated elsewhere. The Tensor may have been
  // passed as an input, or stored in a PersistentTensor during a
  // previous kernel execution, or allocated earlier in the kernel
  // execution at a time when it was not known which output it would
  // be assigned to. In this case the kernel can use set_output or
  // set_output_ref to indicate that the tensor should be used as the
  // designated output. It is legal to use any previously-allocated
  // Tensor as an argument to set_output or set_output_ref, including
  // Tensors allocated via allocate_temp. There may be a performance
  // penalty to using a Tensor that was not allocated using
  // allocate_output. This is because allocate_output uses the
  // AllocatorAttributes stored in output_alloc_attr for the
  // designated output. In some cases, using the wrong attributes may
  // cause an extra copy of the Tensor's buffer.

  // Allocates output for the specified output index with shape.
  // OpKernelContext retains ownership of the returned pointer. See
  // comment above.
  //
  // If memory allocation fails, returns an error status.
  //
  // REQUIRES: !IsRefType(expected_output_dtype(index))
  Status allocate_output(int index, const TensorShape& shape,
                         Tensor** tensor) TF_MUST_USE_RESULT;
  Status allocate_output(const string& name, const TensorShape& shape,
                         Tensor** tensor) TF_MUST_USE_RESULT;
  // The following methods use the supplied attributes instead of
  // those in output_alloc_attr. The caller is responsible for
  // ensuring that the attributes are "compatible" with the
  // output_alloc_attr, e.g. the tensor is allocated on the correct
  // device. See comment above.
  Status allocate_output(int index, const TensorShape& shape, Tensor** tensor,
                         AllocatorAttributes attr) TF_MUST_USE_RESULT;
  Status allocate_output(const string& name, const TensorShape& shape,
                         Tensor** tensor,
                         AllocatorAttributes attr) TF_MUST_USE_RESULT;

  // Allocates a temporary Tensor of the specified type and
  // shape. Devices such as GPUs that enqueue Ops for lazy execution
  // may retain references to the temporary tensors after the Op's
  // Compute method has run. See comment above.
  Status allocate_temp(DataType type, const TensorShape& shape,
                       Tensor* out_temp, AllocatorAttributes attr);
  Status allocate_temp(DataType type, const TensorShape& shape,
                       Tensor* out_temp) {
    return allocate_temp(type, shape, out_temp, AllocatorAttributes());
  }

  // Allocates a Tensor of the specified type and shape which the Op
  // plans to maintain as persistent state. out_persistent holds the
  // PersistentTensor which is the object the caller should store. For
  // convenience, if out_tensor is non-null then it will be filled in
  // with a Tensor* pointing to the newly-allocated tensor which the
  // caller can use instead of calling
  // out_persistent->AccessTensor. The caller does not own out_tensor
  // and should not keep a copy of it. See comment above.
  Status allocate_persistent(DataType type, const TensorShape& shape,
                             PersistentTensor* out_persistent,
                             Tensor** out_tensor, AllocatorAttributes attr);
  Status allocate_persistent(DataType type, const TensorShape& shape,
                             PersistentTensor* out_persistent,
                             Tensor** out_tensor) {
    return allocate_persistent(type, shape, out_persistent, out_tensor,
                               AllocatorAttributes());
  }

  // Copies a tensor (allocated by the caller) to the specified output
  // index.  REQUIRES: !IsRefType(expected_output_dtype(index))
  // REQUIRES: 'tensor' must have the same MemoryType as
  // output_memory_types[index]. See comment above.
  // TODO(mrry): Convert this to return Status.
  void set_output(int index, const Tensor& tensor);
  Status set_output(const string& name, const Tensor& tensor);

  // To output a reference.  Caller retains ownership of mu and tensor_for_ref,
  // and they must outlive all uses within the step. See comment above.
  // REQUIRES: IsRefType(expected_output_dtype(index))
  // TODO(mrry): Convert this to return Status.
  void set_output_ref(int index, mutex* mu, Tensor* tensor_for_ref);
  Status set_output_ref(const string& name, mutex* mu, Tensor* tensor_for_ref);

  // Returns nullptr if allocate_output() or set_output() have not been called.
  // TODO(mrry): Convert this to return Status.
  Tensor* mutable_output(int index);
  Status mutable_output(const string& name, Tensor** tensor);

  // Transfers ownership of an output tensor to the caller.
  // NOTE: For non-reference outputs, the caller takes responsibility
  // for deletion. For reference outputs, the caller does NOT take
  // responsibility for deletion.
  // TODO(mrry): Convert this to return Status.
  TensorValue release_output(int index);
  Status release_output(const string& name, TensorValue* value);

  // Records device specific state about how the input tensors were
  // computed.
  //
  // If using the templated function, the type must be a subclass
  // of DeviceContext.
  //
  // Get the DeviceContext used for the index input.  Returns nullptr
  // if no DeviceContext was provided.
  template <typename T>
  T* input_device_context(int index);
  DeviceContext* input_device_context(int index);

  // Return the DeviceContext that should be used for this Op.
  //
  // If using the templated function, the type must be a subclass
  // of DeviceContext.
  //
  // Returns nullptr if the device did not provide one.
  template <typename T>
  T* op_device_context();
  DeviceContext* op_device_context() {
    DeviceContext* ret = params_.op_device_context;
    if (ret == nullptr) {
      auto* dev_info = device()->tensorflow_gpu_device_info();
      if (dev_info) ret = dev_info->default_context;
    }
    return ret;
  }

  AllocatorAttributes input_alloc_attr(int index) const {
    DCHECK_GE(index, 0);
    DCHECK_LT(index, params_.input_alloc_attrs->size());
    return (*params_.input_alloc_attrs)[index];
  }

  AllocatorAttributes output_alloc_attr(int index) const {
    return params_.output_alloc_attr(index);
  }

  gtl::InlinedVector<WrappedAllocator, 4> wrapped_allocators() const {
    mutex_lock lock(mu_);
    gtl::InlinedVector<WrappedAllocator, 4> retrieved = wrapped_allocators_;
    return retrieved;
  }

  // Communication.
  //
  // An op kernel communicates with outside environment through
  // Rendezvous Send() and Recv().
  Rendezvous* rendezvous() const { return params_.rendezvous; }

  // Function call support.
  //
  // If this kernel invocation is within a function execution,
  // call_frame() returns the call frame for the function call.
  FunctionCallFrame* call_frame() const { return params_.call_frame; }

  // If not nullptr, the kernel invoke functions defined in the
  // library. E.g., CHECK_NOTNULL(function_library())->Run("Foo", ...).
  FunctionLibraryRuntime* function_library() const {
    return params_.function_library;
  }

  // Shared resources accessible to this kernel.
  ResourceMgr* resource_manager() const { return params_.resource_manager; }

  checkpoint::TensorSliceReaderCacheWrapper* slice_reader_cache() const {
    return params_.slice_reader_cache;
  }

  // Execution.
  //
  // OpKernels can use these eigen devices to carry out their
  // numerical computation.
  const Eigen::ThreadPoolDevice& eigen_cpu_device() const {
    return *device()->eigen_cpu_device();
  }
  const Eigen::GpuDevice& eigen_gpu_device() const {
    return eigen_gpu_device_->device();
  }
  template <typename EigenDeviceType>
  const EigenDeviceType& eigen_device() const;

  // Error handling.

  // If expected_inputs == inputs() and expected_outputs == output_types(),
  // returns OK, else returns INVALID_ARGUMENT with an error message.
  // Recommended for Ops with dynamic signatures, where validation can only
  // be performed at runtime.
  Status MatchSignature(const DataTypeSlice expected_inputs,
                        const DataTypeSlice expected_outputs);

  // An OpKernel should call SetStatus() if Compute() encounters an
  // error.
  void SetStatus(const Status& status) { status_.Update(status); }
  const Status& status() const { return status_; }

  // Cancellation.
  //
  // EXPERIMENTAL. See the implementation in tensorflow::TensorQueue for an
  // example of how to use this API.
  CancellationManager* cancellation_manager() const {
    return params_.cancellation_manager;
  }

  // Other accessors.

  // For control flow.
  FrameAndIter frame_iter() const { return params_.frame_iter; }
  bool is_input_dead() const { return params_.is_input_dead; }
  bool* is_output_dead() { return &is_output_dead_; }

  // May be used, e.g., to get GPU handles, etc.
  // TODO(tucker): Add example usage.
  DeviceBase* device() const { return params_.device; }

  // Access to list of temporary tensors.
  int num_temps();
  Tensor* temp(int index);

  // Access to information about whether each output was newly
  // allocated or copied from an existing tensor
  AllocationType output_allocation_type(int index) const {
    return output_allocation_types_[index];
  }

 private:
  Allocator* get_allocator(AllocatorAttributes attr) {
    Allocator* allocator = params_.device->GetAllocator(attr);
    if (params_.track_allocations) {
      mutex_lock lock(mu_);
      for (const auto& wrapped : wrapped_allocators_) {
        if (wrapped.first == allocator) {
          return wrapped.second;
        }
      }
      TrackingAllocator* wrapped_allocator = new TrackingAllocator(allocator);
      wrapped_allocators_.push_back(
          std::make_pair(allocator, wrapped_allocator));
      return wrapped_allocator;
    } else {
      return allocator;
    }
  }

  // Per-step resource manager for use by white-listed internal ops.
  friend class TemporaryVariableOp;
  friend class DestroyTemporaryVariableOp;
  ResourceMgr* step_resource_manager() const {
    return params_.step_resource_manager;
  }

  // Internal common method used when allocating tensor memory
  Status allocate_tensor(DataType type, const TensorShape& shape,
                         Tensor* out_tensor, AllocatorAttributes attr);

  // This is called by PersistentTensor::AccessTensor whenever the
  // wrapped tensor is retrieved, to ensure the runtime knows that the
  // Tensor is being accessed within an Op. This is necessary for
  // memory safety of devices like GPUs that queue Ops for
  // asynchronous execution after the Compute() method completes.
  friend class PersistentTensor;
  void NotifyUseOfPersistentTensor(const Tensor& tensor);

  Status status_;
  Params params_;  // immutable after construction.
  const PerOpGpuDevice* eigen_gpu_device_;  // owned, with a per-op
                                            // wrapped allocator
  mutable mutex mu_;  // mutable so const accessors can acquire the lock
  gtl::InlinedVector<WrappedAllocator, 4> wrapped_allocators_ GUARDED_BY(mu_);
  gtl::InlinedVector<TensorValue, 4> outputs_;
  gtl::InlinedVector<AllocationType, 4> output_allocation_types_;
  gtl::InlinedVector<Tensor*, 4> temp_tensors_;
  bool is_output_dead_ = false;

  TF_DISALLOW_COPY_AND_ASSIGN(OpKernelContext);
};

// Register your OpKernel by specifying the Op's name, the device the
// kernel runs on, any type attr constraints for this kernel, any
// host-memory args, and the class to instantiate.  Examples:
//
//  // A kernel that supports all types.
//  REGISTER_KERNEL_BUILDER(Name("Save").Device(DEVICE_CPU), SaveOp);
//
//  // The following are equivalent ways of specifying that the kernel only
//  // works if the "T" type attr is set to DT_FLOAT.
//  REGISTER_KERNEL_BUILDER(
//      Name("Sub").Device(DEVICE_CPU).TypeConstraint<float>("T"),
//      SubOp<float>);
//  // (You would then repeat this for every type supported by "Sub".)
//
//  // This form allows you to specify a list of types as the constraint.
//  REGISTER_KERNEL_BUILDER(Name("Sub")
//                              .Device(DEVICE_CPU)
//                              .TypeConstraint("T", {DT_FLOAT}),
//                          SubOp<float>);
//
//  // A kernel that expects one of the input tensors in host memory.
//  REGISTER_KERNEL_BUILDER(
//      Name("Reshape").Device(DEVICE_GPU).HostMemory("shape"), ReshapeOp);
//
// See kernel_def_builder for details.

// Instantiate an OpKernel that has been registered.  Returns nullptr
// if no operation for that type of device / input signature combination
// (and a NOT_FOUND *status), or there is an error in construction (and
// an INVALID_ARGUMENT *status).  Otherwise, the caller takes ownership
// of the returned pointer.
// EXPECTED USAGE: unique_ptr<OpKernel> op = CreateOpKernel(...);
// REQUIRES: def has all attrs specified (e.g. using AddDefaultsToNodeDef()).
std::unique_ptr<OpKernel> CreateOpKernel(DeviceType device_type,
                                         DeviceBase* device,
                                         Allocator* allocator,
                                         const NodeDef& def, Status* status);
Status CreateOpKernel(DeviceType device_type, DeviceBase* device,
                      Allocator* allocator, FunctionLibraryRuntime* flib,
                      const NodeDef& def, OpKernel** kernel);

// Returns into 'device_types' the subset of prioritized_types that this
// binary has registered for the given NodeDef.
//
// REQUIRES: * 'device_types' is not nullptr.
//           * def has all attrs specified (e.g. using AddDefaultsToNodeDef()).
Status SupportedDeviceTypesForNode(
    const std::vector<DeviceType>& prioritized_types, const NodeDef& def,
    DeviceTypeVector* device_types);

// Returns into *{input,output}_memory_types the memory type of each
// {input,output} tensor.
//
// REQUIRES: * '*_memory_types' is not nullptr.
//           * def has all attrs specified (e.g. using AddDefaultsToNodeDef()).
Status MemoryTypesForNode(DeviceType device_type, const NodeDef& ndef,
                          const OpDef& op_def,
                          const NameRangeMap& input_name_map,
                          const NameRangeMap& output_name_map,
                          MemoryTypeVector* input_memory_types,
                          MemoryTypeVector* output_memory_types);

Status MemoryTypesForNode(const OpRegistryInterface* op_registry,
                          DeviceType device_type, const NodeDef& ndef,
                          MemoryTypeVector* input_memory_types,
                          MemoryTypeVector* output_memory_types);

// Call once after Op registration has completed.
Status ValidateKernelRegistrations(const OpRegistryInterface* op_registry);

// -----------------------------------------------------------------------------
// OpKernel registration implementation follows, please ignore.

// Allow the REGISTER_KERNEL_BUILDER(Name("op_name").Device(...)...) syntax.
namespace register_kernel {
typedef ::tensorflow::KernelDefBuilder Name;
}  // namespace register_kernel

#define REGISTER_KERNEL_BUILDER(kernel_builder, ...) \
  REGISTER_KERNEL_BUILDER_UNIQ_HELPER(__COUNTER__, kernel_builder, __VA_ARGS__)

#define REGISTER_KERNEL_BUILDER_UNIQ_HELPER(ctr, kernel_builder, ...) \
  REGISTER_KERNEL_BUILDER_UNIQ(ctr, kernel_builder, __VA_ARGS__)

#define REGISTER_KERNEL_BUILDER_UNIQ(ctr, kernel_builder, ...)   \
  static ::tensorflow::kernel_factory::OpKernelRegistrar         \
      registrar__body__##ctr##__object(                          \
          ::tensorflow::register_kernel::kernel_builder.Build(), \
          +[](::tensorflow::OpKernelConstruction* context)       \
              -> ::tensorflow::OpKernel* { return new __VA_ARGS__(context); })

namespace kernel_factory {

class OpKernelRegistrar {
 public:
  typedef OpKernel* (*Factory)(OpKernelConstruction*);
  OpKernelRegistrar(const KernelDef* kernel_def, Factory factory);
};

}  // namespace kernel_factory

// -----------------------------------------------------------------------------
// Template and inline method implementations, please ignore

inline DataType OpKernelContext::input_dtype(int index) const {
  DCHECK_GE(index, 0);
  DCHECK_LT(index, params_.inputs->size());
  const TensorValue& value((*params_.inputs)[index]);
  if (value.is_ref()) {
    return MakeRefType(value->dtype());
  } else {
    return value->dtype();
  }
}

inline DataType OpKernelContext::expected_output_dtype(int index) const {
  DCHECK_GE(index, 0);
  DCHECK_LT(index, params_.op_kernel->output_types().size());
  return params_.op_kernel->output_type(index);
}

inline const Tensor& OpKernelContext::input(int index) const {
  DCHECK_GE(index, 0);
  DCHECK_LT(index, params_.inputs->size());
  DCHECK(!(*params_.inputs)[index].is_ref());
  return *((*params_.inputs)[index].tensor);
}

inline Tensor OpKernelContext::mutable_input(int index, bool lock_held) {
  DCHECK_GE(index, 0);
  DCHECK_LT(index, params_.inputs->size());
  DCHECK((*params_.inputs)[index].is_ref());
  // return a copy of the Ref acquired while holding the mutex
  if (lock_held) {
    return *((*params_.inputs)[index].tensor);
  } else {
    mutex_lock l(*input_ref_mutex(index));
    return *((*params_.inputs)[index].tensor);
  }
}

inline void OpKernelContext::replace_ref_input(int index, const Tensor& tensor,
                                               bool lock_held) {
  DCHECK_GE(index, 0);
  DCHECK_LT(index, params_.inputs->size());
  DCHECK((*params_.inputs)[index].is_ref());
  // should only modify the tensor while holding the mutex
  if (lock_held) {
    *(*params_.inputs)[index].tensor = tensor;
  } else {
    mutex_lock l(*input_ref_mutex(index));
    *(*params_.inputs)[index].tensor = tensor;
  }
}

inline void OpKernelContext::forward_ref_input_to_ref_output(int input_index,
                                                             int output_index) {
  DCHECK_GE(input_index, 0);
  DCHECK_LT(input_index, params_.inputs->size());
  DCHECK((*params_.inputs)[input_index].is_ref());
  set_output_ref(output_index, (*params_.inputs)[input_index].mutex_if_ref,
                 (*params_.inputs)[input_index].tensor);
}

inline void OpKernelContext::delete_ref_input(int index, bool lock_held) {
  DCHECK_GE(index, 0);
  DCHECK_LT(index, params_.inputs->size());
  DCHECK((*params_.inputs)[index].is_ref());
  // should only modify the tensor while holding the mutex
  if (lock_held) {
    delete (*params_.inputs)[index].tensor;
  } else {
    mutex_lock l(*input_ref_mutex(index));
    delete (*params_.inputs)[index].tensor;
  }
}

// no input if tensor == nullptr.
inline bool OpKernelContext::has_input(int index) const {
  DCHECK_GE(index, 0);
  DCHECK_LT(index, params_.inputs->size());
  return (*params_.inputs)[index].tensor != nullptr;
}

inline mutex* OpKernelContext::input_ref_mutex(int index) {
  DCHECK_GE(index, 0);
  DCHECK_LT(index, params_.inputs->size());
  DCHECK((*params_.inputs)[index].is_ref());
  return (*params_.inputs)[index].mutex_if_ref;
}

inline Status OpKernelContext::allocate_output(int index,
                                               const TensorShape& shape,
                                               Tensor** output) {
  DCHECK_GE(index, 0);
  DCHECK_LT(index, num_outputs());
  DCHECK(params_.output_alloc_attr);
  AllocatorAttributes attr = params_.output_alloc_attr(index);
  return allocate_output(index, shape, output, attr);
}

inline Status OpKernelContext::allocate_tensor(DataType type,
                                               const TensorShape& shape,
                                               Tensor* out_tensor,
                                               AllocatorAttributes attr) {
  Allocator* a = get_allocator(attr);
  Tensor new_tensor(a, type, shape);

  if (!new_tensor.IsInitialized() && shape.num_elements() > 0) {
    return errors::ResourceExhausted("OOM when allocating tensor with shape",
                                     shape.DebugString());
  }
  *out_tensor = new_tensor;
  return Status::OK();
}

inline Status OpKernelContext::allocate_output(int index,
                                               const TensorShape& shape,
                                               Tensor** output,
                                               AllocatorAttributes attr) {
  DCHECK_GE(index, 0);
  DCHECK_LT(index, outputs_.size());
  // Record the fact that this output tensor was allocated by the Op
  DCHECK_LT(index, output_allocation_types_.size());
  output_allocation_types_[index] = AT_ALLOCATED;
  const DataType type = params_.op_kernel->output_type(index);
  DCHECK(!IsRefType(type));
  DCHECK(mutable_output(index) == nullptr);
  Tensor* output_tensor = new Tensor();
  Status s = allocate_tensor(type, shape, output_tensor, attr);
  if (s.ok()) {
    outputs_[index] = TensorValue(output_tensor);
    *output = outputs_[index].tensor;
  }
  return s;
}

inline Status OpKernelContext::allocate_temp(DataType type,
                                             const TensorShape& shape,
                                             Tensor* out_temp,
                                             AllocatorAttributes attr) {
  Status s = allocate_tensor(type, shape, out_temp, attr);
  if (s.ok()) {
    if (params_.device->SaveTemporaryTensors()) {
      // keep a reference to the underlying memory around
      temp_tensors_.push_back(new Tensor(*out_temp));
    }
  }
  return s;
}

inline Status OpKernelContext::allocate_persistent(
    DataType type, const TensorShape& shape, PersistentTensor* out_persistent,
    Tensor** out_tensor, AllocatorAttributes attr) {
  // TODO(misard) add specific memory tracking for persistent tensors
  Tensor persistent;
  Status s = allocate_tensor(type, shape, &persistent, attr);
  if (s.ok()) {
    *out_persistent = PersistentTensor(persistent);
    // This call saves a reference to the newly-allocated tensor if we
    // are saving temporary tensors
    Tensor* allocated = out_persistent->AccessTensor(this);
    if (out_tensor) {
      *out_tensor = allocated;
    }
  }
  return s;
}

inline void OpKernelContext::NotifyUseOfPersistentTensor(const Tensor& t) {
  if (t.IsInitialized() && params_.device->SaveTemporaryTensors()) {
    // keep a reference to the underlying memory around
    temp_tensors_.push_back(new Tensor(t));
  }
}

inline int OpKernelContext::num_temps() { return temp_tensors_.size(); }

inline Tensor* OpKernelContext::temp(int index) {
  DCHECK_GE(index, 0);
  DCHECK_LT(index, temp_tensors_.size());
  return temp_tensors_[index];
}

inline void OpKernelContext::set_output(int index, const Tensor& tensor) {
  DCHECK_GE(index, 0);
  DCHECK_LT(index, outputs_.size());
  // Record the fact that this output tensor was set by the Op
  DCHECK_LT(index, output_allocation_types_.size());
  output_allocation_types_[index] = AT_EXISTING;
  DCHECK(!IsRefType(params_.op_kernel->output_type(index)));
  DCHECK_EQ(mutable_output(index), nullptr);
  outputs_[index] = TensorValue(new Tensor(tensor));
}

inline void OpKernelContext::set_output_ref(int index, mutex* mu,
                                            Tensor* tensor_for_ref) {
  DCHECK_GE(index, 0);
  DCHECK_LT(index, outputs_.size());
  // Record the fact that this output tensor was set by reference the Op
  DCHECK_LT(index, output_allocation_types_.size());
  output_allocation_types_[index] = AT_REF;
  DCHECK(IsRefType(params_.op_kernel->output_type(index)));
  outputs_[index] = TensorValue(mu, tensor_for_ref);
}

inline Tensor* OpKernelContext::mutable_output(int index) {
  DCHECK_GE(index, 0);
  DCHECK_LT(index, outputs_.size());
  return outputs_[index].tensor;
}

inline TensorValue OpKernelContext::release_output(int index) {
  DCHECK_GE(index, 0);
  DCHECK_LT(index, outputs_.size());
  TensorValue value = outputs_[index];
  outputs_[index] = TensorValue();
  return value;
}

template <typename T>
T* OpKernelContext::op_device_context() {
  static_assert(std::is_base_of<DeviceContext, T>::value,
                "T is not a subclass of DeviceContext");
  return static_cast<T*>(op_device_context());
}

template <typename T>
T* OpKernelContext::input_device_context(int index) {
  DCHECK_GE(index, 0);
  DCHECK_LT(index, params_.input_device_contexts->size());
  static_assert(std::is_base_of<DeviceContext, T>::value,
                "T is not a subclass of DeviceContext");
  return static_cast<T*>((*params_.input_device_contexts)[index]);
}

inline DeviceContext* OpKernelContext::input_device_context(int index) {
  DCHECK_GE(index, 0);
  DCHECK_LT(index, params_.input_device_contexts->size());
  return (*params_.input_device_contexts)[index];
}

inline const Tensor& OpInputList::operator[](int i) const {
  DCHECK_GE(i, 0);
  DCHECK_LT(i, stop_ - start_);
  return ctx_->input(start_ + i);
}

inline mutex* OpMutableInputList::ref_mutex(int i) {
  DCHECK_GE(i, 0);
  DCHECK_LT(i, stop_ - start_);
  return ctx_->input_ref_mutex(start_ + i);
}

inline Tensor OpMutableInputList::at(int i, bool lock_held) {
  DCHECK_GE(i, 0);
  DCHECK_LT(i, stop_ - start_);
  return ctx_->mutable_input(start_ + i, lock_held);
}

inline Tensor* OpOutputList::operator[](int i) {
  DCHECK_GE(i, 0);
  DCHECK_LT(i, stop_ - start_);
  return ctx_->mutable_output(start_ + i);
}

inline bool OpOutputList::required(int i) const {
  DCHECK_GE(i, 0);
  DCHECK_LT(i, stop_ - start_);
  return ctx_->output_required(start_ + i);
}

inline Status OpOutputList::allocate(int i, const TensorShape& shape,
                                     Tensor** output) {
  DCHECK_GE(i, 0);
  DCHECK_LT(i, stop_ - start_);
  return ctx_->allocate_output(start_ + i, shape, output);
}

inline void OpOutputList::set(int i, const Tensor& tensor) {
  DCHECK_GE(i, 0);
  DCHECK_LT(i, stop_ - start_);
  ctx_->set_output(start_ + i, tensor);
}

inline void OpOutputList::set_ref(int i, mutex* mu, Tensor* tensor_for_ref) {
  DCHECK_GE(i, 0);
  DCHECK_LT(i, stop_ - start_);
  ctx_->set_output_ref(i, mu, tensor_for_ref);
}

}  // namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_OP_KERNEL_H_
