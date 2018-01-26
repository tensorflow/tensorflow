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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_USER_COMPUTATION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_USER_COMPUTATION_H_

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/session.pb.h"
#include "tensorflow/compiler/xla/service/versioned_computation_handle.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla.pb.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// A UserComputation is the built-up computation that users create via the
// XLA Service interface.
//
// The XLA service adds instructions to a user computation via this
// interface. The state of the computation is stored as a SessionComputation
// proto which holds a record of all operation-building requests received by the
// XLA service.
//
// UserComputations are lowered to HloComputations which are passed to the high
// level compiler interface.
class UserComputation {
 public:
  // Factory used when restoring a computation from serialized session
  // computation (computation snapshot) data. Remaps any references to
  // computation handle via the old_to_new mapping.
  //
  // An error will occur if the old_to_new mapping cannot resolve a reference to
  // a computation that is present in session_computation.
  static StatusOr<std::unique_ptr<UserComputation>> MakeWithRemapping(
      const SessionComputation& session_computation,
      const ComputationHandle& handle,
      const std::map<int64, ComputationHandle>& old_to_new);

  // Creates an empty computation with the given name and computation handle.
  explicit UserComputation(const string& name, const ComputationHandle& handle);

  // Enqueues a parameter-retrieving instruction onto this user computation.
  // Returns an error status if the parameter number is already registered with
  // different values.
  StatusOr<ComputationDataHandle> AddParameterInstruction(
      const ParameterRequest& parameter_request);

  // Enqueues a pad instruction onto this user computation.
  StatusOr<ComputationDataHandle> AddPadInstruction(
      const PadRequest& pad_request);

  // Enqueues a tracing instruction onto this user computation.
  // Returns an error status if the operand cannot be resolved.
  Status AddTraceInstruction(const TraceRequest& trace_request);

  // Enqueues a random number generation instruction onto this user computation.
  StatusOr<ComputationDataHandle> AddRngInstruction(
      const RngRequest& rng_request);

  // Enqueues a unary instruction onto this user computation.
  // Returns an error status if the operand index is out of bounds.
  StatusOr<ComputationDataHandle> AddUnaryInstruction(
      const UnaryOpRequest& unary_request);

  // Enqueues a batch norm training instruction onto this user computation.
  StatusOr<ComputationDataHandle> AddBatchNormTrainingInstruction(
      const BatchNormTrainingRequest& batch_norm_training_request);

  // Enqueues a batch norm inference instruction onto this user computation.
  StatusOr<ComputationDataHandle> AddBatchNormInferenceInstruction(
      const BatchNormInferenceRequest& batch_norm_inference_request);

  // Enqueues a batch norm grad instruction onto this user computation.
  StatusOr<ComputationDataHandle> AddBatchNormGradInstruction(
      const BatchNormGradRequest& batch_norm_grad_request);

  // Enqueues a binary instruction onto this user computation.
  // Returns an error status if the operand indices are out of bounds.
  StatusOr<ComputationDataHandle> AddBinaryInstruction(
      const BinaryOpRequest& binary_request);

  // Enqueues a ternary instruction onto this user computation.
  // Returns an error status if the operand indices are out of bounds.
  StatusOr<ComputationDataHandle> AddTernaryInstruction(
      const TernaryOpRequest& ternary_request);

  // Enqueues a variadic instruction onto this user computation.
  // Returns an error status if the operand indices are out of bounds.
  StatusOr<ComputationDataHandle> AddVariadicInstruction(
      const VariadicOpRequest& variadic_request);

  // Enqueues a constant instruction onto this user computation.
  StatusOr<ComputationDataHandle> AddConstantInstruction(
      const ConstantRequest& constant_request);

  // Enqueues a get tuple element instruction onto this user computation.
  StatusOr<ComputationDataHandle> AddGetTupleElementInstruction(
      const GetTupleElementRequest& get_tuple_element_request);

  // Enqueues a map instruction onto this user computation.
  StatusOr<ComputationDataHandle> AddMapInstruction(
      const MapRequest& map_request,
      const UserComputation& to_apply_computation);

  // Enqueues a reduce-precision instruction onto this user computation.
  StatusOr<ComputationDataHandle> AddReducePrecisionInstruction(
      const ReducePrecisionRequest& reduce_precision_request);

  // Enqueues a convolution instruction onto this user computation.
  StatusOr<ComputationDataHandle> AddConvolveInstruction(
      const ConvolveRequest& convolve_request);

  // Enqueues an FFT instruction onto this user computation.
  StatusOr<ComputationDataHandle> AddFftInstruction(
      const FftRequest& fft_request);

  // Enqueues a cross replica sum instruction onto this user computation.
  StatusOr<ComputationDataHandle> AddCrossReplicaSumInstruction(
      const CrossReplicaSumRequest& cross_replica_sum_request);

  // Enqueues an infeed instruction onto this user computation.
  StatusOr<ComputationDataHandle> AddInfeedInstruction(
      const InfeedRequest& infeed_request);

  // Enqueues an outfeed instruction onto this user computation.
  Status AddOutfeedInstruction(const OutfeedRequest& outfeed_request);

  // Enqueues a call instruction onto this user computation.
  StatusOr<ComputationDataHandle> AddCallInstruction(
      const CallRequest& call_request,
      const UserComputation& to_apply_computation);

  // Enqueues a custom call instruction onto this user computation.
  StatusOr<ComputationDataHandle> AddCustomCallInstruction(
      const CustomCallRequest& custom_call_request);

  // Enqueues a dot instruction onto this user computation.
  StatusOr<ComputationDataHandle> AddDotInstruction(
      const DotRequest& dot_request);

  // Enqueues a broadcast instruction onto this user computation.
  StatusOr<ComputationDataHandle> AddBroadcastInstruction(
      const BroadcastRequest& broadcast_request);

  // Enqueues a reshape instruction onto this user computation.
  StatusOr<ComputationDataHandle> AddReshapeInstruction(
      const ReshapeRequest& reshape_request);

  // Enqueues a transpose instruction onto this user computation.
  StatusOr<ComputationDataHandle> AddTransposeInstruction(
      const TransposeRequest& transpose_request);

  // Enqueues a slice instruction onto this user computation.
  StatusOr<ComputationDataHandle> AddSliceInstruction(
      const SliceRequest& slice_request);

  // Enqueues a dynamic slice instruction onto this user computation.
  StatusOr<ComputationDataHandle> AddDynamicSliceInstruction(
      const DynamicSliceRequest& dynamic_slice_request);

  // Enqueues a dynamic update slice instruction onto this user computation.
  StatusOr<ComputationDataHandle> AddDynamicUpdateSliceInstruction(
      const DynamicUpdateSliceRequest& dynamic_update_slice_request);

  // Enqueues a concatenate instruction onto this user computation.
  StatusOr<ComputationDataHandle> AddConcatenateInstruction(
      const ConcatenateRequest& concatenate_request);

  // Enqueues a convert instruction onto this user computation.
  StatusOr<ComputationDataHandle> AddConvertInstruction(
      const ConvertRequest& convert_request);

  // Enqueues a bitcast element instruction onto this user computation.
  StatusOr<ComputationDataHandle> AddBitcastConvertInstruction(
      const ConvertRequest& convert_request);

  // Enqueues a reduce instruction onto this user computation.
  StatusOr<ComputationDataHandle> AddReduceInstruction(
      const ReduceRequest& reduce_request,
      const UserComputation& to_apply_computation);

  // Enqueues a windowed reduce instruction onto this user computation.
  StatusOr<ComputationDataHandle> AddReduceWindowInstruction(
      const ReduceWindowRequest& reduce_window_request,
      const UserComputation& to_apply_computation);

  // Enqueues a select-and-scatter instruction onto this user
  // computation.
  StatusOr<ComputationDataHandle> AddSelectAndScatterInstruction(
      const SelectAndScatterRequest& select_and_scatter_request,
      const UserComputation& select_computation,
      const UserComputation& scatter_computation);

  // Enqueues a reverse instruction onto this user computation.
  StatusOr<ComputationDataHandle> AddReverseInstruction(
      const ReverseRequest& reverse_request);

  // Enqueues a while instruction onto this user computation.
  StatusOr<ComputationDataHandle> AddWhileInstruction(
      const WhileRequest& while_request,
      const UserComputation& condition_computation,
      const UserComputation& body_computation);

  // Enqueues a conditional instruction on this user computation.
  StatusOr<ComputationDataHandle> AddConditionalInstruction(
      const ConditionalRequest& conditional_request,
      const UserComputation& true_computation,
      const UserComputation& false_computation);

  // Enqueues a Send instruction onto this user computation.
  Status AddSendInstruction(const SendRequest& send_request);

  // Enqueues a Recv instruction onto this user computation.
  StatusOr<ComputationDataHandle> AddRecvInstruction(
      const RecvRequest& recv_request);

  // Returns the user-provided name of this user computation, which is provided
  // via the XLA computation-building API.
  const string& name() const { return name_; }

  // Subsequent executions of this computation will compute the value
  // represented by handle, rather than the last expression enqueued
  // on the computation.
  Status SetReturnValue(const ComputationDataHandle& handle);

  // Return a versioned handle for this computation.
  VersionedComputationHandle GetVersionedHandle() const;

  // Return a versioned handle for this computation with a version equal to the
  // point at which given operation was added to the computation.
  VersionedComputationHandle GetVersionedHandleAtOperation(
      const ComputationDataHandle& operation) const;

  // Return a version value representing the current state of the
  // computation.
  VersionedComputationHandle::Version version() const;

  // Computes and returns the program shape for the user computation -- gathers
  // parameters and result type into a single proto. A shared_ptr is used
  // because the returned pointer refers to an internally cached value which may
  // be discarded by the UserComputation object. This avoid unnecessary copies.
  //
  // If the parameter space is not dense (i.e. there are holes in the parameter
  // numbers provided) then an error status is returned.
  StatusOr<std::shared_ptr<const ProgramShape>> ComputeProgramShape(
      VersionedComputationHandle::Version version) const;

  // Returns true if the given data handle does not depend on any parameter with
  // index higher then num_parameters. That is, the value can be computed at
  // compile time if we know the first num_parameters arguments.
  StatusOr<bool> IsConstant(const ComputationDataHandle& handle,
                            int64 num_parameters);

  // Returns the output shape of the operation indicated by the given handle.
  StatusOr<Shape> GetShape(const ComputationDataHandle& handle);

  // Sets metadata on the Hlo instruction referenced by the given handle.
  Status SetOpMetadata(const ComputationDataHandle& handle,
                       const OpMetadata& metadata);

  // Sets the device assignment on the Hlo instruction referenced by 'handle'.
  Status SetOpSharding(const ComputationDataHandle& handle,
                       const OpSharding& sharding);

  // Builds a HLO computation from the UserComputation. The parameter "resolver"
  // is a function which returns a pointer to the HloComputation corresponding
  // to the given ComputationHandle at the given version. The resolver is used
  // for operations, such as map, which call other computations and need a
  // pointer to the called HloComputation to construct the respective HLO
  // instructions. If include_unreachable_instructions is true, then
  // instructions which are not reachable from the root are lowered into
  // HloInstructions.
  using HloComputationResolver =
      std::function<HloComputation*(const VersionedComputationHandle& handle)>;
  StatusOr<std::unique_ptr<HloComputation>> BuildHloComputation(
      VersionedComputationHandle::Version version,
      HloComputationResolver hlo_resolver, const DebugOptions& debug_options,
      bool include_unreachable_instructions = true) const;

  // Return a vector containing the embedded computations used by this
  // UserComputation. Only embedded computations which are called directly by
  // this UserComputation are included. That is, the transitive closure of
  // embedded computations is not included.
  std::vector<VersionedComputationHandle> GetEmbeddedComputations(
      VersionedComputationHandle::Version version) const;

  // Returns the number of OperationRequest objects in this UserComputation.
  // The 'version' of a computation is identical to the number of
  // OperationRequests in the UserComputation.
  int64 request_count(VersionedComputationHandle::Version version) const {
    return version;
  }

  // Returns a copy of the internal session state for this computation -- this
  // is useful for serializing the guts of a user computation, though references
  // to other handles (e.g. referred-to computations) must be handled with care
  // in the serialization / de-serialization process.
  SessionComputation CloneSessionComputation(
      VersionedComputationHandle::Version version) const;

  // Warning: typically we don't want to look up computation data handles until
  // the computation is finished being built, for consistency purposes. We
  // expose this routine for error reporting purposes so that we can provide
  // more meaningful error messages from the XLA service layer.
  //
  // Returns the operation request that the handle comes from.
  StatusOr<const OperationRequest*> LookUpRequestForErrorReporting(
      const ComputationDataHandle& handle) const;

  // Retrieves the parameter metadata for the given parameter number.
  //
  // If the parameter number is invalid for this computation, nullopt is
  // returned. When the return value has_value(), nullptr will never be
  // the held value.
  tensorflow::gtl::optional<const OpMetadata*> ParameterMetadata(
      int parameter_number) const;

 private:
  // Warning: dangerous mutating operation that doesn't respect versioning.
  // This is only used at initialization time when constructing from a
  // SessionComputation a la MakeWithRemapping.
  //
  // Remaps references to old computations (with handle values in the keys of
  // old_to_new) to the computation handle given in the values. This is useful
  // when loading computations from snapshots, to finish initialization, before
  // the user computation is released into the wild.
  Status RemapEmbeddedComputations(
      const std::map<int64, ComputationHandle>& old_to_new)
      EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Returns the OperationRequest corresponding to the given handle.
  StatusOr<const OperationRequest*> LookUpRequest(
      const ComputationDataHandle& handle) const
      EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Creates a new ComputationDataHandle with the next available handle value.
  ComputationDataHandle CreateComputationDataHandle()
      EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Checks whether the parameter numbers of the parameter operations are
  // contiguous starting from zero. Returns appropriate error status if not.
  Status CheckParametersAreContiguous(
      VersionedComputationHandle::Version version) const
      EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  VersionedComputationHandle GetVersionedHandleInternal() const
      EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Name of the computation.
  string name_;

  mutable tensorflow::mutex mutex_;

  // State of the computation as a record of all operation-building requests.
  SessionComputation session_computation_ GUARDED_BY(mutex_);

  // Mapping from parameter number to operation request containing the
  // respective ParameterRequest.
  std::map<int64, OperationRequest*> parameters_ GUARDED_BY(mutex_);

  // The next ComputationDataHandle value to assign. Handle values are assigned
  // sequentially.
  int64 next_handle_value_ GUARDED_BY(mutex_);

  // If handle_to_return_.has_handle() then an Execution of this Computation
  // will compute the value represented by handle_to_return_, otherwise it will
  // compute the value of (next_handle_value_ - 1).
  ComputationDataHandle handle_to_return_ GUARDED_BY(mutex_);

  // Memoized ProgramShape and its version. A shared_ptr is used because
  // references to this object are returned by ComputeProgramShape.
  mutable int64 program_shape_version_ GUARDED_BY(mutex_) = 0;
  mutable std::shared_ptr<const ProgramShape> program_shape_ GUARDED_BY(mutex_);

  TF_DISALLOW_COPY_AND_ASSIGN(UserComputation);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_USER_COMPUTATION_H_
