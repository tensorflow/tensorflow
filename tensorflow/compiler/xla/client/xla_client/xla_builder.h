/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

// TODO(b/74197823): Replace computation_builder.h with this file.
//
// This is NOT YET ready to use.

#ifndef TENSORFLOW_COMPILER_XLA_CLIENT_XLA_CLIENT_XLA_BUILDER_H_
#define TENSORFLOW_COMPILER_XLA_CLIENT_XLA_CLIENT_XLA_BUILDER_H_

#include <map>
#include <string>
#include <utility>

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/stacktrace.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

class XlaBuilder;

// This represents an instruction that has been enqueued using the XlaBuilder.
// This is used to pass to subsequent computations that depends upon the
// instruction as an operand.
//
// TODO(b/74197823): Replace xla::ComputationDataHandle with this one.
class XlaOp {
 public:
  StatusOr<Shape> GetShape() const;

 private:
  XlaOp() : handle_(0), builder_(nullptr) {}
  XlaOp(int64 handle, XlaBuilder* builder)
      : handle_(handle), builder_(builder) {}

  int64 handle() const { return handle_; }
  friend class XlaBuilder;

  int64 handle_;
  XlaBuilder* builder_;  // Not owned.
};

// The computation graph that the user builds up with the XlaBuilder.
//
// TODO(b/74197823): Replace xla::Computation with this one.
class XlaComputation {
 public:
  XlaComputation(const XlaComputation&) = delete;
  XlaComputation& operator=(const XlaComputation&) = delete;

  XlaComputation(XlaComputation&& from) { *this = std::move(from); }

  XlaComputation& operator=(XlaComputation&& from) {
    proto_ = std::move(from.proto());
    unique_id_ = from.unique_id_;
    return *this;
  }

  // Returns the "program shape" (parameter and return shapes) for this
  // computation.
  const ProgramShape& GetProgramShape() const { return proto_.program_shape(); }

  const HloModuleProto& proto() const { return proto_; }

 private:
  // Creates a null Computation.
  XlaComputation(const int64 unique_id) : unique_id_(unique_id) {}
  HloModuleProto* mutable_proto() { return &proto_; }
  friend class XlaBuilder;

  int64 unique_id_;
  HloModuleProto proto_;
};

// A convenient interface for building up computations.
//
// Thread-compatible.
//
// TODO(b/74197823): Replace xla::ComputationBuilder with this one.
class XlaBuilder {
 public:
  // computation_name: name to use for the built computation.
  XlaBuilder(const string& computation_name);

  XlaBuilder(const XlaBuilder&) = delete;
  XlaBuilder& operator=(const XlaBuilder&) = delete;

  ~XlaBuilder();

  // Returns the computation name.
  const string& name() const { return name_; }

  // Sets the builder to a mode where it will die immediately when an error is
  // encountered, rather than producing it in a deferred fashion when Build() is
  // called (which is the default).
  void set_die_immediately_on_error(bool enabled) {
    die_immediately_on_error_ = enabled;
  }

  // Enqueues an add instruction onto the computation.
  XlaOp Add(const XlaOp& lhs, const XlaOp& rhs,
            tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

  // Enqueues a call instruction onto the computation.
  XlaOp Call(const XlaComputation& computation,
             tensorflow::gtl::ArraySlice<XlaOp> operands);

  // Enqueues a "retrieve parameter value" instruction for a parameter that was
  // passed to the computation.
  XlaOp Parameter(int64 parameter_number, const Shape& shape,
                  const string& name);

  // Enqueues a constant with the value of the given literal onto the
  // computation.
  XlaOp ConstantLiteral(const Literal& literal);

  // Enqueues a constant onto the computation. Methods are templated on the
  // native host type (NativeT) which corresponds to a specific XLA
  // PrimitiveType as given in the following table:
  //
  //  Native Type   PrimitiveType
  // -----------------------------
  //   bool           PRED
  //   int32          S32
  //   int64          S64
  //   uint32         U32
  //   uint64         U64
  //   float          F32
  //   double         F64
  //
  // Note: not all primitive types defined in xla_data.proto have a
  // corresponding native type yet.
  template <typename NativeT>
  XlaOp ConstantR0(NativeT value);

  // Returns the shape of the given op.
  StatusOr<Shape> GetShape(const XlaOp& op) const;

  // Builds the computation with the requested operations, or returns a non-ok
  // status.
  StatusOr<XlaComputation> Build();

 private:
  XlaOp AddInstruction(HloInstructionProto&& instr, HloOpcode opcode,
                       tensorflow::gtl::ArraySlice<XlaOp> operands = {});

  // Notes that the error occurred by:
  // * storing it internally and capturing a backtrace if it's the first error
  //   (this deferred value will be produced on the call to Build())
  // * dying if die_immediately_on_error_ is true
  void NoteError(const Status& error);

  XlaOp NoteErrorOrReturn(StatusOr<XlaOp>&& op) {
    if (!op.ok()) {
      NoteError(op.status());
      return XlaOp();
    }
    return op.ConsumeValueOrDie();
  }

  StatusOr<const HloInstructionProto*> LookUpInstruction(const XlaOp& op) const;

  string name_;  // Name to use for the built computation.

  // The first error encountered while building the computation.
  // This is OK until the first error is encountered.
  Status first_error_;

  // The saved stack trace from the point at which the first error occurred.
  tensorflow::SavedStackTrace first_error_backtrace_;

  // The instructions of this computation.
  std::vector<HloInstructionProto> instructions_;

  // The embedded computations used by this computation. Each computation was
  // the entry computation of some XlaComputation, the key is the unique id of
  // that XlaComputation.
  std::map<int64, HloComputationProto> embedded_;

  // The unique parameter numbers.
  tensorflow::gtl::FlatSet<int64> parameter_numbers_;

  // Mode bit that indicates whether to die when a first error is encountered.
  bool die_immediately_on_error_ = false;
};

template <typename NativeT>
XlaOp XlaBuilder::ConstantR0(NativeT value) {
  return ConstantLiteral(*Literal::CreateR0<NativeT>(value));
}

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_CLIENT_XLA_CLIENT_XLA_BUILDER_H_
