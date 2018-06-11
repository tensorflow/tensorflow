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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_BUFFER_VALUE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_BUFFER_VALUE_H_

#include <functional>
#include <string>

#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/int_type.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// Abstract class describing a value used by one of the dataflow analyses -
// TuplePointsToAnalysis or HloDataflowAnalysis.
// TODO(b/78906445) Delete this class when TuplePointsToAnalysis is unused.
//
// XLA arrays are trivially a single BufferValue. Tuples are made up of more
// than one BufferValue: an BufferValue for the pointer vector, and an
// BufferValue for each child element.
//
// Every BufferValue is defined by a particular instruction and most
// instructions define only a single BufferValue. Instructions which define a
// single BufferValue include array-shaped instructions such as Add but also
// includes Tuple-shaped instructions such as Tuple. The Tuple instruction
// defines a single BufferValue which is a vector of pointers to the values
// containing the Tuple instruction's operands. Though the result of the Tuple
// instruction includes multiple values only the top-level BufferValue (the
// vector of pointers) is defined by the Tuple instruction. The values
// containing the tuple elements are defined by earlier instructions, usually
// the operands of the Tuple instruction.
//
// Instructions which construct both the tuple *and* the tuple elements define
// more than one BufferValue. This includes (at least) tuple-shaped Constant,
// Parameter, Infeed and While instructions. These tuple-shaped instructions do
// not assemble a tuple from existing BufferValues like the Tuple instruction
// does, but rather define all the BufferValues in the tuple.
//
// Some instructions, such as Bitcast, define no buffers. These instructions
// simply forward buffers from their operands.
//
// The BufferValue object describes which HLO instruction defines a buffer and
// where within that instruction's output shape the buffer is defined. The
// location within the output shape is indicated by BufferValue::index() which
// is defined identically to the index used in ShapeUtil::GetSubshape().
// Examples:
//
// %add = Add(%foo, %bar)
// %tuple_constant = Constant({1, {42, 43}})
//
// %add defines a single array-shaped buffer BufferValue(%add, {}) which holds
// the array result of the add operation. The nested-tuple-shaped
// %tuple_constant defines 5 buffers described by the following BufferValue
// objects:
//
//   BufferValue(%tuple_constant, {})      // "Top-level" buffer: vector of
//                                         //  pointers to BufferValues at
//                                         //  indices {0} and {1}
//   BufferValue(%tuple_constant, {0})     // Holds value "1"
//   BufferValue(%tuple_constant, {1})     // Holds nested tuple: vector of
//                                         //  pointers to BufferValues at
//                                         //  indices {1, 0} and {1, 1}
//   BufferValue(%tuple_constant, {1, 0})  // Holds value "42"
//   BufferValue(%tuple_constant, {1, 1})  // Holds value "43"

class BufferValue {
 public:
  TF_LIB_GTL_DEFINE_INT_TYPE(Color, int64);

  // Id is a unique identifier for the BufferValue to facilitate efficient
  // collections of BufferValues with stable iteration order.
  using Id = int64;

  // Functions which return the size and alignment of a logical buffer in bytes.
  using SizeFunction = std::function<int64(const BufferValue&)>;
  using AlignmentFunction = std::function<int64(BufferValue::Color)>;

  virtual ~BufferValue();

  Id id() const { return id_; }

  // Return the instruction that defines the buffer.
  virtual HloInstruction* instruction() const = 0;

  // Return the index within the output of the instruction where the buffer is
  // defined. Index used defined as in ShapeUtil::GetSubshape()
  virtual const ShapeIndex& index() const = 0;

  // Return the color of the BufferValue. Differently colored buffers can not be
  // parts of the same allocation.
  Color color() const {
    CHECK_NE(color_, kInvalidColor)
        << "Should not query the color of a buffer that was never colored";
    return color_;
  }

  void set_color(Color color) {
    CHECK_NE(color, kInvalidColor)
        << "Should not set the color of a buffer to the invalid color";
    color_ = color;
  }

  bool has_color() const { return color_ != kInvalidColor; }

  // Return the shape of the buffer. This reference points into the shape field
  // of the instruction defining the buffer.  Therefore, the returned shape will
  // contain the layout of instruction, if any.
  virtual const Shape& shape() const = 0;

  // Returns true if this buffer is the top-level output buffer of the defining
  // HLO instruction. This is equivalent to index == {}.
  bool IsTopLevel() const { return index().empty(); }

  // Whether this buffer contains a tuple.
  bool IsTuple() const { return is_tuple_; }

  // Whether this buffer contains an array.
  bool IsArray() const { return is_array_; }

  // operator< is required for std::set.
  bool operator<(const BufferValue& other) const { return id_ < other.id_; }

  virtual string ToString() const = 0;

  // TODO(lauj) rename LogicalBufferProto to BufferValueProto.
  LogicalBufferProto ToProto(const SizeFunction& size_fn) const;

  // Returns the LogicalBufferProto::Location that serializes the given
  // instruction and index.
  static LogicalBufferProto::Location ToLocationProto(
      const HloInstruction& instruction, const ShapeIndex& index);

  const Color kInvalidColor = Color(-1);

 protected:
  BufferValue(HloInstruction* instruction, const ShapeIndex& index, Id id);

 private:
  // The definining instruction and index are not stored here; they can be found
  // in the LogicalBuffer and HloValue subclasses. This class exists only to
  // support migrations from TuplePointsToAnalysis to HloDataflowAnalysis, by
  // allowing abstract use of LogicalBuffer or HloValue. After those migrations
  // are complete, this class should be deleted (b/78906445). Because we plan to
  // delete LogicalBuffer and this class, we don't refactor all the shared
  // features from LogicalBuffer and HloValue into this class.
  Id id_ : 62;
  bool is_array_ : 1;
  bool is_tuple_ : 1;
  Color color_ = kInvalidColor;
};

std::ostream& operator<<(std::ostream& out, const BufferValue& buffer);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_BUFFER_VALUE_H_
