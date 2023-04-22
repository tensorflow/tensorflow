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

#ifndef TENSORFLOW_CC_FRAMEWORK_OPS_H_
#define TENSORFLOW_CC_FRAMEWORK_OPS_H_

#include <type_traits>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {

/// @defgroup core Core Tensorflow API

class Output;

/// @addtogroup core
/// @{

/// Represents a node in the computation graph.
class Operation {
 public:
  Operation() : node_(nullptr) {}
  explicit Operation(Node* n);

  int32 num_inputs() const { return node_->num_inputs(); }
  DataType input_type(int32 o) const { return node_->input_type(o); }
  Output input(int32 i) const;

  int32 num_outputs() const { return node_->num_outputs(); }
  DataType output_type(int32 o) const { return node_->output_type(o); }
  Output output(int32 i) const;

  Node* node() const { return node_; }

  uint64 hash(int32 index) const;

  bool operator==(const Operation& other) const { return node_ == other.node_; }

 private:
  typedef std::vector<std::pair<Node*, int32>> Inputs;
  static Inputs GetInputs(Node* node);

  Inputs inputs_;
  Node* node_;
};

/// Represents a tensor value produced by an Operation.
class Output {
 public:
  Output() = default;
  explicit Output(Node* n) : op_(n) {}
  Output(Node* n, int32 index) : op_(n), index_(index) {}
  Output(const Operation& op, int32 index) : op_(op), index_(index) {}

  Operation op() const { return op_; }
  Node* node() const { return op().node(); }
  int32 index() const { return index_; }
  DataType type() const { return op_.output_type(index_); }
  std::string name() const {
    return strings::StrCat(node()->name(), ":", index());
  }
  bool operator==(const Output& other) const {
    return op_ == other.op_ && index_ == other.index_;
  }

  uint64 hash() const { return op_.hash(index_); }

 private:
  Operation op_ = Operation(nullptr);
  int32 index_ = 0;
};

/// Hash class that can be used for e.g. storing Outputs in an unordered_map
struct OutputHash {
  std::size_t operator()(const Output& output) const {
    return Hash64Combine(std::hash<Node*>()(output.node()),
                         std::hash<int32>()(output.index()));
  }
};

/// Represents a tensor value that can be used as an operand to an Operation.
class Input {
 public:
  /// Initializer enables constructing an Input object from various kinds of C++
  /// constants such as simple primitive constants and nested initializer lists
  /// representing a multi-dimensional array. Initializer constructors are all
  /// templates, so the aforementioned kinds of C++ constants can be used to
  /// construct an Initializer. Initializer stores the value it got constructed
  /// with in a Tensor object.
  struct Initializer {
    /// Construct from a scalar value of an arithmetic type or a type that can
    /// be converted to a string (eg. a string literal).
    template <typename T, typename = typename std::enable_if<
                              std::is_arithmetic<T>::value ||
                              std::is_convertible<T, std::string>::value>::type>
    Initializer(const T& v) {  // NOLINT(runtime/explicit)
      typedef typename RealType<T>::type RealT;
      Tensor t(DataTypeToEnum<RealT>::v(), TensorShape());
      t.flat<RealT>()(0) = RealT(v);
      tensor = t;
    }

    Initializer(const Tensor& t) : tensor(t) {}  // NOLINT(runtime/explicit)

    /// Construct from a scalar value and an explicit shape
    template <typename T, typename = typename std::enable_if<
                              std::is_arithmetic<T>::value ||
                              std::is_convertible<T, std::string>::value>::type>
    Initializer(const T& v, const TensorShape& shape) {
      typedef typename RealType<T>::type RealT;
      Tensor t(DataTypeToEnum<RealT>::v(), shape);
      for (int64_t i = 0; i < t.NumElements(); ++i) {
        t.flat<RealT>()(i) = RealT(v);
      }
      tensor = t;
    }

    /// Construct from a initializer list of scalars (a one-dimensional tensor).
    template <typename T, typename = typename std::enable_if<
                              std::is_arithmetic<T>::value ||
                              std::is_convertible<T, std::string>::value>::type>
    Initializer(
        const std::initializer_list<T>& v) {  // NOLINT(runtime/explicit)
      typedef typename RealType<T>::type RealT;
      Tensor t(DataTypeToEnum<RealT>::v(),
               TensorShape{static_cast<int>(v.size())});
      std::copy_n(v.begin(), v.size(), t.flat<RealT>().data());
      tensor = t;
    }

    /// Construct from a initializer list of scalars and an explicit shape.
    template <typename T, typename = typename std::enable_if<
                              std::is_arithmetic<T>::value ||
                              std::is_convertible<T, std::string>::value>::type>
    Initializer(const std::initializer_list<T>& v, const TensorShape& shape) {
      typedef typename RealType<T>::type RealT;
      Tensor t(DataTypeToEnum<RealT>::v(), shape);
      if (t.NumElements() != static_cast<int64>(v.size())) {
        status = errors::InvalidArgument(
            "Cannot construct a tensor with ", t.NumElements(),
            " from an initializer list with ", v.size(), " elements");
        return;
      }
      std::copy_n(v.begin(), v.size(), t.flat<RealT>().data());
      tensor = t;
    }

    /// Construct a multi-dimensional tensor from a nested initializer
    /// list. Note that C++ syntax allows nesting of arbitrarily typed
    /// initializer lists, so such invalid initializers cannot be disallowed at
    /// compile time. This function performs checks to make sure that the nested
    /// initializer list is indeed a valid multi-dimensional tensor.
    Initializer(const std::initializer_list<Initializer>& v);

    // START_SKIP_DOXYGEN
    template <typename T, bool = std::is_convertible<T, std::string>::value>
    struct RealType {
      typedef tstring type;
    };

    template <typename T>
    struct RealType<T, false> {
      typedef T type;
    };
    // END_SKIP_DOXYGEN

    TensorProto AsTensorProto() {
      TensorProto tensor_proto;
      if (tensor.NumElements() > 1) {
        tensor.AsProtoTensorContent(&tensor_proto);
      } else {
        tensor.AsProtoField(&tensor_proto);
      }
      return tensor_proto;
    }

    Status status;
    Tensor tensor;
  };

  /// All of Input's constructors are implicit. Input can be implicitly
  /// constructed from the following objects :
  /// * Output: This is so that the output of an Operation can be directly used
  ///   as the input to a op wrapper, which takes Inputs.
  /// * A scalar, or a multi-dimensional tensor specified as a recursive
  ///   initializer list. This enables directly passing constants as
  ///   inputs to op wrappers.
  /// * A Tensor object.
  Input(const Output& o) : output_(o) {}  // NOLINT(runtime/explicit)

  template <typename T, typename = typename std::enable_if<
                            std::is_arithmetic<T>::value ||
                            std::is_convertible<T, std::string>::value>::type>
  Input(const T& v)  // NOLINT(runtime/explicit)
      : Input(Initializer(v)) {}

  Input(const Initializer& init)  // NOLINT(runtime/explicit)
      : status_(init.status),
        tensor_(init.tensor) {}

  Input(const Tensor& t)  // NOLINT(runtime/explicit)
      : status_(Status::OK()),
        tensor_(t) {}

  Input(const std::initializer_list<Initializer>&
            init) {  // NOLINT(runtime/explicit)
    for (const auto& i : init) {
      if (!i.status.ok()) {
        status_ = i.status;
        return;
      }
    }
    tensor_ = Initializer(init).tensor;
  }

  /// Constructor specifying a node name, index and datatype. This should only
  /// be used for specifying a backward edge, needed by control flow.
  Input(const std::string& name, int32 i, DataType dt)
      : node_name_(name), index_(i), data_type_(dt) {}

  Node* node() const { return output_.node(); }
  std::string node_name() const { return node_name_; }
  int32 index() const { return node_name_.empty() ? output_.index() : index_; }
  DataType data_type() const { return data_type_; }
  Status status() const { return status_; }
  const Tensor& tensor() const { return tensor_; }

 private:
  Status status_;
  Output output_ = Output(Operation(nullptr), 0);
  Tensor tensor_;
  const std::string node_name_ = "";
  int32 index_ = 0;
  DataType data_type_ = DT_INVALID;
};

/// A type for representing the output of ops that produce more than one output,
/// or a list of tensors.
typedef std::vector<Output> OutputList;

/// A type for representing the input to ops that require a list of tensors.
class InputList {
 public:
  /// Implicitly convert a list of outputs to a list of inputs. This is useful
  /// to write code such as ops::Concat(ops::Split(x, 4)).
  InputList(const OutputList& out) {  // NOLINT(runtime/explicit)
    for (auto const& x : out) {
      inputs_.push_back(x);
    }
  }

  InputList(
      const std::initializer_list<Input>& inputs)  // NOLINT(runtime/explicit)
      : inputs_(inputs.begin(), inputs.end()) {}

  InputList(const tensorflow::gtl::ArraySlice<Input>&
                inputs)  // NOLINT(runtime/explicit)
      : inputs_(inputs.begin(), inputs.end()) {}

  InputList(
      const std::initializer_list<Output>& out) {  // NOLINT(runtime/explicit)
    for (auto const& x : out) {
      inputs_.push_back(x);
    }
  }

  typename std::vector<Input>::iterator begin() { return inputs_.begin(); }
  typename std::vector<Input>::iterator end() { return inputs_.end(); }
  typename std::vector<Input>::const_iterator begin() const {
    return inputs_.begin();
  }
  typename std::vector<Input>::const_iterator end() const {
    return inputs_.end();
  }

 private:
  std::vector<Input> inputs_;
};

/// @}

}  // namespace tensorflow

#endif  // TENSORFLOW_CC_FRAMEWORK_OPS_H_
