/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ML_ADJACENT_LIB_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ML_ADJACENT_LIB_H_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ml_adj {

/// Standard Types ///

// Length of axis.
typedef uint32_t dim_t;

// Dimensions of data.
typedef std::vector<dim_t> dims_t;

// 1d index into data.
typedef uint64_t ind_t;

// Integral type of data element.
enum etype_t : uint8_t {
  i32 = 0,
  f32 = 1,
  f64 = 2,
};

// Size in bytes of data element.
typedef uint8_t width_t;

namespace data {

// Get size (int bytes) of single element of type.
inline width_t TypeWidth(etype_t type) {
  switch (type) {
    case etype_t::i32:
      return sizeof(int32_t);
    case etype_t::f32:
      return sizeof(float);
    case etype_t::f64:
      return sizeof(double);
  }
}

/// Input/Output Wrapper for Algos ///

// Encapsulates a input or output to an algorithm. Management of
// buffers is to be implemented outside of algorithms.

// Read only wrapper.
class DataRef {
 public:
  explicit DataRef(etype_t type) : element_type_(type) {}

  DataRef(const DataRef&) = delete;
  DataRef(DataRef&&) = delete;
  DataRef& operator=(const DataRef&) = delete;
  DataRef& operator=(DataRef&&) = delete;

  // Read only buffer, allocated to be of size == Bytes().
  virtual const void* Data() const = 0;

  // Number of elements currently allocated.
  virtual ind_t NumElements() const = 0;

  // Size of buffer.
  virtual size_t Bytes() const = 0;

  // Type of elements.
  etype_t Type() const { return element_type_; }

  // Implicit dimensions of buffer.
  const dims_t& Dims() const { return dims_; }

  virtual ~DataRef() = default;

 protected:
  dims_t dims_;
  // Data can be reshaped but not change types.
  const etype_t element_type_;
};

// Read/write wrapper which can be resized.
class MutableDataRef : public DataRef {
 public:
  using DataRef::Data;

  explicit MutableDataRef(etype_t type) : DataRef(type) {}

  // Takes ownership of dims_t. Implementations must set bytes and
  // `num_elements_` field.
  virtual void Resize(dims_t&& dims) = 0;

  // Write buffer, allocated to be of size == Bytes().
  virtual void* Data() = 0;
};

}  // namespace data

namespace algo {

/// Function Interface for Operations on DataRefs ///

// Inputs to algorithm.
typedef std::vector<data::DataRef*> InputPack;
// Outputs to algorithm.
typedef std::vector<data::MutableDataRef*> OutputPack;

// Generic algorithm, computes outputs via inputs.
typedef void ComputeFunc(const InputPack& inputs, const OutputPack& outputs);

// Optional hook to compute output shapes when they are non data-dependent.
// This is a place-holder for now.
// TODO(b/292143456) Figure out what the signature of this should be.
typedef size_t ShapeFunc();

struct Algo {
  ComputeFunc* process = nullptr;
  ShapeFunc* output_size = nullptr;
};

}  // namespace algo
}  // namespace ml_adj

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ML_ADJACENT_LIB_H_
