/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");;
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CORE_FUNCTION_TRACE_TYPE_STANDARD_TRACE_TYPE_H_
#define TENSORFLOW_CORE_FUNCTION_TRACE_TYPE_STANDARD_TRACE_TYPE_H_

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace tensorflow {
namespace trace_type {

// Represents the type of object(s) from a tf.function point of view.
// C++ counterpart of TraceType as defined in tensorflow/python/types/trace.py.
class TraceType {
 public:
  // Returns a new copy of the original object.
  virtual std::unique_ptr<TraceType> clone() const = 0;

  // Returns True if the object is a subtype of "other".
  virtual bool is_subtype_of(const TraceType& other) const = 0;

  // Returns a new object representing the "most specific common supertype" of
  // original object and "others".
  // Returns nullptr if a supertype does not exist.
  virtual std::unique_ptr<TraceType> join(
      const std::vector<const TraceType*>& others) const = 0;

  // Returns a string representation of the object.
  virtual std::string to_string() const = 0;

  // Returns a hash value for the object.
  virtual std::size_t hash() const = 0;

  // Default equality operator.
  virtual bool operator==(const TraceType& other) const = 0;

  // Inequality is strictly defined through the equality operator.
  bool operator!=(const TraceType& other) const { return !(*this == other); }

  virtual ~TraceType() {}
};

}  // namespace trace_type
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FUNCTION_TRACE_TYPE_STANDARD_TRACE_TYPE_H_
