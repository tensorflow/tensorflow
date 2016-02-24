/* Copyright 2015 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_UTIL_DEVICE_NAME_UTILS_H_
#define TENSORFLOW_UTIL_DEVICE_NAME_UTILS_H_

#include <string>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"

namespace tensorflow {

// In TensorFlow a device name is a string of the following form:
//   /job:<name>/replica:<replica>/task:<task>/device:<type>:<device_num>
//
// <name> is a short identifier conforming to the regexp
//     [a-zA-Z][_a-zA-Z]*
// <type> is a supported device type (e.g. 'cpu' or 'gpu')
// <replica>, <task>, <device_num> are small non-negative integers and are
// densely allocated (except in tests).
//
// For some purposes, we also allow device patterns, which can specify
// some or none of the specific fields above, with missing components,
// or "<component>:*" indicating "any value allowed for that component.
//
// For example:
//   "/job:param_server"   - Consider any devices in the "param_server" job
//   "/device:cpu:*"       - Consider any cpu devices in any job/task/replica
//   "/job:*/replica:*/task:*/device:cpu:*"  - Consider any cpu devices in any
//                                             job/task/replica
//   "/job:w/replica:0/task:0/device:gpu:*"  - Consider any gpu devices in
//                                             replica 0, task 0, of job "w"
class DeviceNameUtils {
 public:
  // Returns a fully qualified device name given the parameters.
  static string FullName(const string& job, int replica, int task,
                         const string& type, int id);

  struct ParsedName {
    void Clear() {
      has_job = false;
      has_replica = false;
      has_task = false;
      has_type = false;
      has_id = false;
      job.clear();
      replica = 0;
      task = 0;
      type.clear();
      id = 0;
    }

    bool operator==(const ParsedName& other) const {
      return (has_job ? (other.has_job && job == other.job) : !other.has_job) &&
             (has_replica ? (other.has_replica && replica == other.replica)
                          : !other.has_replica) &&
             (has_task ? (other.has_task && task == other.task)
                       : !other.has_task) &&
             (has_type ? (other.has_type && type == other.type)
                       : !other.has_type) &&
             (has_id ? (other.has_id && id == other.id) : !other.has_id);
    }

    bool has_job = false;
    string job;
    bool has_replica = false;
    int replica = 0;
    bool has_task = false;
    int task = 0;
    bool has_type = false;
    string type;
    bool has_id = false;
    int id = 0;
  };
  // Parses "fullname" into "*parsed". Returns true iff succeeds.
  static bool ParseFullName(StringPiece fullname, ParsedName* parsed);

  // Returns true if "name" specifies any non-trivial constraint on the device.
  static bool HasSomeDetails(const ParsedName& name) {
    return name.has_job || name.has_replica || name.has_task || name.has_type ||
           name.has_id;
  }

  // Returns true if more_specific is a specification of
  // less_specific, i.e. everywhere that less-specific has a
  // non-wildcard component value, more_specific has the same value
  // for that component.
  static bool IsSpecification(const ParsedName& less_specific,
                              const ParsedName& more_specific);

  // Like IsSpecification, but the second argument "name" must have a
  // non-wildcard value for all of its components.
  static bool IsCompleteSpecification(const ParsedName& pattern,
                                      const ParsedName& name);

  // True iff there exists any possible complete device name that is
  // a specification of both "a" and "b".
  static inline bool AreCompatibleDevNames(const ParsedName& a,
                                           const ParsedName& b) {
    return IsSpecification(a, b) || IsSpecification(b, a);
  }

  // Merges the device specifications in "*target" and "other", and
  // stores the result in "*target". Returns OK if "*target" and
  // "other" are compatible, otherwise returns an error.
  static Status MergeDevNames(ParsedName* target, const ParsedName& other) {
    return MergeDevNames(target, other, false);
  }
  static Status MergeDevNames(ParsedName* target, const ParsedName& other,
                              bool allow_soft_placement);

  // Returns true iff devices identified by 'src' and 'dst' are in the
  // same address space.
  static bool IsSameAddressSpace(StringPiece src, StringPiece dst);
  static bool IsSameAddressSpace(const ParsedName& src, const ParsedName& dst);

  // Returns the local device given its "type" and "id".
  static string LocalName(StringPiece type, int id);

  // Returns a short local device name (cpu:0, gpu:1, etc) based on
  // the given fullname.
  static string LocalName(StringPiece fullname);

  // If "name" is a valid local device name (cpu:0, gpu:1, etc.),
  // fills in parsed.type and parsed.id accordingly. Returns true iff
  // succeeds.
  static bool ParseLocalName(StringPiece name, ParsedName* parsed);

  // Splits a fully-qualified device name into a task identifier and a
  // relative device identifier. It first parses "name" using
  // ParseFullName(), then assigns *task with everything except for
  // the local device component, and assigns the relative device
  // component into *device.  This function will still return true if
  // the task component is empty, but it requires the relative device
  // component to be fully specified.
  static bool SplitDeviceName(StringPiece name, string* task, string* device);

  static string ParsedNameToString(const ParsedName& pn);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_UTIL_DEVICE_NAME_UTILS_H_
