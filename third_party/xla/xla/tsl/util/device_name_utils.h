/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_TSL_UTIL_DEVICE_NAME_UTILS_H_
#define XLA_TSL_UTIL_DEVICE_NAME_UTILS_H_

#include <string>

#include "xla/tsl/platform/status.h"
#include "tsl/platform/stringpiece.h"

namespace tsl {

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
  static std::string FullName(const std::string& job, int replica, int task,
                              const std::string& type, int id);

  // TODO(b/278776328): Convert this to a Protobuf, since emptiness of a field
  // is a standardized pattern in Protobuf.
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

    bool operator!=(const ParsedName& other) const {
      return !operator==(other);
    }

    bool operator<(const ParsedName& other) const {
      if (has_job != other.has_job) return !has_job;
      if (has_job) {
        if (job < other.job) {
          return true;
        }
        if (job > other.job) {
          return false;
        }
      }
      if (has_replica != other.has_replica) return !has_replica;
      if (has_replica) {
        if (replica < other.replica) {
          return true;
        }
        if (replica > other.replica) {
          return false;
        }
      }
      if (has_task != other.has_task) return !has_task;
      if (has_task) {
        if (task < other.task) {
          return true;
        }
        if (task > other.task) {
          return false;
        }
      }
      if (has_type != other.has_type) return !has_type;
      if (has_type) {
        if (type < other.type) {
          return true;
        }
        if (type > other.type) {
          return false;
        }
      }
      if (has_id != other.has_id) return !has_id;
      if (has_id) {
        if (id < other.id) {
          return true;
        }
        if (id > other.id) {
          return false;
        }
      }
      return false;
    }

    template <typename H>
    friend H AbslHashValue(H h, const ParsedName& n) {
      return H::combine(std::move(h), n.has_job ? n.job : "",
                        n.has_replica ? n.replica : 0, n.has_task ? n.task : 0,
                        n.has_type ? n.type : "", n.has_id ? n.id : 0);
    }

    bool has_job = false;
    std::string job;
    bool has_replica = false;
    int replica = 0;
    bool has_task = false;
    int task = 0;
    bool has_type = false;
    std::string type;
    bool has_id = false;
    int id = 0;
  };

  // Parses the device name, first as a full name, then, if it fails, as a
  // global one. Returns `false` if both attempts fail.
  static bool ParseFullOrLocalName(absl::string_view fullname,
                                   ParsedName* parsed);

  // Parses "fullname" into "*parsed". Returns true iff succeeds.
  // Legacy names like "/cpu:0" that don't contain "device",
  // are parsed to mean their current counterparts "/device:CPU:0". More
  // specifically, the lower case "cpu" and "gpu" is capitalized and "device"
  // is added. "/tpu:0" is not treated the same way - it has use the current
  // full syntax.
  // Also, note that lower case "cpu" and "gpu" device types in current syntax
  // are not capitalized. For example, "/device:CPU:0" is different from
  // "/device:cpu:0"
  static bool ParseFullName(absl::string_view fullname, ParsedName* parsed);

  // Canonicalizes "fullname" into "*canonical_name". Uses a fully specified
  // basename to fill in fields that are missing. Accepts both legacy, newer
  // and local versions of the device spec. Returns the newer version of the
  // device spec. If we were unable to interpret / parse "fullname" returns
  // an error and *canonical_name is set to "".
  static absl::Status CanonicalizeDeviceName(absl::string_view fullname,
                                             absl::string_view basename,
                                             std::string* canonical_name);

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

  // Makes minimal changes to more_specific so that it becomes a
  // specification of less_specific.
  static void EnsureSpecification(ParsedName* more_specific,
                                  const ParsedName& less_specific);

  // Like IsSpecification, but the second argument "name" must have a
  // non-wildcard value for all of its components.
  static bool IsCompleteSpecification(const ParsedName& pattern,
                                      const ParsedName& name);

  // True iff there exists any possible device name that is a specification of
  // both "a" and "b".
  static bool AreCompatibleDevNames(const ParsedName& a, const ParsedName& b);

  // Merges the device specifications in "*target" and "other", and
  // stores the result in "*target". Returns OK if "*target" and
  // "other" are compatible, otherwise returns an error.
  static absl::Status MergeDevNames(ParsedName* target,
                                    const ParsedName& other) {
    return MergeDevNames(target, other, false);
  }
  static absl::Status MergeDevNames(ParsedName* target, const ParsedName& other,
                                    bool allow_soft_placement);
  // Same as MergeDevNames with allow_soft_placement=true, but instead of
  // clearing conflicting fields, overrides them with `other`'s values.
  static absl::Status MergeOverrideDevNames(ParsedName* target,
                                            const ParsedName& other);

  // Merges the device specifications in "*target" and "other", and
  // stores the result in "*target" by setting all unset values in target with
  // corresponding set ones in other.
  static void MergeUnsetDevNames(ParsedName* target, const ParsedName& other);

  // Returns true iff devices identified by 'src' and 'dst' are in the
  // same address space.
  static bool IsSameAddressSpace(absl::string_view src, absl::string_view dst);
  static bool IsSameAddressSpace(const ParsedName& src, const ParsedName& dst);

  // Returns true iff devices identified by 'a' and 'b' are in different
  // address space.
  static bool IsDifferentAddressSpace(const ParsedName& a, const ParsedName& b);

  // Returns the an address space specification containing only the
  // job/replica/task of the given name.
  static const ParsedName AddressSpace(const ParsedName& name);

  // Returns the local device given its "type" and "id".
  static std::string LocalName(absl::string_view type, int id);

  // Returns a short local device name (cpu:0, gpu:1, etc) based on
  // the given fullname.
  static std::string LocalName(absl::string_view fullname);

  // If "name" is a valid local device name (cpu:0, gpu:1, etc.),
  // fills in parsed.type and parsed.id accordingly. Returns true iff
  // succeeds.
  static bool ParseLocalName(absl::string_view name, ParsedName* parsed);

  // Splits a fully-qualified device name into a task identifier and a
  // relative device identifier. It first parses "name" using
  // ParseFullName(), then assigns *task with everything except for
  // the local device component, and assigns the relative device
  // component into *device.  This function will still return true if
  // the task component is empty, but it requires the relative device
  // component to be fully specified.
  static bool SplitDeviceName(absl::string_view name, std::string* task,
                              std::string* device);

  // Get the task name from ParsedName. Return false if the task component is
  // not fully specified.
  static bool GetTaskName(const ParsedName& pn, std::string* task);

  static std::string ParsedNameToString(const ParsedName& pn);

  // Returns canonical and legacy full names for the given parsed
  // device name 'pn'. The returned string names are often useful to
  // look up devices from a mapping.
  static std::vector<string> GetNamesForDeviceMappings(const ParsedName& pn);

  // Returns canonical and legacy local names for the given parsed device name
  // 'pn'. The returned string names are often useful to look up devices from a
  // mapping.
  static std::vector<string> GetLocalNamesForDeviceMappings(
      const ParsedName& pn);

  // Returns name of the CPU:0 device on the same host as the device
  // `device_name`.
  static absl::Status DeviceNameToCpuDeviceName(const std::string& device_name,
                                                std::string* host_device_name);

  static bool CompareFullNames(absl::string_view a, absl::string_view b) {
    ParsedName parsed_a;
    ParsedName parsed_b;
    bool a_status = ParseFullName(a, &parsed_a);
    bool b_status = ParseFullName(b, &parsed_b);
    // Orders unparsable names first.
    if (a_status != b_status) return !a_status;
    if (!a_status) return a < b;
    return parsed_a < parsed_b;
  }
};

std::ostream& operator<<(std::ostream& os,
                         const DeviceNameUtils::ParsedName& x);

}  // namespace tsl

#endif  // XLA_TSL_UTIL_DEVICE_NAME_UTILS_H_
