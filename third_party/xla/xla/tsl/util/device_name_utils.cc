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

#include "xla/tsl/util/device_name_utils.h"

#include <algorithm>

#include "tsl/platform/errors.h"

namespace tsl {

static bool IsAlpha(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

static bool IsAlphaNumOrUnderscore(char c) {
  return IsAlpha(c) || (c >= '0' && c <= '9') || c == '_';
}

// Returns true iff "in" is a valid job name.
static bool IsJobName(absl::string_view in) {
  return !in.empty() && IsAlpha(in.front()) &&
         std::all_of(in.begin(), in.end(), IsAlphaNumOrUnderscore);
}

static bool ConsumePrefix(absl::string_view* in, string* out,
                          absl::string_view prefix_terminators) {
  if (in->empty() || !IsAlpha(in->front())) return false;
  const auto end_it =
      std::find_first_of(in->begin(), in->end(), prefix_terminators.begin(),
                         prefix_terminators.end());
  if (!std::all_of(in->begin(), end_it, IsAlphaNumOrUnderscore)) {
    return false;
  }
  out->assign(in->begin(), end_it);
  in->remove_prefix(end_it - in->begin());
  return true;
}

// Returns true and fills in "*job" iff "*in" starts with a job name.
static bool ConsumeJobName(absl::string_view* in, string* job) {
  return ConsumePrefix(in, job, "/");
}

// Returns true and fills in "*device_type" iff "*in" starts with a device type
// name.
static bool ConsumeDeviceType(absl::string_view* in, string* device_type) {
  return ConsumePrefix(in, device_type, "/:");
}

// Returns true and fills in "*val" iff "*in" starts with a decimal
// number.
static bool ConsumeNumber(absl::string_view* in, int* val) {
  uint64 tmp;
  if (str_util::ConsumeLeadingDigits(in, &tmp)) {
    *val = tmp;
    return true;
  } else {
    return false;
  }
}

// Returns a fully qualified device name given the parameters.
static string DeviceName(const string& job, int replica, int task,
                         const string& device_prefix, const string& device_type,
                         int id) {
  CHECK(IsJobName(job)) << job;
  CHECK_LE(0, replica);
  CHECK_LE(0, task);
  CHECK(!device_type.empty());
  CHECK_LE(0, id);
  return strings::StrCat("/job:", job, "/replica:", replica, "/task:", task,
                         device_prefix, device_type, ":", id);
}

/* static */
string DeviceNameUtils::FullName(const string& job, int replica, int task,
                                 const string& type, int id) {
  return DeviceName(job, replica, task, "/device:", type, id);
}

namespace {
string LegacyName(const string& job, int replica, int task, const string& type,
                  int id) {
  return DeviceName(job, replica, task, "/", absl::AsciiStrToLower(type), id);
}
}  // anonymous namespace

bool DeviceNameUtils::ParseFullName(absl::string_view fullname, ParsedName* p) {
  p->Clear();
  if (fullname == "/") {
    return true;
  }
  while (!fullname.empty()) {
    bool progress = false;
    if (absl::ConsumePrefix(&fullname, "/job:")) {
      p->has_job = !absl::ConsumePrefix(&fullname, "*");
      if (p->has_job && !ConsumeJobName(&fullname, &p->job)) {
        return false;
      }
      progress = true;
    }
    if (absl::ConsumePrefix(&fullname, "/replica:")) {
      p->has_replica = !absl::ConsumePrefix(&fullname, "*");
      if (p->has_replica && !ConsumeNumber(&fullname, &p->replica)) {
        return false;
      }
      progress = true;
    }
    if (absl::ConsumePrefix(&fullname, "/task:")) {
      p->has_task = !absl::ConsumePrefix(&fullname, "*");
      if (p->has_task && !ConsumeNumber(&fullname, &p->task)) {
        return false;
      }
      progress = true;
    }
    if (absl::ConsumePrefix(&fullname, "/device:")) {
      p->has_type = !absl::ConsumePrefix(&fullname, "*");
      if (p->has_type && !ConsumeDeviceType(&fullname, &p->type)) {
        return false;
      }
      if (!absl::ConsumePrefix(&fullname, ":")) {
        p->has_id = false;
      } else {
        p->has_id = !absl::ConsumePrefix(&fullname, "*");
        if (p->has_id && !ConsumeNumber(&fullname, &p->id)) {
          return false;
        }
      }
      progress = true;
    }

    // Handle legacy naming convention for cpu and gpu.
    if (absl::ConsumePrefix(&fullname, "/cpu:") ||
        absl::ConsumePrefix(&fullname, "/CPU:")) {
      p->has_type = true;
      p->type = "CPU";  // Treat '/cpu:..' as uppercase '/device:CPU:...'
      p->has_id = !absl::ConsumePrefix(&fullname, "*");
      if (p->has_id && !ConsumeNumber(&fullname, &p->id)) {
        return false;
      }
      progress = true;
    }
    if (absl::ConsumePrefix(&fullname, "/gpu:") ||
        absl::ConsumePrefix(&fullname, "/GPU:")) {
      p->has_type = true;
      p->type = "GPU";  // Treat '/gpu:..' as uppercase '/device:GPU:...'
      p->has_id = !absl::ConsumePrefix(&fullname, "*");
      if (p->has_id && !ConsumeNumber(&fullname, &p->id)) {
        return false;
      }
      progress = true;
    }

    if (!progress) {
      return false;
    }
  }
  return true;
}

bool DeviceNameUtils::ParseFullOrLocalName(absl::string_view fullname,
                                           ParsedName* p) {
  return ParseFullName(fullname, p) || ParseLocalName(fullname, p);
}

namespace {

void CompleteName(const DeviceNameUtils::ParsedName& parsed_basename,
                  DeviceNameUtils::ParsedName* parsed_name) {
  if (!parsed_name->has_job) {
    parsed_name->job = parsed_basename.job;
    parsed_name->has_job = true;
  }
  if (!parsed_name->has_replica) {
    parsed_name->replica = parsed_basename.replica;
    parsed_name->has_replica = true;
  }
  if (!parsed_name->has_task) {
    parsed_name->task = parsed_basename.task;
    parsed_name->has_task = true;
  }
  if (!parsed_name->has_type) {
    parsed_name->type = parsed_basename.type;
    parsed_name->has_type = true;
  }
  if (!parsed_name->has_id) {
    parsed_name->id = parsed_basename.id;
    parsed_name->has_id = true;
  }
}

}  // namespace

/* static */
absl::Status DeviceNameUtils::CanonicalizeDeviceName(absl::string_view fullname,
                                                     absl::string_view basename,
                                                     string* canonical_name) {
  *canonical_name = "";
  ParsedName parsed_basename;
  if (!ParseFullName(basename, &parsed_basename)) {
    return errors::InvalidArgument("Could not parse basename: ", basename,
                                   " into a device specification.");
  }
  if (!(parsed_basename.has_job && parsed_basename.has_replica &&
        parsed_basename.has_task && parsed_basename.has_type &&
        parsed_basename.has_id)) {
    return errors::InvalidArgument("Basename: ", basename,
                                   " should be fully "
                                   "specified.");
  }
  ParsedName parsed_name;
  if (ParseLocalName(fullname, &parsed_name)) {
    CompleteName(parsed_basename, &parsed_name);
    *canonical_name = ParsedNameToString(parsed_name);
    return absl::OkStatus();
  }
  if (ParseFullName(fullname, &parsed_name)) {
    CompleteName(parsed_basename, &parsed_name);
    *canonical_name = ParsedNameToString(parsed_name);
    return absl::OkStatus();
  }
  return errors::InvalidArgument("Could not parse ", fullname,
                                 " into a device "
                                 "specification.");
}

/* static */
string DeviceNameUtils::ParsedNameToString(const ParsedName& pn) {
  string buf;
  if (pn.has_job) strings::StrAppend(&buf, "/job:", pn.job);
  if (pn.has_replica) strings::StrAppend(&buf, "/replica:", pn.replica);
  if (pn.has_task) strings::StrAppend(&buf, "/task:", pn.task);
  if (pn.has_type) {
    strings::StrAppend(&buf, "/device:", pn.type, ":");
    if (pn.has_id) {
      strings::StrAppend(&buf, pn.id);
    } else {
      strings::StrAppend(&buf, "*");
    }
  }
  return buf;
}

/* static */
bool DeviceNameUtils::IsSpecification(const ParsedName& less_specific,
                                      const ParsedName& more_specific) {
  if (less_specific.has_job &&
      (!more_specific.has_job || (less_specific.job != more_specific.job))) {
    return false;
  }
  if (less_specific.has_replica &&
      (!more_specific.has_replica ||
       (less_specific.replica != more_specific.replica))) {
    return false;
  }
  if (less_specific.has_task &&
      (!more_specific.has_task || (less_specific.task != more_specific.task))) {
    return false;
  }
  if (less_specific.has_type &&
      (!more_specific.has_type || (less_specific.type != more_specific.type))) {
    return false;
  }
  if (less_specific.has_id &&
      (!more_specific.has_id || (less_specific.id != more_specific.id))) {
    return false;
  }
  return true;
}

/* static */
bool DeviceNameUtils::AreCompatibleDevNames(const ParsedName& a,
                                            const ParsedName& b) {
  if (a.has_job && b.has_job && (a.job != b.job)) {
    return false;
  }
  if (a.has_replica && b.has_replica && (a.replica != b.replica)) {
    return false;
  }
  if (a.has_task && b.has_task && (a.task != b.task)) {
    return false;
  }
  if (a.has_type && b.has_type && (a.type != b.type)) {
    return false;
  }
  if (a.has_id && b.has_id && (a.id != b.id)) {
    return false;
  }
  return true;
}

void DeviceNameUtils::EnsureSpecification(ParsedName* more_specific,
                                          const ParsedName& less_specific) {
  if (less_specific.has_job) {
    more_specific->has_job = true;
    more_specific->job = less_specific.job;
  }
  if (less_specific.has_replica) {
    more_specific->has_replica = true;
    more_specific->replica = less_specific.replica;
  }
  if (less_specific.has_task) {
    more_specific->has_task = true;
    more_specific->task = less_specific.task;
  }
  if (less_specific.has_type) {
    more_specific->has_type = true;
    more_specific->type = less_specific.type;
  }
  if (less_specific.has_id) {
    more_specific->has_id = true;
    more_specific->id = less_specific.id;
  }
}

/* static */
bool DeviceNameUtils::IsCompleteSpecification(const ParsedName& pattern,
                                              const ParsedName& name) {
  CHECK(name.has_job && name.has_replica && name.has_task && name.has_type &&
        name.has_id);

  if (pattern.has_job && (pattern.job != name.job)) return false;
  if (pattern.has_replica && (pattern.replica != name.replica)) return false;
  if (pattern.has_task && (pattern.task != name.task)) return false;
  if (pattern.has_type && (pattern.type != name.type)) return false;
  if (pattern.has_id && (pattern.id != name.id)) return false;
  return true;
}

namespace {
absl::Status MergeDevNamesImpl(DeviceNameUtils::ParsedName* target,
                               const DeviceNameUtils::ParsedName& other,
                               bool allow_soft_placement,
                               bool override_conflicts) {
  const auto& ParsedNameToString = DeviceNameUtils::ParsedNameToString;
  if (other.has_job) {
    if (target->has_job && target->job != other.job) {
      return errors::InvalidArgument(
          "Cannot merge devices with incompatible jobs: '",
          ParsedNameToString(*target), "' and '", ParsedNameToString(other),
          "'");
    } else {
      target->has_job = other.has_job;
      target->job = other.job;
    }
  }

  if (other.has_replica) {
    if (target->has_replica && target->replica != other.replica) {
      return errors::InvalidArgument(
          "Cannot merge devices with incompatible replicas: '",
          ParsedNameToString(*target), "' and '", ParsedNameToString(other),
          "'");
    } else {
      target->has_replica = other.has_replica;
      target->replica = other.replica;
    }
  }

  if (other.has_task) {
    if (target->has_task && target->task != other.task) {
      return errors::InvalidArgument(
          "Cannot merge devices with incompatible tasks: '",
          ParsedNameToString(*target), "' and '", ParsedNameToString(other),
          "'");
    } else {
      target->has_task = other.has_task;
      target->task = other.task;
    }
  }

  if (other.has_type) {
    if (target->has_type && target->type != other.type) {
      if (!allow_soft_placement) {
        return errors::InvalidArgument(
            "Cannot merge devices with incompatible types: '",
            ParsedNameToString(*target), "' and '", ParsedNameToString(other),
            "'");
      } else if (override_conflicts) {
        target->type = other.type;
      } else {
        target->has_id = false;
        target->has_type = false;
        return absl::OkStatus();
      }
    } else {
      target->has_type = other.has_type;
      target->type = other.type;
    }
  }

  if (other.has_id) {
    if (target->has_id && target->id != other.id) {
      if (!allow_soft_placement) {
        return errors::InvalidArgument(
            "Cannot merge devices with incompatible ids: '",
            ParsedNameToString(*target), "' and '", ParsedNameToString(other),
            "'");
      } else if (override_conflicts) {
        target->id = other.id;
      } else {
        target->has_id = false;
        return absl::OkStatus();
      }
    } else {
      target->has_id = other.has_id;
      target->id = other.id;
    }
  }

  return absl::OkStatus();
}

}  // namespace

/* static */
absl::Status DeviceNameUtils::MergeDevNames(ParsedName* target,
                                            const ParsedName& other,
                                            bool allow_soft_placement) {
  return MergeDevNamesImpl(target, other, allow_soft_placement,
                           /*override_conflicts=*/false);
}

/* static */
absl::Status DeviceNameUtils::MergeOverrideDevNames(ParsedName* target,
                                                    const ParsedName& other) {
  return MergeDevNamesImpl(target, other, /*allow_soft_placement=*/true,
                           /*override_conflicts=*/true);
}

/* static */
void DeviceNameUtils::MergeUnsetDevNames(ParsedName* target,
                                         const ParsedName& other) {
  if (other.has_job && !target->has_job) {
    target->has_job = other.has_job;
    target->job = other.job;
  }

  if (other.has_replica && !target->has_replica) {
    target->has_replica = other.has_replica;
    target->replica = other.replica;
  }

  if (other.has_task && !target->has_task) {
    target->has_task = other.has_task;
    target->task = other.task;
  }

  if (other.has_type && !target->has_type) {
    target->has_type = other.has_type;
    target->type = other.type;
  }

  if (other.has_id && !target->has_id) {
    target->has_id = other.has_id;
    target->id = other.id;
  }
}

/* static */
bool DeviceNameUtils::IsSameAddressSpace(const ParsedName& a,
                                         const ParsedName& b) {
  return (a.has_job && b.has_job && (a.job == b.job)) &&
         (a.has_replica && b.has_replica && (a.replica == b.replica)) &&
         (a.has_task && b.has_task && (a.task == b.task));
}

/* static */
bool DeviceNameUtils::IsSameAddressSpace(absl::string_view src,
                                         absl::string_view dst) {
  ParsedName x;
  ParsedName y;
  return ParseFullName(src, &x) && ParseFullName(dst, &y) &&
         IsSameAddressSpace(x, y);
}

/* static */
bool DeviceNameUtils::IsDifferentAddressSpace(const ParsedName& a,
                                              const ParsedName& b) {
  return (a.has_job && b.has_job && (a.job != b.job)) ||
         (a.has_replica && b.has_replica && (a.replica != b.replica)) ||
         (a.has_task && b.has_task && (a.task != b.task));
}

/* static */
const DeviceNameUtils::ParsedName DeviceNameUtils::AddressSpace(
    const ParsedName& name) {
  ParsedName address_space;
  address_space.has_job = name.has_job;
  address_space.has_replica = name.has_replica;
  address_space.has_task = name.has_task;
  address_space.job = name.job;
  address_space.replica = name.replica;
  address_space.task = name.task;
  return address_space;
}

/* static */
string DeviceNameUtils::LocalName(absl::string_view type, int id) {
  return strings::StrCat("/device:", type, ":", id);
}

namespace {
// Returns the legacy local device name given its "type" and "id" (which is
// '/device:type:id').
string LegacyLocalName(absl::string_view type, int id) {
  return strings::StrCat(type, ":", id);
}
}  // anonymous namespace

/* static */
string DeviceNameUtils::LocalName(absl::string_view fullname) {
  ParsedName x;
  CHECK(ParseFullName(fullname, &x)) << fullname;
  return LocalName(x.type, x.id);
}

/* static */
bool DeviceNameUtils::ParseLocalName(absl::string_view name, ParsedName* p) {
  if (!ConsumeDeviceType(&name, &p->type)) {
    return false;
  }
  p->has_type = true;
  if (!absl::ConsumePrefix(&name, ":")) {
    return false;
  }
  if (!ConsumeNumber(&name, &p->id)) {
    return false;
  }
  p->has_id = true;
  return name.empty();
}

/* static */
bool DeviceNameUtils::SplitDeviceName(absl::string_view name, string* task,
                                      string* device) {
  ParsedName pn;
  if (ParseFullName(name, &pn) && pn.has_type && pn.has_id) {
    task->clear();
    task->reserve(
        (pn.has_job ? (5 + pn.job.size()) : 0) +
        (pn.has_replica ? (9 + 4 /*estimated UB for # replica digits*/) : 0) +
        (pn.has_task ? (6 + 4 /*estimated UB for # task digits*/) : 0));
    if (pn.has_job) {
      strings::StrAppend(task, "/job:", pn.job);
    }
    if (pn.has_replica) {
      strings::StrAppend(task, "/replica:", pn.replica);
    }
    if (pn.has_task) {
      strings::StrAppend(task, "/task:", pn.task);
    }
    device->clear();
    strings::StrAppend(device, pn.type, ":", pn.id);
    return true;
  }
  return false;
}

/* static */
bool DeviceNameUtils::GetTaskName(const ParsedName& pn, string* task) {
  if (pn.has_job && pn.has_replica && pn.has_task) {
    task->clear();
    task->reserve((5 + pn.job.size()) +
                  (9 + 4 /*estimated UB for # replica digits*/) +
                  (6 + 4 /*estimated UB for # task digits*/));
    strings::StrAppend(task, "/job:", pn.job);
    strings::StrAppend(task, "/replica:", pn.replica);
    strings::StrAppend(task, "/task:", pn.task);
    return true;
  }
  return false;
}

std::vector<string> DeviceNameUtils::GetNamesForDeviceMappings(
    const ParsedName& pn) {
  if (pn.has_job && pn.has_replica && pn.has_task && pn.has_type && pn.has_id) {
    return {
        DeviceNameUtils::FullName(pn.job, pn.replica, pn.task, pn.type, pn.id),
        LegacyName(pn.job, pn.replica, pn.task, pn.type, pn.id)};
  } else {
    return {};
  }
}

std::vector<string> DeviceNameUtils::GetLocalNamesForDeviceMappings(
    const ParsedName& pn) {
  if (pn.has_type && pn.has_id) {
    return {DeviceNameUtils::LocalName(pn.type, pn.id),
            LegacyLocalName(pn.type, pn.id)};
  } else {
    return {};
  }
}

/*static*/ absl::Status DeviceNameUtils::DeviceNameToCpuDeviceName(
    const string& device_name, string* host_device_name) {
  DeviceNameUtils::ParsedName device;
  if (!DeviceNameUtils::ParseFullName(device_name, &device)) {
    return errors::Internal("Could not parse device name ", device_name);
  }
  device.type = "CPU";
  device.has_type = true;
  device.id = 0;
  device.has_id = true;
  *host_device_name = DeviceNameUtils::ParsedNameToString(device);
  return absl::OkStatus();
}

std::ostream& operator<<(std::ostream& os,
                         const DeviceNameUtils::ParsedName& x) {
  os << DeviceNameUtils::ParsedNameToString(x);
  return os;
}

}  // namespace tsl
