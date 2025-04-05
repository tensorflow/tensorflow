#ifndef PROTON_DATA_METRIC_H_
#define PROTON_DATA_METRIC_H_

#include "Utility/String.h"
#include "Utility/Traits.h"
#include <variant>
#include <vector>

namespace proton {

enum class MetricKind { Flexible, Kernel, PCSampling, Count };

using MetricValueType = std::variant<uint64_t, int64_t, double, std::string>;

/// A metric is a class that can be associated with a context.
/// `Metric` is the base class for all metrics.
/// Each `Metric` has a name and a set of values.
/// Each value could be of type `uint64_t`, `int64_t`, or `double`,
/// Each value can be inclusive (inc), exclusive (exc), or a property (pty).
/// Inclusive values are aggregated by addition and can be propagated to the
/// parent.
/// Exclusive values can be aggregated at a context but cannot be
/// propagated to the parent.
/// Property values are not aggregated and cannot be propagated to the parent.
class Metric {
public:
  Metric(MetricKind kind, size_t size) : kind(kind), values(size) {}

  virtual ~Metric() = default;

  virtual const std::string getName() const = 0;

  virtual const std::string getValueName(int valueId) const = 0;

  virtual bool isProperty(int valueId) const = 0;

  virtual bool isExclusive(int valueId) const = 0;

  std::vector<MetricValueType> getValues() const { return values; }

  MetricValueType getValue(int valueId) { return values[valueId]; }

  /// Update a specific value id with the new value.
  void updateValue(int valueId, MetricValueType value) {
    // Handle string and other values separately
    if (std::holds_alternative<std::string>(value)) {
      values[valueId] = std::get<std::string>(value);
    } else {
      std::visit(
          [&](auto &&currentValue, auto &&otherValue) {
            using CurrentType = std::decay_t<decltype(currentValue)>;
            using ValueType = std::decay_t<decltype(otherValue)>;
            if constexpr (std::is_same_v<ValueType, CurrentType>) {
              if (isProperty(valueId)) {
                currentValue = otherValue;
              } else {
                currentValue += otherValue;
              }
            }
          },
          values[valueId], value);
    }
  }

  /// Update all values of the metric with the same value.
  void updateValue(MetricValueType value) {
    for (int i = 0; i < values.size(); ++i) {
      updateValue(i, value);
    }
  }

  /// Update all values with another metric.
  void updateMetric(Metric &other) {
    for (int i = 0; i < values.size(); ++i) {
      updateValue(i, other.values[i]);
    }
  }

  MetricKind getKind() const { return kind; }

private:
  const MetricKind kind;
  const std::string name;

protected:
  std::vector<MetricValueType> values;
};

/// A flexible metric is provided by users but not the backend profiling API.
/// Each flexible metric has a single value.
class FlexibleMetric : public Metric {
public:
  FlexibleMetric(const std::string &valueName,
                 std::variant<MetricValueType> value)
      : Metric(MetricKind::Flexible, 1), valueName(valueName) {
    this->exclusive = endWith(valueName, "(exc)");
    this->property = endWith(valueName, "(pty)");
    this->valueName = trim(replace(this->valueName, "(exc)", ""));
    this->valueName = trim(replace(this->valueName, "(pty)", ""));
    std::visit([&](auto &&v) { this->values[0] = v; }, value);
  }

  FlexibleMetric(const std::string &valueName,
                 std::variant<MetricValueType> value, bool property,
                 bool exclusive)
      : Metric(MetricKind::Flexible, 1), valueName(valueName),
        property(property), exclusive(exclusive) {
    std::visit([&](auto &&v) { this->values[0] = v; }, value);
  }

  const std::string getName() const override { return "FlexibleMetric"; }

  const std::string getValueName(int valueId) const override {
    return valueName;
  }

  bool isProperty(int valueId) const override { return property; }

  bool isExclusive(int valueId) const override { return exclusive; }

private:
  bool property{};
  bool exclusive{};
  std::string valueName;
};

class KernelMetric : public Metric {
public:
  enum kernelMetricKind : int {
    StartTime,
    EndTime,
    Invocations,
    Duration,
    DeviceId,
    DeviceType,
    Count,
  };

  KernelMetric() : Metric(MetricKind::Kernel, kernelMetricKind::Count) {}

  KernelMetric(uint64_t startTime, uint64_t endTime, uint64_t invocations,
               uint64_t deviceId, uint64_t deviceType)
      : KernelMetric() {
    this->values[StartTime] = startTime;
    this->values[EndTime] = endTime;
    this->values[Invocations] = invocations;
    this->values[Duration] = endTime - startTime;
    this->values[DeviceId] = deviceId;
    this->values[DeviceType] = deviceType;
  }

  virtual const std::string getName() const { return "KernelMetric"; }

  virtual const std::string getValueName(int valueId) const {
    return VALUE_NAMES[valueId];
  }

  virtual bool isProperty(int valueId) const { return PROPERTY[valueId]; }

  virtual bool isExclusive(int valueId) const { return EXCLUSIVE[valueId]; }

private:
  const static inline bool PROPERTY[kernelMetricKind::Count] = {
      true, true, false, false, true, true};
  const static inline bool EXCLUSIVE[kernelMetricKind::Count] = {
      false, false, false, false, true, true};
  const static inline std::string VALUE_NAMES[kernelMetricKind::Count] = {
      "start_time (ns)", "end_time (ns)", "count",
      "time (ns)",       "device_id",     "device_type",
  };
};

class PCSamplingMetric : public Metric {
public:
  enum PCSamplingMetricKind : int {
    NumSamples,
    NumStalledSamples,
    StalledBranchResolving,
    StalledNoInstruction,
    StalledShortScoreboard,
    StalledWait,
    StalledLongScoreboard,
    StalledTexThrottle,
    StalledBarrier,
    StalledMembar,
    StalledIMCMiss,
    StalledMIOThrottle,
    StalledMathPipeThrottle,
    StalledDrain,
    StalledLGThrottle,
    StalledNotSelected,
    StalledMisc,
    StalledDispatchStall,
    StalledSleeping,
    StalledSelected,
    Count,
  };

  PCSamplingMetric()
      : Metric(MetricKind::PCSampling, PCSamplingMetricKind::Count) {}

  PCSamplingMetric(PCSamplingMetricKind kind, uint64_t samples,
                   uint64_t stalledSamples)
      : PCSamplingMetric() {
    this->values[kind] = stalledSamples;
    this->values[PCSamplingMetricKind::NumSamples] = samples;
    this->values[PCSamplingMetricKind::NumStalledSamples] = stalledSamples;
  }

  virtual const std::string getName() const { return "PCSamplingMetric"; }

  virtual const std::string getValueName(int valueId) const {
    return VALUE_NAMES[valueId];
  }

  virtual bool isProperty(int valueId) const { return false; }

  virtual bool isExclusive(int valueId) const { return false; }

private:
  const static inline std::string VALUE_NAMES[PCSamplingMetricKind::Count] = {
      "num_samples",
      "num_stalled_samples",
      "stalled_branch_resolving",
      "stalled_no_instruction",
      "stalled_short_scoreboard",
      "stalled_wait",
      "stalled_long_scoreboard",
      "stalled_tex_throttle",
      "stalled_barrier",
      "stalled_membar",
      "stalled_imc_miss",
      "stalled_mio_throttle",
      "stalled_math_pipe_throttle",
      "stalled_drain",
      "stalled_lg_throttle",
      "stalled_not_Selected",
      "stalled_misc",
      "stalled_dispatch_stall",
      "stalled_sleeping",
      "stalled_selected",
  };
};

} // namespace proton

#endif // PROTON_DATA_METRIC_H_
