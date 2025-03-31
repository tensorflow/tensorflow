#ifndef PROTON_DATA_DATA_H_
#define PROTON_DATA_DATA_H_

#include "Context/Context.h"
#include "Metric.h"
#include <map>
#include <memory>
#include <shared_mutex>
#include <string>

namespace proton {

enum class OutputFormat { Hatchet, Count };

class Data : public ScopeInterface {
public:
  Data(const std::string &path, ContextSource *contextSource = nullptr)
      : path(path), contextSource(contextSource) {}
  virtual ~Data() = default;

  /// Add an op to the data.
  /// If scopeId is already present, add an op under/inside it.
  /// Otherwise obtain the current context and append opName to it if opName is
  /// not empty.
  virtual size_t addOp(size_t scopeId, const std::string &opName = {}) = 0;

  /// Add a single metric to the data.
  virtual void addMetric(size_t scopeId, std::shared_ptr<Metric> metric) = 0;

  /// Add multiple metrics to the data.
  virtual void
  addMetrics(size_t scopeId,
             const std::map<std::string, MetricValueType> &metrics) = 0;

  /// Clear all caching data.
  virtual void clear() = 0;

  /// Dump the data to the given output format.
  void dump(OutputFormat outputFormat);

protected:
  /// The actual implementation of the dump operation.
  virtual void doDump(std::ostream &os, OutputFormat outputFormat) const = 0;

  mutable std::shared_mutex mutex;
  const std::string path{};
  ContextSource *contextSource{};
};

OutputFormat parseOutputFormat(const std::string &outputFormat);

const std::string outputFormatToString(OutputFormat outputFormat);

} // namespace proton

#endif // PROTON_DATA_DATA_H_
