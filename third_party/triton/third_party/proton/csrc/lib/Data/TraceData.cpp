#include "Data/TraceData.h"
#include "Utility/Errors.h"

#include <stdexcept>

namespace proton {

void TraceData::enterScope(const Scope &scope) { throw NotImplemented(); }

void TraceData::exitScope(const Scope &scope) { throw NotImplemented(); }

size_t TraceData::addOp(size_t scopeId, const std::string &name) {
  throw NotImplemented();
}

void TraceData::addMetric(size_t scopeId, std::shared_ptr<Metric> metric) {
  throw NotImplemented();
}

void TraceData::addMetrics(
    size_t scopeId, const std::map<std::string, MetricValueType> &metrics) {
  throw NotImplemented();
}

void TraceData::clear() { throw NotImplemented(); }

void TraceData::doDump(std::ostream &os, OutputFormat outputFormat) const {
  throw NotImplemented();
}

} // namespace proton
