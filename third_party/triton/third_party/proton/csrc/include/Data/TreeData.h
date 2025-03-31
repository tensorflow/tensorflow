#ifndef PROTON_DATA_TREE_DATA_H_
#define PROTON_DATA_TREE_DATA_H_

#include "Context/Context.h"
#include "Data.h"
#include <stdexcept>
#include <unordered_map>

namespace proton {

class TreeData : public Data {
public:
  TreeData(const std::string &path, ContextSource *contextSource);
  virtual ~TreeData();

  TreeData(const std::string &path) : TreeData(path, nullptr) {}

  size_t addOp(size_t scopeId, const std::string &name) override;

  void addMetric(size_t scopeId, std::shared_ptr<Metric> metric) override;

  void
  addMetrics(size_t scopeId,
             const std::map<std::string, MetricValueType> &metrics) override;

  void clear() override;

protected:
  // ScopeInterface
  void enterScope(const Scope &scope) override;

  void exitScope(const Scope &scope) override;

private:
  void init();
  void dumpHatchet(std::ostream &os) const;
  void doDump(std::ostream &os, OutputFormat outputFormat) const override;

  // `tree` and `scopeIdToContextId` can be accessed by both the user thread and
  // the background threads concurrently, so methods that access them should be
  // protected by a (shared) mutex.
  class Tree;
  std::unique_ptr<Tree> tree;
  // ScopeId -> ContextId
  std::unordered_map<size_t, size_t> scopeIdToContextId;
};

} // namespace proton

#endif // PROTON_DATA_TREE_DATA_H_
