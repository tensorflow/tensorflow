#ifndef THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_STEP_STATS_COLLECTOR_H_
#define THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_STEP_STATS_COLLECTOR_H_

#include <string>

#include "tensorflow/core/platform/port.h"  // Must be first.
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {

class NodeExecStats;
class StepStats;

class StepStatsCollector {
 public:
  explicit StepStatsCollector(StepStats* ss);

  void Save(const string& device, NodeExecStats* nt);

  void Swap(StepStats* ss);

 private:
  friend class StepStatsMgr;
  mutex mu_;
  StepStats* step_stats_ GUARDED_BY(mu_);
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_STEP_STATS_COLLECTOR_H_
