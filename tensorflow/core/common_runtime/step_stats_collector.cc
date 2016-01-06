#include "tensorflow/core/common_runtime/step_stats_collector.h"

#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

StepStatsCollector::StepStatsCollector(StepStats* ss) : step_stats_(ss) {}

void StepStatsCollector::Save(const string& device, NodeExecStats* nt) {
  VLOG(1) << "Save dev " << device << " nt " << nt;
  {
    mutex_lock l(mu_);
    DeviceStepStats* dss = nullptr;
    // Slow linear scan, but it should only be called
    // by a Worker in a context with < ~10 devices.
    // TODO(tucker): consider adding a std::unordered_map.
    for (auto& ds : *step_stats_->mutable_dev_stats()) {
      if (ds.device() == device) {
        dss = &ds;
        break;
      }
    }
    if (dss == nullptr) {
      dss = step_stats_->add_dev_stats();
      dss->set_device(device);
    }
    nt->Swap(dss->add_node_stats());
  }
  delete nt;
}

void StepStatsCollector::Swap(StepStats* ss) {
  mutex_lock l(mu_);
  CHECK(step_stats_);
  ss->Swap(step_stats_);
}

}  // namespace tensorflow
