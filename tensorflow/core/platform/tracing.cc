#include "tensorflow/core/platform/tracing.h"

#include <atomic>
#include <map>
#include <string>
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/lib/strings/str_util.h"
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

namespace port {

int32 Tracing::category_id_[kEventCategoryMax];
uint64 Tracing::event_mask_ = 0;
std::map<string, int32>* Tracing::name_map_ = new std::map<string, int32>;

// This needs to be kept in sync with the EventCategory enumeration.
const char* Tracing::EventCategoryString(EventCategory category) {
  switch (category) {
    case EventCategory::kScheduleClosure:
      return "ScheduleClosure";
    case EventCategory::kRunClosure:
      return "RunClosure";
    case EventCategory::kCompute:
      return "Compute";
    case EventCategory::kEventCategoryMax:
      return "EventCategoryMax";
  }
  return "Unknown";
}

// This function allows the user to specify arbitrary subsets of the
// supported Threadscape events and activities.
bool Tracing::ParseEventMask(const char* flagname, const string& value) {
  VLOG(1) << flagname << " set to " << value;
  int64 new_mask = 0;
  std::vector<string> events =
      str_util::Split(value, ',', str_util::SkipEmpty());
  for (string name : events) {
    bool clear = false;
    int64 mask = 0;
    if (name[0] == '!') {
      // invert the sense of the flag
      clear = true;
      name = name.substr(1);
    }
    if (name == "ALL") {
      mask = ~0;
    } else {
      auto it = name_map_->find(name);
      int32 id;
      if (it == name_map_->end()) {
        id = -1;
      } else {
        id = it->second;
      }
      if (id < 0) {
        LOG(ERROR) << "Can't parse event mask name " << name;
        return false;
      }
      mask = 1 << id;
    }
    if (clear) {
      new_mask &= ~mask;
    } else {
      new_mask |= mask;
    }
  }
  // parsing was successful; set the permanent event mask
  event_mask_ = new_mask;
  return true;
}

static std::atomic<Tracing::Engine*> tracing_engine;

void Tracing::RegisterEngine(Engine* e) {
  tracing_engine.store(e, std::memory_order_release);
}

static Tracing::Engine* engine() {
  return tracing_engine.load(std::memory_order_acquire);
}

Tracing::Engine::~Engine() {}
Tracing::Engine::Annotation::~Annotation() {}
Tracing::Engine::Tracer::~Tracer() {}

Tracing::ScopedAnnotation::ScopedAnnotation(StringPiece name) {
  auto e = engine();
  if (e) {
    annotation_.reset(e->PushAnnotation(name));
  }
}

Tracing::TraceMe::TraceMe(StringPiece name) {
  auto e = engine();
  if (e) {
    tracer_.reset(e->StartTracing(name));
  }
}

}  // namespace port
}  // namespace tensorflow
