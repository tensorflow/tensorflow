/* Copyright 2016 Google Inc. All Rights Reserved.

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

// Master implements the service MasterSerivce.
//
// A Master maintains the state of live graph computation
// sessions, each session orchestrates both local and remote devices
// to carry out the graph computation.
//
// A Master knows ahead of time local devices available as
// client devices.
//
// A Master discovers remote devices on-demand and keeps track of
// statistics of those remote devices.
//
// Each session analyses the graph, places nodes across available
// devices, and ultimately drives the graph computation by initiating
// RunGraph on the workers.

#include "tensorflow/core/distributed_runtime/master.h"

#include <unordered_set>
#include <vector>

#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/distributed_runtime/remote_device.h"
#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/core/distributed_runtime/worker_interface.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/master.pb.h"
#include "tensorflow/core/protobuf/worker.pb.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

Master::Master(MasterEnv* env, double session_gc_seconds)
    : env_(env),
      last_1000_steps_(1000),
      step_count_(0),
      session_gc_seconds_(session_gc_seconds) {
  // Right now, a master service must be co-located with a device.
  // Otherwise, fetches do not work.
  CHECK(!env->local_devices.empty());

  if (session_gc_seconds_ > 0.0) {
    gc_thread_ = env_->env->StartThread(ThreadOptions(), "TF_master_GC",
                                        [this]() { GC(); });
  } else {
    gc_thread_ = nullptr;
  }
}

Master::~Master() {
  if (gc_thread_) {
    mutex_lock l(mu_);
    shutdown_ = true;
    shutdown_cv_.notify_all();
    delete gc_thread_;
  }
}

void Master::GC() {
  Env* env = Env::Default();
  while (true) {
    mutex_lock l(mu_);
    const int kTimeoutMilliseconds = 10 * 1000;  // 10 seconds.
    WaitForMilliseconds(&l, &shutdown_cv_, kTimeoutMilliseconds);
    if (shutdown_) {
      break;
    }
    std::vector<string> handles;
    const int64 num_micros = static_cast<int64>(session_gc_seconds_ * 1000000);
    for (const auto& entry : sessions_) {
      auto lat = entry.second->last_access_time_usec();
      if (env->NowMicros() - lat > num_micros) {
        handles.push_back(entry.first);
        auto* sess = entry.second;
        SchedClosure([this, sess]() {
          LOG(WARNING) << "GC session " << sess->handle() << " after "
                       << session_gc_seconds_ << " seconds.  "
                       << "Note that if you are starting multiple replicas "
                       << "on a staggered delay, session_gc_seconds may need "
                       << "to be raised.";
          sess->Close();
        });
      }
    }
    for (const auto& handle : handles) sessions_.erase(handle);
  }
}

class DeviceFinder {
 public:
  explicit DeviceFinder(
      const protobuf::RepeatedPtrField<string>& device_filters, MasterEnv* env)
      : env_(env) {
    auto process_filter = [this](const string& filter) {
      DeviceNameUtils::ParsedName parsed;
      if (DeviceNameUtils::ParseFullName(filter, &parsed)) {
        filters_.push_back(parsed);
      } else {
        LOG(FATAL) << "Skipping invalid filter: " << filter;
      }
    };
    for (const string& filter : device_filters) {
      process_filter(filter);
    }
  }

  ~DeviceFinder() {
    for (Device* dev : found_) delete dev;
  }

  void Start() {
    // Enumerates all known workers' target. A target name is a
    // prefix of a device name. E.g., /job:mnist/replica:0/task:10.
    std::vector<string> workers;
    env_->worker_cache->ListWorkers(&workers);
    std::vector<string> targets;
    if (filters_.empty()) {
      swap(workers, targets);
    } else {
      for (const string& name : workers) {
        if (MatchFilters(name)) {
          targets.push_back(name);
        }
      }
    }
    {
      mutex_lock l(mu_);
      num_pending_ = targets.size();
      if (num_pending_ == 0) {
        pending_zero_.notify_all();
      }
    }
    // Talk to all workers to get the list of available devices.
    using std::placeholders::_1;
    using std::placeholders::_2;
    for (size_t i = 0; i < targets.size(); ++i) {
      NewRemoteDevices(env_->env, env_->worker_cache, targets[i],
                       std::bind(&ME::WhenFound, this, _1, _2));
    }
  }

  void Wait() {
    mutex_lock l(mu_);
    while (num_pending_ != 0) {
      pending_zero_.wait(l);
    }
  }

  // The caller takes the ownership of returned remote devices.
  void GetRemoteDevices(const std::vector<Device*>& local,
                        std::vector<Device*>* remote) {
    std::unordered_set<string> names(local.size());
    for (Device* dev : local) names.insert(dev->name());
    mutex_lock l(mu_);
    for (Device* dev : found_) {
      const string& name = dev->name();
      if (names.insert(name).second && MatchFilters(name)) {
        remote->push_back(dev);
      } else {
        delete dev;
      }
    }
    found_.clear();
  }

 private:
  typedef DeviceFinder ME;
  const MasterEnv* env_;
  std::vector<DeviceNameUtils::ParsedName> filters_;

  mutex mu_;
  int num_pending_ GUARDED_BY(mu_);
  condition_variable pending_zero_;
  std::vector<Device*> found_ GUARDED_BY(mu_);

  void WhenFound(const Status& s, std::vector<Device*>* devices) {
    mutex_lock l(mu_);
    if (!s.ok()) {
      LOG(ERROR) << "Master init: " << s;
    } else {
      found_.insert(found_.end(), devices->begin(), devices->end());
      devices->clear();
    }
    --num_pending_;
    if (num_pending_ == 0) {
      pending_zero_.notify_all();
    }
  }

  // Returns true iff the set of devices allowed by 'x' intersects
  // with the set of devices allowed by 'y'.
  bool Intersects(const DeviceNameUtils::ParsedName& x,
                  const DeviceNameUtils::ParsedName& y) {
    return (!x.has_job || !y.has_job || x.job == y.job) &&
           (!x.has_replica || !y.has_replica || x.replica == y.replica) &&
           (!x.has_task || !y.has_task || x.task == y.task) &&
           (!x.has_type || !y.has_type || x.type == y.type) &&
           (!x.has_id || !y.has_id || x.id == y.id);
  }

  // Returns true iff 'name' matches one of the filters_.
  bool MatchFilters(const string& name) {
    if (filters_.empty()) return true;
    DeviceNameUtils::ParsedName x;
    if (DeviceNameUtils::ParseFullName(name, &x)) {
      for (const auto& filter : filters_) {
        if (Intersects(x, filter)) return true;
      }
    }
    return false;
  }

  TF_DISALLOW_COPY_AND_ASSIGN(DeviceFinder);
};

void Master::CreateSession(const CreateSessionRequest* req,
                           CreateSessionResponse* resp, MyClosure done) {
  SchedClosure([this, req, resp, done]() {
    Status status = ValidateExternalGraphDefSyntax(req->graph_def());
    if (status.ok()) {
      // Ping all the workers and build the list of devices that the
      // session will use.
      DeviceFinder finder(req->config().device_filters(), env_);
      finder.Start();
      finder.Wait();
      std::vector<Device*> remote_devices;
      finder.GetRemoteDevices(env_->local_devices, &remote_devices);
      SessionOptions options;
      options.config = req->config();
      MasterSessionInterface* session =
          env_->master_session_factory(options, env_, &remote_devices);
      GraphDef* gdef =
          const_cast<CreateSessionRequest*>(req)->mutable_graph_def();
      Status create_status = session->Create(gdef);
      if (!create_status.ok()) {
        done(create_status);
        return;
      }
      resp->set_session_handle(session->handle());
      // Insert into the session map.
      {
        mutex_lock l(mu_);
        CHECK(sessions_.insert({session->handle(), session}).second);
      }
    }
    done(status);
  });
}

void Master::ExtendSession(const ExtendSessionRequest* req,
                           ExtendSessionResponse* resp, MyClosure done) {
  mu_.lock();
  MasterSessionInterface* session = nullptr;
  session = gtl::FindPtrOrNull(sessions_, req->session_handle());
  if (session == nullptr) {
    mu_.unlock();
    done(errors::Aborted("Session ", req->session_handle(), " is not found."));
    return;
  }

  SchedClosure([session, req, resp, done]() {
    Status status = ValidateExternalGraphDefSyntax(req->graph_def());
    if (status.ok()) {
      status = session->Extend(req, resp);
    }
    done(status);
  });
  mu_.unlock();
}

void Master::RunStep(CallOptions* opts, const RunStepRequest* req,
                     RunStepResponse* resp, MyClosure done) {
  mu_.lock();
  uint64 start_time = env_->env->NowMicros();
  MasterSessionInterface* session =
      gtl::FindPtrOrNull(sessions_, req->session_handle());
  if (session == nullptr) {
    mu_.unlock();
    done(errors::Aborted("Session ", req->session_handle(), " is not found."));
    return;
  }

  SchedClosure([this, start_time, session, opts, req, resp, done]() {
    Status status = session->Run(opts, req, resp);
    uint64 done_time = env_->env->NowMicros();
    done(status);
    mutex_lock l(mu_);
    last_1000_steps_.AddValue((done_time - start_time) / 1e9);
    ++step_count_;
  });
  mu_.unlock();
}

void Master::CloseSession(const CloseSessionRequest* req,
                          CloseSessionResponse* resp, MyClosure done) {
  MasterSessionInterface* session = nullptr;
  {
    mu_.lock();
    auto iter = sessions_.find(req->session_handle());
    if (iter == sessions_.end()) {
      mu_.unlock();
      done(errors::Aborted(
          "Session ", req->session_handle(),
          " is not found. Possibly, this master has restarted."));
      return;
    }
    session = iter->second;
    sessions_.erase(iter);
    mu_.unlock();
  }

  // Session Close() blocks on thread shutdown. Therefore, we need to
  // delete it in non-critical thread.
  SchedClosure([session, done]() {
    Status s = session->Close();
    done(s);
  });
}

void Master::ListDevices(const ListDevicesRequest* req,
                         ListDevicesResponse* resp, MyClosure done) {
  SchedClosure([this, req, resp, done]() {
    DeviceFinder finder({}, env_);
    finder.Start();
    finder.Wait();
    std::vector<Device*> remote_devices;
    finder.GetRemoteDevices(env_->local_devices, &remote_devices);
    for (Device* dev : env_->local_devices) {
      *(resp->add_local_device()) = dev->attributes();
    }
    for (Device* dev : remote_devices) {
      *(resp->add_remote_device()) = dev->attributes();
      delete dev;
    }
    done(Status::OK());
  });
}

void Master::CleanupWorkers(const ResetRequest& reset) {
  std::vector<string> worker_names;
  env_->worker_cache->ListWorkers(&worker_names);
  if (!worker_names.empty()) {
    const int num_workers = worker_names.size();
    std::vector<Notification> n(num_workers);
    CleanupAllRequest req;
    (*req.mutable_container()) = reset.container();
    std::vector<CleanupAllResponse> resp(num_workers);
    int c = 0;
    for (int i = 0; i < num_workers; ++i) {
      auto worker = env_->worker_cache->CreateWorker(worker_names[i]);
      if (worker) {
        worker->CleanupAllAsync(&req, &resp[i], [&n, worker, c](Status s) {
          TF_CHECK_OK(s);
          delete worker;
          n[c].Notify();
        });
      } else {
        n[c].Notify();
      }
      ++c;
    }
    for (int i = 0; i < n.size(); ++i) {
      n[i].WaitForNotification();
    }
  }
}

void Master::Reset(const ResetRequest* req, ResetResponse* resp,
                   MyClosure done) {
  // Vector to hold the session pointers present in the sessions_
  // (string->Session*) map.
  std::vector<MasterSessionInterface*> sessions;
  {
    mutex_lock l(mu_);
    for (const auto& entry : sessions_) {
      sessions.push_back(entry.second);
    }
    sessions_.clear();
  }

  CleanupWorkers(*req);

  SchedClosure([sessions, done]() {
    Status s;
    for (MasterSessionInterface* session : sessions) {
      s.Update(session->Close());
    }
    done(s);
  });
}

}  // end namespace tensorflow
