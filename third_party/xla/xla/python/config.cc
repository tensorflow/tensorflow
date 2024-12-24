/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/python/config.h"

#include <Python.h>

#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"  // IWYU pragma: keep
#include "xla/python/python_ref_manager.h"
#include "tsl/platform/logging.h"

namespace jax {

namespace nb = nanobind;

// Singleton object used to represet "value not set" in thread-local configs.
nb::object UnsetObject() {
  return nb::steal(PyObject_CallObject(
      reinterpret_cast<PyObject*>(&PyBaseObject_Type), nullptr));
}

// Each configuration object has:
// * a global value, and
// * a thread-local value.
// When querying the state of a config, the thread-local value is used if it is
// set. Otherwise, the global value is used.

// This class represents all of the thread-local configuration state for a
// thread.
class ThreadLocalConfigState {
 public:
  ThreadLocalConfigState();
  ~ThreadLocalConfigState();

  static ThreadLocalConfigState& Instance() {
    thread_local auto state = std::make_unique<ThreadLocalConfigState>();
    return *state;
  }

  nb::object Get(int key) {
    DCHECK_GE(key, 0);
    return key >= entries_.size() ? nb::object() : entries_[key];
  }

  void Set(int key, nb::object value);

 private:
  friend class GlobalConfigState;

  // These values are accessed in one of two ways:
  // * The owning thread reads or writes them, while holding the GIL, or, under
  //   free-threading, while the owning thread is in ATTACHED gc state.
  // * Other threads may read or clear values while performing a garbarge
  //   collection.
  // No locking is needed because a GC thread cannot run concurrently with other
  // Python threads; even under free-threading Python uses a stop-the-world GC.
  std::vector<nb::object> entries_;
};

// This class represents all of the global configuration state.
// TODO(phawkins): to support free-threading, we will need to add locking to
// this class.
class GlobalConfigState {
 public:
  static GlobalConfigState& Instance() {
    static auto state = new GlobalConfigState();
    return *state;
  }

  nb::object Get(int key) const;
  void Set(int key, nb::object value);

  // Adds or removes a thread-local state from the set of thread-local states.
  void AddThreadLocalState(ThreadLocalConfigState* state) {
    absl::MutexLock lock(&mu_);
    thread_local_states_.insert(state);
  }
  void RemoveThreadLocalState(ThreadLocalConfigState* state) {
    absl::MutexLock lock(&mu_);
    thread_local_states_.erase(state);
  }

  // Python GC helpers. These are called from the tp_traverse and tp_clear
  // methods of the Config class.
  int tp_traverse(int key, PyObject* self, visitproc visit, void* arg);
  int tp_clear(int key, PyObject* self);

  // Returns the singleton object representing "value not set".
  const nb::object& unset() const { return unset_; }

  // Returns the set of keys that should be included in the jit key.
  absl::Span<int const> include_in_jit_key() const {
    return include_in_jit_key_;
  }

 private:
  friend class Config;

  // The set of thread-local states. This is used during garbarge collection to
  // visit thread-local values.
  absl::Mutex mu_;
  absl::flat_hash_set<ThreadLocalConfigState*> thread_local_states_
      ABSL_GUARDED_BY(mu_);
  std::vector<nb::object> entries_;
  std::vector<int> include_in_jit_key_;
  nb::object unset_ = UnsetObject();
};

ThreadLocalConfigState::ThreadLocalConfigState() {
  GlobalConfigState::Instance().AddThreadLocalState(this);
}

ThreadLocalConfigState::~ThreadLocalConfigState() {
  // It's important that we remove the thread-local state before we access
  // entries_. This ensures that accesses to entries_ are ordered with respect
  // any garbage collection.
  GlobalConfigState::Instance().RemoveThreadLocalState(this);
  // We do not hold the GIL, so we must use deferred destruction.
  xla::GlobalPyRefManager()->AddGarbage(absl::MakeSpan(entries_));
}

void ThreadLocalConfigState::Set(int key, nb::object value) {
  DCHECK_GE(key, 0);
  if (key >= entries_.size()) {
    entries_.resize(key + 1);
  }
  std::swap(entries_[key], value);
}

nb::object GlobalConfigState::Get(int key) const {
  DCHECK_GE(key, 0);
  DCHECK_LT(key, entries_.size());
  return entries_[key];
}

void GlobalConfigState::Set(int key, nb::object value) {
  DCHECK_GE(key, 0);
  DCHECK_LT(key, entries_.size());
  std::swap(entries_[key], value);
}

int GlobalConfigState::tp_traverse(int key, PyObject* self, visitproc visit,
                                   void* arg) {
  DCHECK_GE(key, 0);
  if (key < entries_.size()) {
    PyObject* value = entries_[key].ptr();
    Py_VISIT(value);
  }
  absl::MutexLock lock(&mu_);
  for (const auto* state : thread_local_states_) {
    if (key < state->entries_.size()) {
      PyObject* value = state->entries_[key].ptr();
      Py_VISIT(value);
    }
  }
  return 0;
}

int GlobalConfigState::tp_clear(int key, PyObject* self) {
  if (key < entries_.size()) {
    nb::object tmp;
    std::swap(entries_[key], tmp);
  }
  // We destroy the python objects outside of the lock out of an abundance of
  // caution.
  std::vector<nb::object> to_destroy;
  absl::MutexLock lock(&mu_);
  to_destroy.reserve(thread_local_states_.size());
  for (auto* state : thread_local_states_) {
    if (key < state->entries_.size()) {
      nb::object tmp;
      std::swap(state->entries_[key], tmp);
      to_destroy.push_back(std::move(tmp));
    }
  }
  return 0;
}

// A Config object represents a configurable object with both global and
// thread-local state. This class is wrapped using nanobind and exposed to
// Python.
class Config {
 public:
  Config(nb::object value, bool include_in_jit_key);

  // Returns the thread-local value if it is set, otherwise the global value.
  nb::object Get();

  // Returns the global value.
  nb::object GetGlobal();

  // Sets the global value.
  void SetGlobal(nb::object value);

  // Returns the thread-local value.
  nb::object GetLocal();

  // Sets the thread-local value. May be `unset`.
  void SetLocal(nb::object value);

  // Swaps the thread-local value with `value`. Returns the previous value.
  // Either may be `unset`.
  nb::object SwapLocal(nb::object value);

  // This class doesn't actually hold any data, but it's the only type
  // known to Python. We pretend that this object owns both the global and any
  // thread-local values corresponding to this key.
  static int tp_traverse(PyObject* self, visitproc visit, void* arg);
  static int tp_clear(PyObject* self);
  static PyType_Slot slots_[];

 private:
  int key_;
};

Config::Config(nb::object value, bool include_in_jit_key) {
  auto& instance = GlobalConfigState::Instance();
  key_ = instance.entries_.size();
  instance.entries_.push_back(std::move(value));
  if (include_in_jit_key) {
    instance.include_in_jit_key_.push_back(key_);
  }
}

nb::object Config::GetLocal() {
  nb::object result = ThreadLocalConfigState::Instance().Get(key_);
  if (!result.is_valid()) {
    return GlobalConfigState::Instance().unset();
  }
  return result;
}

nb::object Config::GetGlobal() {
  return GlobalConfigState::Instance().Get(key_);
}

nb::object Config::Get() {
  nb::object local = ThreadLocalConfigState::Instance().Get(key_);
  if (local.is_valid()) {
    return local;
  }
  return GetGlobal();
}

void Config::SetLocal(nb::object value) {
  const auto& instance = GlobalConfigState::Instance();
  if (value.ptr() == instance.unset().ptr()) {
    value = nb::object();
  }
  ThreadLocalConfigState::Instance().Set(key_, std::move(value));
}

nb::object Config::SwapLocal(nb::object value) {
  const auto& global_instance = GlobalConfigState::Instance();
  auto& instance = ThreadLocalConfigState::Instance();
  auto result = instance.Get(key_);
  if (value.ptr() == global_instance.unset().ptr()) {
    value = nb::object();
  }
  instance.Set(key_, std::move(value));
  if (!result.is_valid()) {
    return global_instance.unset();
  }
  return result;
}

void Config::SetGlobal(nb::object value) {
  GlobalConfigState::Instance().Set(key_, value);
}

/* static */ int Config::tp_traverse(PyObject* self, visitproc visit,
                                     void* arg) {
  Config* c = nb::inst_ptr<Config>(self);
  // For the purposes of GC, we pretend that this object owns both the global
  // and any thread-local values corresponding to this key.
  return GlobalConfigState::Instance().tp_traverse(c->key_, self, visit, arg);
}

/* static */ int Config::tp_clear(PyObject* self) {
  Config* c = nb::inst_ptr<Config>(self);
  return GlobalConfigState::Instance().tp_clear(c->key_, self);
}

PyType_Slot Config::slots_[] = {
    {Py_tp_traverse, reinterpret_cast<void*>(Config::tp_traverse)},
    {Py_tp_clear, reinterpret_cast<void*>(Config::tp_clear)},
    {0, nullptr},
};

void BuildConfigSubmodule(nanobind::module_& m) {
  nb::module_ config_module = m.def_submodule("config", "Config library");

  config_module.attr("unset") = GlobalConfigState::Instance().unset();

  nb::class_<Config> config(config_module, "Config",
                            nb::type_slots(Config::slots_), nb::is_generic());
  config.def(nb::init<nb::object, bool>(), nb::arg("value").none(),
             nb::arg("include_in_jit_key") = false);
  config.def_prop_ro("value", &Config::Get);
  config.def("get_local", &Config::GetLocal);
  config.def("get_global", &Config::GetGlobal);
  config.def("set_local", &Config::SetLocal, nb::arg("value").none());
  config.def("swap_local", &Config::SwapLocal, nb::arg("value").none());
  config.def("set_global", &Config::SetGlobal, nb::arg("value").none());
}

std::vector<nanobind::object> JitConfigs() {
  auto& instance = GlobalConfigState::Instance();
  auto& thread_local_instance = ThreadLocalConfigState::Instance();
  std::vector<nanobind::object> result;
  result.reserve(instance.include_in_jit_key().size());
  for (int i : instance.include_in_jit_key()) {
    nb::object local = thread_local_instance.Get(i);
    if (local.is_valid()) {
      result.push_back(std::move(local));
    } else {
      result.push_back(instance.Get(i));
    }
  }
  return result;
}

}  // namespace jax
