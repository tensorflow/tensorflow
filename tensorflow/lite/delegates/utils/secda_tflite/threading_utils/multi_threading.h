#ifndef SECDA_THREADING
#define SECDA_THREADING
#include <assert.h>   // NOLINT
#include <algorithm>  // NOLINT
#include <atomic>     // NOLINT
#include <thread>
#include <vector>

// Follows by GEMMLOWP multi-threading interface
inline int DoSomeNOPs() { return 16; }
const int kMaxBusyWaitNOPs = 4 * 1000 * 1000;

// Workload
struct Task {
  Task() {}
  virtual ~Task() {}
  virtual void Run() = 0;
};

template <typename T>
T WaitForVariableChangeV2(std::atomic<T>* var, T initial_value,
                        pthread_cond_t* cond, pthread_mutex_t* mutex) {
  T new_value = var->load(std::memory_order_acquire);
  if (new_value != initial_value) {
    return new_value;
  }

  int nops = 0;
  while (nops < kMaxBusyWaitNOPs) {
    nops += DoSomeNOPs();
    new_value = var->load(std::memory_order_acquire);
    if (new_value != initial_value) {
      return new_value;
    }
  }

  pthread_mutex_lock(mutex);
  new_value = var->load(std::memory_order_acquire);
  while (new_value == initial_value) {
    pthread_cond_wait(cond, mutex);
    new_value = var->load(std::memory_order_acquire);
  }
  pthread_mutex_unlock(mutex);
  return new_value;
}

class BlockingCounter {
 public:
  BlockingCounter() : count_(0) {}

  void Reset(std::size_t initial_count) {
    std::size_t old_count_value = count_.load(std::memory_order_relaxed);
    (void)old_count_value;
    count_.store(initial_count, std::memory_order_release);
  }

  bool DecrementCount() {
    std::size_t old_count_value =
        count_.fetch_sub(1, std::memory_order_acq_rel);
    std::size_t count_value = old_count_value - 1;
    return count_value == 0;
  }

  void Wait() {
    int nops = 0;
    while (count_.load(std::memory_order_acquire)) {
      nops += DoSomeNOPs();
      if (nops > kMaxBusyWaitNOPs) {
        nops = 0;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }
  }

 private:
  std::atomic<std::size_t> count_;
};

// A worker thread.
class Worker {
 public:
  enum class State {
    ThreadStartup,  // The initial state before the thread main loop runs.
    Ready,          // Is not working, has not yet received new work to do.
    HasWork,        // Has work to do.
    ExitAsSoonAsPossible  // Should exit at earliest convenience.
  };

  explicit Worker(BlockingCounter* counter_to_decrement_when_ready)
      : task_(nullptr),
        state_(State::ThreadStartup),
        counter_to_decrement_when_ready_(counter_to_decrement_when_ready) {
    pthread_cond_init(&state_cond_, nullptr);
    pthread_mutex_init(&state_mutex_, nullptr);
    pthread_create(&thread_, nullptr, ThreadFunc, this);
  }

  ~Worker() {
    ChangeState(State::ExitAsSoonAsPossible);
    pthread_join(thread_, nullptr);
    pthread_cond_destroy(&state_cond_);
    pthread_mutex_destroy(&state_mutex_);
  }

  void ChangeState(State new_state, Task* task = nullptr) {
    pthread_mutex_lock(&state_mutex_);
    State old_state = state_.load(std::memory_order_relaxed);
    assert(old_state != new_state);
    switch (old_state) {
      case State::ThreadStartup:
        assert(new_state == State::Ready);
        break;
      case State::Ready:
        assert(new_state == State::HasWork ||
               new_state == State::ExitAsSoonAsPossible);
        break;
      case State::HasWork:
        assert(new_state == State::Ready ||
               new_state == State::ExitAsSoonAsPossible);
        break;
      default:
        abort();
    }
    switch (new_state) {
      case State::Ready:
        if (task_) {
          // Doing work is part of reverting to 'ready' state.
          task_->Run();
          task_ = nullptr;
        }
        break;
      case State::HasWork:
        assert(!task_);
        task_ = task;
        break;
      default:
        break;
    }
    state_.store(new_state, std::memory_order_relaxed);
    pthread_cond_broadcast(&state_cond_);
    pthread_mutex_unlock(&state_mutex_);
    if (new_state == State::Ready) {
      counter_to_decrement_when_ready_->DecrementCount();
    }
  }

  // Thread entry point.
  void ThreadFunc() {
    ChangeState(State::Ready);

    // Thread main loop
    while (true) {
      State state_to_act_upon = WaitForVariableChangeV2(
          &state_, State::Ready, &state_cond_, &state_mutex_);

      // We now have a state to act on, so act.
      switch (state_to_act_upon) {
        case State::HasWork:
          ChangeState(State::Ready);
          break;
        case State::ExitAsSoonAsPossible:
          return;
        default:
          abort();
      }
    }
  }

  static void* ThreadFunc(void* arg) {
    static_cast<Worker*>(arg)->ThreadFunc();
    return nullptr;
  }

  // Called by the master thead to give this worker work to do.
  void StartWork(Task* task) { ChangeState(State::HasWork, task); }

 private:

  pthread_t thread_;
  Task* task_;
  pthread_cond_t state_cond_;
  pthread_mutex_t state_mutex_;
  std::atomic<State> state_;
  BlockingCounter* const counter_to_decrement_when_ready_;
};

class WorkersPool {
 public:
  WorkersPool() {}

  ~WorkersPool() {
    for (auto w : workers_) {
      delete w;
    }
  }

  template <typename TaskType>
  void Execute(int tasks_count, TaskType* tasks) {
    assert(tasks_count >= 1);
    std::size_t workers_count = tasks_count - 1;
    CreateWorkers(workers_count);
    assert(workers_count <= workers_.size());
    counter_to_decrement_when_ready_.Reset(workers_count);
    for (std::size_t i = 0; i < tasks_count - 1; i++) {
      workers_[i]->StartWork(&tasks[i]);
    }
    Task* task = &tasks[tasks_count - 1];
    task->Run();
    counter_to_decrement_when_ready_.Wait();
  }

  void LegacyExecuteAndDestroyTasks(const std::vector<Task*>& tasks) {
    std::size_t tasks_count = tasks.size();
    assert(tasks_count >= 1);
    std::size_t workers_count = tasks_count - 1;
    CreateWorkers(workers_count);
    assert(workers_count <= workers_.size());
    counter_to_decrement_when_ready_.Reset(workers_count);
    for (int i = 0; i < tasks_count - 1; i++) {
      workers_[i]->StartWork(tasks[i]);
    }
    Task* task = tasks[tasks_count - 1];
    task->Run();
    counter_to_decrement_when_ready_.Wait();
    std::for_each(tasks.begin(), tasks.end(), [](Task* task) { delete task; });
  }

  void Execute(const std::vector<Task*>& tasks) {
    LegacyExecuteAndDestroyTasks(tasks);
  }

 private:
  void CreateWorkers(std::size_t workers_count) {
    if (workers_.size() >= workers_count) {
      return;
    }
    counter_to_decrement_when_ready_.Reset(workers_count - workers_.size());
    while (workers_.size() < workers_count) {
      workers_.push_back(new Worker(&counter_to_decrement_when_ready_));
    }
    counter_to_decrement_when_ready_.Wait();
  }
  WorkersPool(const WorkersPool&) = delete;
  std::vector<Worker*> workers_;

  BlockingCounter counter_to_decrement_when_ready_;
};

class MultiThreadContext {
 public:
  WorkersPool* workers_pool() { return &workers_pool_; }
  void set_max_num_threads(int n) { max_num_threads_ = n; }

  int max_num_threads() const { return max_num_threads_; }

 private:
  WorkersPool workers_pool_;

 protected:
  int max_num_threads_ = 1;
};

class secda_threading {
 public:
  std::vector<std::thread> threads;

  void add_thread(std::thread t) { threads.push_back(std::move(t)); }

  void join_threads() {
    for (auto& th : threads) th.join();
  }
};

#endif  // SECDA_THREADING