/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_TSL_PLATFORM_DEBUG_ME_CONTEXT_H_
#define XLA_TSL_PLATFORM_DEBUG_ME_CONTEXT_H_

#include <string>
#include <type_traits>
#include <vector>

#include "absl/container/flat_hash_map.h"

namespace tsl {

// DebugMeContext: A thread-local, RAII-based context manager for debugging.
//
// This class provides a mechanism to associate contextual string information
// with specific keys within a given scope. It is designed to be used with the
// RAII (Resource Acquisition Is Initialization) pattern: creating an object on
// the stack pushes a context string, and the context is automatically popped
// when the object goes out of scope.
//
// Typical usage involves creating an instance of `DebugMeContext` at the
// beginning of a scope (e.g., a function, a loop, or a compiler pass) to "tag"
// that scope with some information. Deeper in the call stack, other code can
// then retreive the current context.
//
//
// Note: The entire context is thread-local, meaning each thread maintains its
// own independent set of key-value stacks. If you are spawning new threads and
// want the context to carry forward, it is your responsibility to shuttle the
// context over the boundary between threads.
//
//
// Example usage:
//
//   // 1. Define the types of context you want to track.
//   enum class DebugContextKey { kCompilerPass, kModuleName };
//
//   void RunMyCompilerPass() {
//     // 2. At the start of the pass, push the pass name onto the context
//     stack. DebugMeContext<DebugContextKey>
//     ctx(DebugContextKey::kCompilerPass, "MyPass");
//
//     // 3. Now, any code executed before `ctx` goes out of scope, when
//     querying the context, will see "MyPass" on the kCompilerPass context
//     stack. GenerateError("Something went wrong");
//
//   } // <-- `ctx` is destroyed here, and "MyPass" is popped from the stack.
//
//   void GenerateError(const std::string& msg) {
//     // 4. Retrieve the current compiler pass stack.
//     std::vector<std::string> pass_stack =
//         DebugMeContext<DebugContextKey>::GetValues(DebugContextKey::kCompilerPass);
//     if (!pass_stack.empty()) {
//       // pass_stack.back() will be "MyPass"
//       LOG(ERROR) << "Error in pass '" << pass_stack.back() << "': " << msg;
//     }
//   }
//
//
// Template Parameters:
//   KeyType: An enum used to identify different context stacks.
template <typename KeyType,
          typename = std::enable_if_t<std::is_enum_v<KeyType>>>
class DebugMeContext {
 public:
  DebugMeContext(KeyType key, const std::string& value) : key_(key) {
    std::vector<std::string>& value_stack = context_.key_value_stack_map[key];
    value_stack.push_back(value);
  }
  ~DebugMeContext() {
    std::vector<std::string>& value_stack = context_.key_value_stack_map[key_];
    value_stack.pop_back();
  }

  // Returns the values associated with the given key. If the key does not
  // exist, returns an empty vector.
  static std::vector<std::string> GetValues(KeyType key) {
    // Here we use a const reference of `context_` to try to minimize the chance
    // of this function modifying `context_`.
    const Context& const_context = context_;
    auto it = const_context.key_value_stack_map.find(key);
    if (it == const_context.key_value_stack_map.end()) {
      // Key does not exist in map.
      return {};
    }
    return it->second;
  }

  // Returns true if any key has a non-empty stack.
  static bool HasAnyValues() {
    const Context& const_context = context_;
    for (const auto& pair : const_context.key_value_stack_map) {
      if (!pair.second.empty()) {
        return true;
      }
    }

    return false;
  }

 private:
  struct Context {
    absl::flat_hash_map<KeyType, std::vector<std::string>> key_value_stack_map;
  };

  // We disable linting here to avoid complaints about global (static) variables
  // with non-trivial destructors ("static deinitialization order fiasco").
  // thread_locals are not subject to this potential problem because only the
  // owning thread is calling the destructors, which, by definition, means that
  // the thread is not also using the object.
  inline static thread_local Context context_;  // NOLINT
  const KeyType key_;
};

}  // namespace tsl

#endif  // XLA_TSL_PLATFORM_DEBUG_ME_CONTEXT_H_
