/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/contrib/lite/testing/message.h"

#include <stack>

#include "tensorflow/contrib/lite/testing/tokenize.h"

namespace tflite {
namespace testing {

// A token processor that builds messages and forward calls to the current
// message object. Place a new message at the top of the stack when it start
// and remove it when it is finished.
class MessageStack : public TokenProcessor {
 public:
  // Start a new MessageStack with the given first_node, which will be used to
  // process freestanding fields and submessages.
  explicit MessageStack(Message* first_node) {
    nodes_.push(first_node);
    valid_ = true;
  }

  void ConsumeToken(std::string* token) override {
    if (!valid_) return;
    Message* current_node = nodes_.top();
    if (*token == "{") {
      // This is the beginning of a new message, names after the previous token.
      if (previous_token_.empty()) {
        valid_ = false;
        return;
      }
      nodes_.push(current_node ? current_node->AddChild(previous_token_)
                               : nullptr);
      previous_token_.clear();
    } else if (*token == "}") {
      // A message is being completed. There should be no previous token.  Note
      // that the top-level message never closes, so we should always have at
      // least one entry in the stack.
      if (nodes_.size() == 1 || !previous_token_.empty()) {
        valid_ = false;
        return;
      }
      if (current_node) {
        current_node->Finish();
      }
      nodes_.pop();
    } else if (*token == ":") {
      // We reached the end of the 'key' portion of a field. Store the token
      // until we have the 'value' portion.
      if (previous_token_.empty()) {
        valid_ = false;
        return;
      }
    } else {
      if (previous_token_.empty()) {
        previous_token_.swap(*token);
      } else {
        // This is the 'value' portion of a field. The previous token is the
        // 'key'.
        if (current_node) {
          current_node->SetField(previous_token_, *token);
        }
        previous_token_.clear();
      }
    }
  }

  bool valid() const { return valid_; }

 private:
  std::stack<Message*> nodes_;
  std::string previous_token_;
  bool valid_;
};

bool Message::Read(std::istream* input, Message* message) {
  MessageStack stack(message);
  Tokenize(input, &stack);
  return stack.valid();
}

}  // namespace testing
}  // namespace tflite
