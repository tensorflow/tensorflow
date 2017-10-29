/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/reader_base.h"
#include "tensorflow/core/framework/reader_op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"

#include "src-cpp/rdkafkacpp.h"

namespace tensorflow {

class KafkaReader : public ReaderBase {
 public:
  KafkaReader(const string& servers, const string& group, const bool eof,
              const string& name)
      : ReaderBase(strings::StrCat("KafkaReader '", name, "'")),
        servers_(servers),
        group_(group),
        eof_(eof) {}

  Status OnWorkStartedLocked() override {
    std::vector<string> parts = str_util::Split(current_work(), ":");
    if (parts.size() < 1) {
      return errors::InvalidArgument("Invalid parameters: ", current_work());
    }
    topic_str_ = parts[0];
    topic_partition_ = 0;
    if (parts.size() > 1) {
      if (!strings::safe_strto32(parts[1], &topic_partition_)) {
        return errors::InvalidArgument("Invalid parameters: ", current_work());
      }
    }
    topic_offset_ = 0;
    if (parts.size() > 2) {
      if (!strings::safe_strto64(parts[2], &topic_offset_)) {
        return errors::InvalidArgument("Invalid parameters: ", current_work());
      }
    }
    offset_ = topic_offset_;

    topic_limit_ = -1;
    if (parts.size() > 3) {
      if (!strings::safe_strto64(parts[3], &topic_limit_)) {
        return errors::InvalidArgument("Invalid parameters: ", current_work());
      }
    }

    std::unique_ptr<RdKafka::Conf> conf(
        RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL));
    std::unique_ptr<RdKafka::Conf> topic_conf(
        RdKafka::Conf::create(RdKafka::Conf::CONF_TOPIC));

    std::string errstr;

    RdKafka::Conf::ConfResult result =
        conf->set("bootstrap.servers", servers_, errstr);
    if (result != RdKafka::Conf::CONF_OK) {
      return errors::Internal("Failed to set bootstrap.servers ", servers_, ":",
                              errstr);
    }
    if (group_.length() != 0) {
      RdKafka::Conf::ConfResult result = conf->set("group.id", group_, errstr);
      if (result != RdKafka::Conf::CONF_OK) {
        return errors::Internal("Failed to set group.id ", group_, ":", errstr);
      }
    }

    consumer_.reset(RdKafka::Consumer::create(conf.get(), errstr));
    if (!consumer_.get()) {
      return errors::Internal("Failed to create consumer:", errstr);
    }

    topic_.reset(RdKafka::Topic::create(consumer_.get(), topic_str_,
                                        topic_conf.get(), errstr));
    if (!topic_.get()) {
      return errors::Internal("Failed to create topic:", errstr);
    }

    RdKafka::ErrorCode resp =
        consumer_->start(topic_.get(), topic_partition_, topic_offset_);
    if (resp != RdKafka::ERR_NO_ERROR) {
      return errors::Internal("Failed to start consumer:",
                              RdKafka::err2str(resp));
    }
    return Status::OK();
  }

  Status OnWorkFinishedLocked() override {
    consumer_->stop(topic_.get(), topic_partition_);
    consumer_->poll(1000);
    topic_.reset(nullptr);
    consumer_.reset(nullptr);
    return Status::OK();
  }

  Status ReadLocked(string* key, string* value, bool* produced,
                    bool* at_end) override {
    if (topic_limit_ >= 0 &&
        (topic_offset_ >= topic_limit_ || offset_ >= topic_limit_)) {
      *at_end = true;
      return Status::OK();
    }
    while (true) {
      std::unique_ptr<RdKafka::Message> message(
          consumer_->consume(topic_.get(), topic_partition_, 1000));
      if (message->err() == RdKafka::ERR_NO_ERROR) {
        if (message->key()) {
          *key = strings::StrCat(topic_str_, ":", topic_partition_, ":",
                                 message->offset(), ":", *message->key());
        } else {
          *key = strings::StrCat(topic_str_, ":", topic_partition_, ":",
                                 message->offset());
        }

        *value = std::string(static_cast<const char*>(message->payload()),
                             message->len());
        *produced = true;
        // Sync offset
        offset_ = message->offset();
        return Status::OK();
      } else if (message->err() == RdKafka::ERR__PARTITION_EOF) {
        if (eof_) {
          *at_end = true;
          return Status::OK();
        }

      } else if (message->err() != RdKafka::ERR__TIMED_OUT) {
        return errors::Internal("Failed to consume:", message->errstr());
      }
      message.reset(nullptr);
      consumer_->poll(0);
    }
    return Status::OK();
  }

  Status ResetLocked() override {
    consumer_->stop(topic_.get(), topic_partition_);
    consumer_->poll(1000);
    topic_.reset(nullptr);
    consumer_.reset(nullptr);
    return ReaderBase::ResetLocked();
  }

 private:
  std::string servers_;
  std::string group_;
  bool eof_;
  std::string topic_str_;
  int32 topic_partition_;
  int64 topic_offset_;
  int64 topic_limit_;
  std::unique_ptr<RdKafka::Consumer> consumer_;  // must outlive topic_
  std::unique_ptr<RdKafka::Topic> topic_;
  int64 offset_;
};

class KafkaReaderOp : public ReaderOpKernel {
 public:
  explicit KafkaReaderOp(OpKernelConstruction* context)
      : ReaderOpKernel(context) {
    std::string servers;
    OP_REQUIRES_OK(context, context->GetAttr("servers", &servers));
    std::string group;
    OP_REQUIRES_OK(context, context->GetAttr("group", &group));
    bool eof;
    OP_REQUIRES_OK(context, context->GetAttr("eof", &eof));
    SetReaderFactory([this, servers, group, eof]() {
      return new KafkaReader(servers, group, eof, name());
    });
  }
};

REGISTER_KERNEL_BUILDER(Name("KafkaReader").Device(DEVICE_CPU), KafkaReaderOp);

}  // namespace tensorflow
