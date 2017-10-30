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
    string topic = parts[0];
    int32 partition = 0;
    if (parts.size() > 1) {
      if (!strings::safe_strto32(parts[1], &partition)) {
        return errors::InvalidArgument("Invalid parameters: ", current_work());
      }
    }
    int64 offset = 0;
    if (parts.size() > 2) {
      if (!strings::safe_strto64(parts[2], &offset)) {
        return errors::InvalidArgument("Invalid parameters: ", current_work());
      }
    }

    topic_partition_.reset(
        RdKafka::TopicPartition::create(topic, partition, offset));

    offset_ = topic_partition_->offset();
    limit_ = -1;
    if (parts.size() > 3) {
      if (!strings::safe_strto64(parts[3], &limit_)) {
        return errors::InvalidArgument("Invalid parameters: ", current_work());
      }
    }

    std::unique_ptr<RdKafka::Conf> conf(
        RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL));
    std::unique_ptr<RdKafka::Conf> topic_conf(
        RdKafka::Conf::create(RdKafka::Conf::CONF_TOPIC));

    std::string errstr;

    RdKafka::Conf::ConfResult result =
        conf->set("default_topic_conf", topic_conf.get(), errstr);
    if (result != RdKafka::Conf::CONF_OK) {
      return errors::Internal("Failed to set default_topic_conf:", errstr);
    }

    result = conf->set("bootstrap.servers", servers_, errstr);
    if (result != RdKafka::Conf::CONF_OK) {
      return errors::Internal("Failed to set bootstrap.servers ", servers_, ":",
                              errstr);
    }
    result = conf->set("group.id", group_, errstr);
    if (result != RdKafka::Conf::CONF_OK) {
      return errors::Internal("Failed to set group.id ", group_, ":", errstr);
    }

    consumer_.reset(RdKafka::KafkaConsumer::create(conf.get(), errstr));
    if (!consumer_.get()) {
      return errors::Internal("Failed to create consumer:", errstr);
    }

    std::vector<RdKafka::TopicPartition*> partitions;
    partitions.emplace_back(topic_partition_.get());
    RdKafka::ErrorCode err = consumer_->assign(partitions);
    if (err != RdKafka::ERR_NO_ERROR) {
      return errors::Internal(
          "Failed to assign partition [", topic_partition_->topic(), ", ",
          topic_partition_->partition(), ", ", topic_partition_->offset(), "]:",
          RdKafka::err2str(err));
    }

    return Status::OK();
  }

  Status OnWorkFinishedLocked() override {
    consumer_->unassign();
    consumer_->close();
    consumer_.reset(nullptr);
    return Status::OK();
  }

  Status ReadLocked(string* key, string* value, bool* produced,
                    bool* at_end) override {
    if (limit_ >= 0 &&
        (topic_partition_->offset() >= limit_ || offset_ >= limit_)) {
      *at_end = true;
      return Status::OK();
    }
    while (true) {
      std::unique_ptr<RdKafka::Message> message(consumer_->consume(1000));
      if (message->err() == RdKafka::ERR_NO_ERROR) {
        if (message->key()) {
          *key = strings::StrCat(topic_partition_->topic(), ":",
                                 topic_partition_->partition(), ":",
                                 message->offset(), ":", *message->key());
        } else {
          *key = strings::StrCat(topic_partition_->topic(), ":",
                                 topic_partition_->partition(), ":",
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
    consumer_->unassign();
    consumer_->close();
    consumer_.reset(nullptr);
    return ReaderBase::ResetLocked();
  }

 private:
  std::string servers_;
  std::string group_;
  bool eof_;
  int64 limit_;
  std::unique_ptr<RdKafka::KafkaConsumer> consumer_;
  std::unique_ptr<RdKafka::TopicPartition> topic_partition_;
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
