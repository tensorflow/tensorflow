/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_DATA_SERVICE_AUTO_SCALER_H_
#define TENSORFLOW_CORE_DATA_SERVICE_AUTO_SCALER_H_

namespace tensorflow {
namespace data {

// Exports a metric (/tensorflow/data/service/optimal_number_of_workers) with
// the current estimated optimal number of tf.data service workers, according to
// the observed cluster workload.
//
// Glossary:
// * Consumer: A client that consumes elements from tf.data service.
// * Worker: A tf.data service worker.
// * Processing time (PT): The estimated time it takes a worker to process and
// produce an element.
// * Target processing time (TPT): From the perspective of a consumer,
// it is the maximum time a tf.data input pipeline can take to produce an
// element such that the downstream processor wait time is 0. In other words,
// this is the ideal time the tf.data pipeline should take to produce an element
// so that training doesn't slow down due to waiting for elements. This means
// that we want processing time <= target processing time, so that when an
// element is requested, the pipeline has processed it already.
// * Worker throughput (WT): It is the multiplicative inverse of processing time
// (1 / PT). This refers to the number of elements produced by a worker per
// second.
// * Consumption rate (CR): It is the multiplicative inverse of target
// processing time (1 / TPT). This refers to the number of elements requested by
// a consumer per second.
//
// **AutoScaler overview**
//
// 1. It keeps track of the most recent worker throughputs reported by each
// worker in the data service cluster, as well as the most recent consumption
// rates reported by each consumer. WTs and CRs are derived from reporting PTs
// and TPTs, respectively.
// 2. Having this information, it estimates the optimal number of workers N as
// follows:
//  N = (Sum of CRs reported by all consumers) /
//      (Average of WTs reported by all workers)
//
// AutoScaler is thread-safe.
class AutoScaler {
 public:
  AutoScaler() = default;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_AUTO_SCALER_H_
