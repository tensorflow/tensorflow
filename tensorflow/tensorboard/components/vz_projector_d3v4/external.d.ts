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

// TODO(smilkov): Split into weblas.d.ts and numeric.d.ts and write
// typings for numeric.
interface Tensor {
  new(size: [number, number], data: Float32Array);
  transfer(): Float32Array;
  delete(): void;
}

interface Weblas {
  sgemm(M: number, N: number, K: number, alpha: number,
      A: Float32Array, B: Float32Array, beta: number, C: Float32Array):
      Float32Array;
  pipeline: {
     Tensor: Tensor;
     sgemm(alpha: number, A: Tensor, B: Tensor, beta: number,
         C: Tensor): Tensor;
  };
  util: {
    transpose(M: number, N: number, data: Float32Array): Tensor;
  };

}

declare let numeric: any;
declare let weblas: Weblas;

interface AnalyticsEventType {
  hitType: string;
  page?: string;
  eventCategory?: string;
  eventAction?: string;
  eventLabel?: string;
  eventValue?: number;
}

declare let ga: (command: string, eventObj: AnalyticsEventType) => void;