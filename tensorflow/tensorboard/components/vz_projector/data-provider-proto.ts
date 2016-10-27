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

import {DataPoint, DataProto, DataSet, MetadataInfo, PointMetadata, State} from './data';
import {analyzeMetadata, CheckpointInfo, DataProvider} from './data-provider';


export class ProtoDataProvider implements DataProvider {
  private dataProto: DataProto;

  constructor(dataProto: DataProto) {
    this.dataProto = dataProto;
  }

  retrieveRuns(callback: (runs: string[]) => void): void {
    callback(['proto']);
  }

  retrieveCheckpointInfo(run: string, callback: (d: CheckpointInfo) => void) {
    callback({
      tensors: {
        'proto': {
          name: 'proto',
          shape: this.dataProto.shape,
          metadataFile: 'proto',
          bookmarksFile: null
        }
      },
      checkpointFile: 'proto'
    });
  }

  retrieveTensor(run: string, tensorName: string,
      callback: (ds: DataSet) => void) {
    callback(this.flatArrayToDataset(this.dataProto.tensor));
  }

  retrieveMetadata(run: string, tensorName: string,
      callback: (r: MetadataInfo) => void): void {
    let columnNames = this.dataProto.metadata.columns.map(c => c.name);
    let n = this.dataProto.shape[0];
    let pointsMetadata: PointMetadata[] = new Array(n);
    this.dataProto.metadata.columns.forEach(c => {
      let values = c.numericValues || c.stringValues;
      for (let i = 0; i < n; i++) {
        pointsMetadata[i] = pointsMetadata[i] || {};
        pointsMetadata[i][c.name] = values[i];
      }
    });
    callback({
      stats: analyzeMetadata(columnNames, pointsMetadata),
      pointsInfo: pointsMetadata
    });
  }

  getDefaultTensor(run: string, callback: (tensorName: string) => void): void {
    callback('proto');
  }

  getBookmarks(run: string, tensorName: string,
      callback: (r: State[]) => void): void {
    return callback([]);
  }

  private flatArrayToDataset(tensor: number[]): DataSet {
    let points: DataPoint[] = [];
    let n = this.dataProto.shape[0];
    let d = this.dataProto.shape[1];
    if (n * d !== tensor.length) {
      throw 'The shape doesn\'t match the length of the flattened array';
    }
    for (let i = 0; i < n; i++) {
      let vector: number[] = [];
      let offset = i * d;
      for (let j = 0; j < d; j++) {
        vector.push(tensor[offset++]);
      }
      points.push({
        vector: vector,
        metadata: {},
        projections: null,
        projectedPoint: null,
        index: i
      });
    }
    return new DataSet(points);
  }
}
