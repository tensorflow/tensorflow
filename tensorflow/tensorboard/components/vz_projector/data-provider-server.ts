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

import {DataSet, MetadataInfo, State} from './data';
import {CheckpointInfo, DataProvider, METADATA_MSG_ID, parseMetadata, parseTensors, TENSORS_MSG_ID} from './data-provider';
import * as logging from './logging';


/**
 * Data provider that loads data provided by a python server (usually backed
 * by a checkpoint file).
 */
export class ServerDataProvider implements DataProvider {
  private routePrefix: string;
  private runCheckpointInfoCache: {[run: string]: CheckpointInfo} = {};

  constructor(routePrefix: string) {
    this.routePrefix = routePrefix;
  }

  retrieveRuns(callback: (runs: string[]) => void): void {
    let msgId = logging.setModalMessage('Fetching runs...');
    d3.json(`${this.routePrefix}/runs`, (err, runs) => {
      if (err) {
        logging.setModalMessage('Error: ' + err.responseText);
        return;
      }
      logging.setModalMessage(null, msgId);
      callback(runs);
    });
  }

  retrieveCheckpointInfo(run: string, callback: (d: CheckpointInfo) => void)
      : void {
    if (run in this.runCheckpointInfoCache) {
      callback(this.runCheckpointInfoCache[run]);
      return;
    }

    let msgId = logging.setModalMessage('Fetching checkpoint info...');
    d3.json(`${this.routePrefix}/info?run=${run}`, (err, checkpointInfo) => {
      if (err) {
        logging.setModalMessage('Error: ' + err.responseText);
        return;
      }
      logging.setModalMessage(null, msgId);
      this.runCheckpointInfoCache[run] = checkpointInfo;
      callback(checkpointInfo);
    });
  }

  retrieveTensor(run: string, tensorName: string, callback: (ds: DataSet) => void) {
    // Get the tensor.
    logging.setModalMessage('Fetching tensor values...', TENSORS_MSG_ID);
    d3.text(
        `${this.routePrefix}/tensor?run=${run}&name=${tensorName}`,
        (err: any, tsv: string) => {
          if (err) {
            logging.setModalMessage('Error: ' + err.responseText);
            return;
          }
          parseTensors(tsv).then(dataPoints => {
            callback(new DataSet(dataPoints));
          });
        });
  }

  retrieveMetadata(run: string, tensorName: string,
      callback: (r: MetadataInfo) => void) {
    logging.setModalMessage('Fetching metadata...', METADATA_MSG_ID);
    d3.text(
        `${this.routePrefix}/metadata?run=${run}&name=${tensorName}`,
        (err: any, rawMetadata: string) => {
          if (err) {
            logging.setModalMessage('Error: ' + err.responseText);
            return;
          }
          parseMetadata(rawMetadata).then(result => callback(result));
        });
  }

  getDefaultTensor(run: string, callback: (tensorName: string) => void) {
    this.retrieveCheckpointInfo(run, checkpointInfo => {
      let tensorNames = Object.keys(checkpointInfo.tensors);
      // Return the first tensor that has metadata.
      for (let i = 0; i < tensorNames.length; i++) {
        let tensorName = tensorNames[i];
        if (checkpointInfo.tensors[tensorName].metadataFile) {
          callback(tensorName);
          return;
        }
      }
      callback(tensorNames.length >= 1 ? tensorNames[0] : null);
    });
  }

  getBookmarks(
      run: string, tensorName: string, callback: (r: State[]) => void) {
    let msgId = logging.setModalMessage('Fetching bookmarks...');
    d3.json(
        `${this.routePrefix}/bookmarks?run=${run}&name=${tensorName}`,
        (err, bookmarks) => {
          logging.setModalMessage(null, msgId);
          if (!err) {
            callback(bookmarks as State[]);
          }
        });
  }
}
