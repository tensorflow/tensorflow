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

import {DataSet, DatasetMetadata, MetadataInfo, State} from './data';
import {CheckpointInfo, DataProvider, fetchImage, METADATA_MSG_ID, parseMetadata, parseTensors, TensorInfo, TENSORS_MSG_ID} from './data-provider';
import * as logging from './logging';


type DemoDataset = {
  fpath: string; metadata_path?: string; metadata?: DatasetMetadata;
  bookmarks_path?: string;
  shape: [number, number];
};

/** Data provider that loads data from a demo folder. */
export class DemoDataProvider implements DataProvider {
  /** List of demo datasets for showing the capabilities of the tool. */
  private static DEMO_DATASETS: {[name: string]: DemoDataset} = {
    'Word2Vec 5K': {
      shape: [5000, 200],
      fpath: 'word2vec_5000_200d_tensors.tsv',
      metadata_path: 'word2vec_5000_200d_labels.tsv'
    },
    'Word2Vec 10K': {
      shape: [10000, 200],
      fpath: 'word2vec_10000_200d_tensors.tsv',
      metadata_path: 'word2vec_10000_200d_labels.tsv'
    },
    'Word2Vec All': {
      shape: [71291, 200],
      fpath: 'word2vec_full_200d_tensors.tsv',
      metadata_path: 'word2vec_full_200d_labels.tsv'
    },
    'SmartReply 5K': {
      shape: [5000, 256],
      fpath: 'smartreply_5000_256d_tensors.tsv',
      metadata_path: 'smartreply_5000_256d_labels.tsv'
    },
    'SmartReply All': {
      shape: [35860, 256],
      fpath: 'smartreply_full_256d_tensors.tsv',
      metadata_path: 'smartreply_full_256d_labels.tsv'
    },
    'Mnist with images 10K': {
      shape: [10000, 784],
      fpath: 'mnist_10k_784d_tensors.tsv',
      metadata_path: 'mnist_10k_784d_labels.tsv',
      metadata: {
        image:
            {sprite_fpath: 'mnist_10k_sprite.png', single_image_dim: [28, 28]}
      },
    },
    'Iris': {
      shape: [150, 4],
      fpath: 'iris_tensors.tsv',
      metadata_path: 'iris_labels.tsv'
    },
    'Unit Cube': {
      shape: [8, 3],
      fpath: 'cube_tensors.tsv',
      metadata_path: 'cube_metadata.tsv'
    }
  };
  /** Name of the folder where the demo datasets are stored. */
  private static DEMO_FOLDER = 'data';

  retrieveRuns(callback: (runs: string[]) => void): void {
    callback(['Demo']);
  }

  retrieveCheckpointInfo(run: string, callback: (d: CheckpointInfo) => void)
      : void {
    let tensorsInfo: {[name: string]: TensorInfo} = {};
    for (let name in DemoDataProvider.DEMO_DATASETS) {
      if (!DemoDataProvider.DEMO_DATASETS.hasOwnProperty(name)) {
        continue;
      }
      let demoInfo = DemoDataProvider.DEMO_DATASETS[name];
      tensorsInfo[name] = {
        name: name,
        shape: demoInfo.shape,
        metadataFile: demoInfo.metadata_path,
        bookmarksFile: demoInfo.bookmarks_path
      };
    }
    callback({
      tensors: tensorsInfo,
      checkpointFile: 'Demo datasets',
    });
  }

  getDefaultTensor(run: string, callback: (tensorName: string) => void) {
    callback('SmartReply 5K');
  }

  retrieveTensor(run: string, tensorName: string,
      callback: (ds: DataSet) => void) {
    let demoDataSet = DemoDataProvider.DEMO_DATASETS[tensorName];
    let separator = demoDataSet.fpath.substr(-3) === 'tsv' ? '\t' : ' ';
    let url = `${DemoDataProvider.DEMO_FOLDER}/${demoDataSet.fpath}`;
    logging.setModalMessage('Fetching tensors...', TENSORS_MSG_ID);
    d3.text(url, (error: any, dataString: string) => {
      if (error) {
        logging.setModalMessage('Error: ' + error.responseText);
        return;
      }
      parseTensors(dataString, separator).then(points => {
        callback(new DataSet(points));
      });
    });
  }

  retrieveMetadata(run: string, tensorName: string,
      callback: (r: MetadataInfo) => void) {
    let demoDataSet = DemoDataProvider.DEMO_DATASETS[tensorName];
    let dataSetPromise: Promise<MetadataInfo> = null;
    if (demoDataSet.metadata_path) {
      dataSetPromise = new Promise<MetadataInfo>((resolve, reject) => {
        logging.setModalMessage('Fetching metadata...', METADATA_MSG_ID);
        d3.text(
            `${DemoDataProvider.DEMO_FOLDER}/${demoDataSet.metadata_path}`,
            (err: any, rawMetadata: string) => {
              if (err) {
                logging.setModalMessage('Error: ' + err.responseText);
                reject(err);
                return;
              }
              resolve(parseMetadata(rawMetadata));
            });
      });
    }
    let spriteMsgId = null;
    let spritesPromise: Promise<HTMLImageElement> = null;
    if (demoDataSet.metadata && demoDataSet.metadata.image) {
      let spriteFilePath = demoDataSet.metadata.image.sprite_fpath;
      spriteMsgId = logging.setModalMessage('Fetching sprite image...');
      spritesPromise =
          fetchImage(`${DemoDataProvider.DEMO_FOLDER}/${spriteFilePath}`);
    }

    // Fetch the metadata and the image in parallel.
    Promise.all([dataSetPromise, spritesPromise]).then(values => {
      if (spriteMsgId) {
        logging.setModalMessage(null, spriteMsgId);
      }
      let [metadata, spriteImage] = values;
      metadata.spriteImage = spriteImage;
      metadata.datasetInfo = demoDataSet.metadata;
      callback(metadata);
    });
  }

  getBookmarks(
      run: string, tensorName: string, callback: (r: State[]) => void) {
    callback([]);
  }
}
