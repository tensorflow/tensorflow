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

import {DataSet, SpriteAndMetadataInfo, State} from './data';
import {ProjectorConfig, DataProvider, EmbeddingInfo, TENSORS_MSG_ID} from './data-provider';
import * as dataProvider from './data-provider';
import * as logging from './logging';

/** Data provider that loads data from a demo folder. */
export class DemoDataProvider implements DataProvider {
  /** List of demo datasets for showing the capabilities of the tool. */
  private DEMO_CONFIG: ProjectorConfig = {
    embeddings: [
      {
        tensorName: 'Word2Vec 5K',
        tensorShape: [5000, 200],
        tensorPath: 'word2vec_5000_200d_tensors.tsv',
        metadataPath: 'word2vec_5000_200d_labels.tsv'
      },
      {
        tensorName: 'Word2Vec 10K',
        tensorShape: [10000, 200],
        tensorPath: 'word2vec_10000_200d_tensors.tsv',
        metadataPath: 'word2vec_10000_200d_labels.tsv'
      },
      {
        tensorName: 'Word2Vec All',
        tensorShape: [71291, 200],
        tensorPath: 'word2vec_full_200d_tensors.tsv',
        metadataPath: 'word2vec_full_200d_labels.tsv'
      },
      {
        tensorName: 'SmartReply 5K',
        tensorShape: [5000, 256],
        tensorPath: 'smartreply_5000_256d_tensors.tsv',
        metadataPath: 'smartreply_5000_256d_labels.tsv'
      },
      {
        tensorName: 'SmartReply All',
        tensorShape: [35860, 256],
        tensorPath: 'smartreply_full_256d_tensors.tsv',
        metadataPath: 'smartreply_full_256d_labels.tsv'
      },
      {
        tensorName: 'Mnist with images 10K',
        tensorShape: [10000, 784],
        tensorPath: 'mnist_10k_784d_tensors.tsv',
        metadataPath: 'mnist_10k_784d_labels.tsv',
        sprite: {
          imagePath: 'mnist_10k_sprite.png',
          singleImageDim: [28, 28]
        }
      },
      {
        tensorName: 'Iris',
        tensorShape: [150, 4],
        tensorPath: 'iris_tensors.tsv',
        metadataPath: 'iris_labels.tsv'
      },
      {
        tensorName: 'Unit Cube',
        tensorShape: [8, 3],
        tensorPath: 'cube_tensors.tsv',
        metadataPath: 'cube_metadata.tsv'
      }
    ],
    modelCheckpointPath: 'Demo datasets'
  };
  /** Name of the folder where the demo datasets are stored. */
  private DEMO_FOLDER = 'data';

  private getEmbeddingInfo(tensorName: string): EmbeddingInfo {
    let embeddings = this.DEMO_CONFIG.embeddings;
    for (let i = 0; i < embeddings.length; i++) {
      let embedding = embeddings[i];
      if (embedding.tensorName === tensorName) {
        return embedding;
      }
    }
    return null;
  }

  retrieveRuns(callback: (runs: string[]) => void): void {
    callback(['Demo']);
  }

  retrieveProjectorConfig(run: string, callback: (d: ProjectorConfig) => void)
      : void {
    callback(this.DEMO_CONFIG);
  }

  getDefaultTensor(run: string, callback: (tensorName: string) => void) {
    callback('SmartReply 5K');
  }

  retrieveTensor(run: string, tensorName: string,
      callback: (ds: DataSet) => void) {
    let embedding = this.getEmbeddingInfo(tensorName);
    let separator = embedding.tensorPath.substr(-3) === 'tsv' ? '\t' : ' ';
    let url = `${this.DEMO_FOLDER}/${embedding.tensorPath}`;
    logging.setModalMessage('Fetching tensors...', TENSORS_MSG_ID);
    d3.text(url, (error: any, dataString: string) => {
      if (error) {
        logging.setModalMessage('Error: ' + error.responseText);
        return;
      }
      dataProvider.parseTensors(dataString, separator).then(points => {
        callback(new DataSet(points));
      });
    });
  }

  retrieveSpriteAndMetadata(run: string, tensorName: string,
      callback: (r: SpriteAndMetadataInfo) => void) {
    let embedding = this.getEmbeddingInfo(tensorName);
    let metadataPath = null;
    if (embedding.metadataPath) {
      metadataPath = `${this.DEMO_FOLDER}/${embedding.metadataPath}`;
    }
    let spriteImagePath = null;
    if (embedding.sprite && embedding.sprite.imagePath) {
      spriteImagePath = `${this.DEMO_FOLDER}/${embedding.sprite.imagePath}`;
    }
    dataProvider.retrieveSpriteAndMetadataInfo(metadataPath, spriteImagePath,
        embedding.sprite, callback);
  }

  getBookmarks(
      run: string, tensorName: string, callback: (r: State[]) => void) {
    callback([]);
  }
}
