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

const BYTES_EXTENSION = '.bytes';

/** Data provider that loads data from a demo folder. */
export class DemoDataProvider implements DataProvider {
  private projectorConfigPath: string;
  private projectorConfig: ProjectorConfig;

  constructor(projectorConfigPath: string) {
    this.projectorConfigPath = projectorConfigPath;
  }

  private getEmbeddingInfo(tensorName: string): EmbeddingInfo {
    let embeddings = this.projectorConfig.embeddings;
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
    const msgId = logging.setModalMessage('Fetching projector config...');

    const xhr = new XMLHttpRequest();
    xhr.open('GET', this.projectorConfigPath);
    xhr.onerror = (err) => {
      let errorMessage = err.message;
      // If the error is a valid XMLHttpResponse, it's possible this is a
      // cross-origin error.
      if (xhr.responseText != null) {
        errorMessage = 'Cannot fetch projector config, possibly a ' +
            'Cross-Origin request error.';
      }
      logging.setErrorMessage(errorMessage, 'fetching projector config');
    };
    xhr.onload = () => {
      const projectorConfig = JSON.parse(xhr.responseText) as ProjectorConfig;
      logging.setModalMessage(null, msgId);
      this.projectorConfig = projectorConfig;
      callback(projectorConfig);
    };
    xhr.send();
  }

  retrieveTensor(run: string, tensorName: string,
      callback: (ds: DataSet) => void) {
    let embedding = this.getEmbeddingInfo(tensorName);
    let url = `${embedding.tensorPath}`;
    if (embedding.tensorPath.substr(-1 * BYTES_EXTENSION.length) ===
        BYTES_EXTENSION) {
      dataProvider.retrieveTensorAsBytes(
          this, this.getEmbeddingInfo(tensorName), run, tensorName, url,
          callback);
    } else {
      logging.setModalMessage('Fetching tensors...', TENSORS_MSG_ID);
      const request = new XMLHttpRequest();
      request.open('GET', url);
      request.responseType = 'arraybuffer';

      request.onerror = () => {
        logging.setErrorMessage(request.responseText, 'fetching tensors');
      };
      request.onload = () => {
        dataProvider.parseTensors(request.response).then(points => {
          callback(new DataSet(points));
        });
      };
      request.send();
    }
  }

  retrieveSpriteAndMetadata(run: string, tensorName: string,
      callback: (r: SpriteAndMetadataInfo) => void) {
    let embedding = this.getEmbeddingInfo(tensorName);
    let spriteImagePath = null;
    if (embedding.sprite && embedding.sprite.imagePath) {
      spriteImagePath = embedding.sprite.imagePath;
    }
    dataProvider.retrieveSpriteAndMetadataInfo(
        embedding.metadataPath, spriteImagePath, embedding.sprite, callback);
  }

  getBookmarks(
      run: string, tensorName: string, callback: (r: State[]) => void) {
    let embedding = this.getEmbeddingInfo(tensorName);
    let msgId = logging.setModalMessage('Fetching bookmarks...');

    const xhr = new XMLHttpRequest();
    xhr.open('GET', embedding.bookmarksPath);
    xhr.onerror = (err) => {
      logging.setErrorMessage(xhr.responseText);
    };
    xhr.onload = () => {
      const bookmarks = JSON.parse(xhr.responseText) as State[];
      logging.setModalMessage(null, msgId);
      callback(bookmarks);
    };
    xhr.send();
  }
}
