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

import {DataPoint, SpriteAndMetadataInfo} from '../data';
import * as data_provider from '../data-provider';

/**
 * Converts a string to an ArrayBuffer.
 */
function stringToArrayBuffer(str: string): Promise<ArrayBuffer> {
  return new Promise<ArrayBuffer>((resolve, reject) => {
    let blob = new Blob([str]);
    let file = new FileReader();
    file.onload = (e: any) => {
      resolve(e.target.result);
    };
    file.readAsArrayBuffer(blob);
  });
}

/**
 * Converts an data array to TSV format.
 */
function dataToTsv(data: string[][]|number[][]) {
  let lines = [];
  for (let i = 0; i < data.length; i++) {
    lines.push(data[i].join('\t'));
  }
  return lines.join('\n');
}

describe('parse tensors', () => {
  it('parseTensors', (doneFn) => {
    let tensors = [[1.0, 2.0], [2.0, 3.0]];
    stringToArrayBuffer(dataToTsv(tensors))
        .then((tensorsArrayBuffer: ArrayBuffer) => {
          data_provider.parseTensors(tensorsArrayBuffer)
              .then((data: DataPoint[]) => {
                assert.equal(2, data.length);

                assert.deepEqual(new Float32Array(tensors[0]), data[0].vector);
                assert.equal(0, data[0].index);
                assert.isNull(data[0].projections);

                assert.deepEqual(new Float32Array(tensors[1]), data[1].vector);
                assert.equal(1, data[1].index);
                assert.isNull(data[1].projections);
                doneFn();
              });
        });
  });
  it('parseMetadata', (doneFn) => {
    let metadata = [['label', 'fakecol'], ['Г', '0'], ['label1', '1']];

    stringToArrayBuffer(dataToTsv(metadata))
        .then((metadataArrayBuffer: ArrayBuffer) => {
          data_provider.parseMetadata(metadataArrayBuffer)
              .then((spriteAndMetadataInfo: SpriteAndMetadataInfo) => {
                assert.equal(2, spriteAndMetadataInfo.stats.length);
                assert.equal(metadata[0][0],
                             spriteAndMetadataInfo.stats[0].name);
                assert.isFalse(spriteAndMetadataInfo.stats[0].isNumeric);
                assert.isFalse(
                    spriteAndMetadataInfo.stats[0].tooManyUniqueValues);
                assert.equal(metadata[0][1],
                             spriteAndMetadataInfo.stats[1].name);
                assert.isTrue(spriteAndMetadataInfo.stats[1].isNumeric);
                assert.isFalse(
                    spriteAndMetadataInfo.stats[1].tooManyUniqueValues);

                assert.equal(2, spriteAndMetadataInfo.pointsInfo.length);
                assert.equal(metadata[1][0],
                             spriteAndMetadataInfo.pointsInfo[0]['label']);
                assert.equal(+metadata[1][1],
                             spriteAndMetadataInfo.pointsInfo[0]['fakecol']);
                assert.equal(metadata[2][0],
                             spriteAndMetadataInfo.pointsInfo[1]['label']);
                assert.equal(+metadata[2][1],
                             spriteAndMetadataInfo.pointsInfo[1]['fakecol']);
                doneFn();
              });
        });
  });
});
