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

import {DataPoint, SpriteAndMetadataInfo} from './data';
import * as data_provider from './data-provider';

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
                expect(data.length).toBe(2);

                expect(data[0].vector).toEqual(new Float32Array(tensors[0]));
                expect(data[0].index).toEqual(0);
                expect(data[0].projections).toBeNull();

                expect(data[1].vector).toEqual(new Float32Array(tensors[1]));
                expect(data[1].index).toEqual(1);
                expect(data[1].projections).toBeNull();
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
                expect(spriteAndMetadataInfo.stats.length).toBe(2);
                expect(spriteAndMetadataInfo.stats[0].name)
                    .toBe(metadata[0][0]);
                expect(spriteAndMetadataInfo.stats[0].isNumeric).toBe(false);
                expect(spriteAndMetadataInfo.stats[0].tooManyUniqueValues)
                    .toBe(false);
                expect(spriteAndMetadataInfo.stats[1].name)
                    .toBe(metadata[0][1]);
                expect(spriteAndMetadataInfo.stats[1].isNumeric).toBe(true);
                expect(spriteAndMetadataInfo.stats[1].tooManyUniqueValues)
                    .toBe(false);

                expect(spriteAndMetadataInfo.pointsInfo.length).toBe(2);
                expect(spriteAndMetadataInfo.pointsInfo[0]['label'])
                    .toBe(metadata[1][0]);
                expect(spriteAndMetadataInfo.pointsInfo[0]['fakecol'])
                    .toBe(+metadata[1][1]);
                expect(spriteAndMetadataInfo.pointsInfo[1]['label'])
                    .toBe(metadata[2][0]);
                expect(spriteAndMetadataInfo.pointsInfo[1]['fakecol'])
                    .toBe(+metadata[2][1]);
                doneFn();
              });
        });
  });
});
