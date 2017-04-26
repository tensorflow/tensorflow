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

import {PointMetadata} from './data';
// tslint:disable-next-line:no-unused-variable
import {PolymerElement, PolymerHTMLElement} from './vz-projector-util';

// tslint:disable-next-line
export let MetadataCardPolymer = PolymerElement({
  is: 'vz-projector-metadata-card',
  properties: {
    hasMetadata: {type: Boolean, value: false},
    metadata: {type: Array},
    label: String
  }
});

export class MetadataCard extends MetadataCardPolymer {
  hasMetadata: boolean;
  metadata: Array<{key: string, value: string}>;
  label: string;

  private labelOption: string;
  private pointMetadata: PointMetadata;

  private expandLessButton: HTMLButtonElement;
  private expandMoreButton: HTMLButtonElement;

  ready() {
    this.expandLessButton =
        this.querySelector('#expand-less') as HTMLButtonElement;
    this.expandMoreButton =
        this.querySelector('#expand-more') as HTMLButtonElement;
  }
  /** Handles a click on the expand more icon. */
  _expandMore() {
    (this.$$('#metadata-container') as any).toggle();

    this.expandMoreButton.style.display = 'none';
    this.expandLessButton.style.display = '';
  }

  /** Handles a click on the expand less icon. */
  _expandLess() {
    (this.$$('#metadata-container') as any).toggle();
    this.expandMoreButton.style.display = '';
    this.expandLessButton.style.display = 'none';
  }

  updateMetadata(pointMetadata?: PointMetadata) {
    this.pointMetadata = pointMetadata;
    this.hasMetadata = (pointMetadata != null);

    if (pointMetadata) {
      let metadata = [];
      for (let metadataKey in pointMetadata) {
        if (!pointMetadata.hasOwnProperty(metadataKey)) {
          continue;
        }
        metadata.push({key: metadataKey, value: pointMetadata[metadataKey]});
      }

      this.metadata = metadata;
      this.label = '' + this.pointMetadata[this.labelOption];
    }
  }

  setLabelOption(labelOption: string) {
    this.labelOption = labelOption;
    if (this.pointMetadata) {
      this.label = '' + this.pointMetadata[this.labelOption];
    }
  }
}

document.registerElement(MetadataCard.prototype.is, MetadataCard);
