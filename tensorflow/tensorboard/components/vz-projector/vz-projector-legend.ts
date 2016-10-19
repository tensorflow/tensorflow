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

// tslint:disable-next-line:no-unused-variable
import {PolymerElement, PolymerHTMLElement} from './vz-projector-util';

// tslint:disable-next-line
export let LegendPolymer = PolymerElement({
  is: 'vz-projector-legend',
  properties: {renderInfo: {type: Object, observer: '_renderInfoChanged'}}
});

export interface ColorLegendRenderInfo {
  // To be used for categorical map.
  items: ColorLegendItem[];
  // To be used for gradient map.
  thresholds: ColorLegendThreshold[];
}

/** An item in the categorical color legend. */
export interface ColorLegendItem {
  color: string;
  label: string;
  count: number;
}

/** An item in the gradient color legend. */
export interface ColorLegendThreshold {
  color: string;
  value: number;
}

export class Legend extends LegendPolymer {
  renderInfo: ColorLegendRenderInfo;
  dom: d3.Selection<HTMLElement>;

  ready() {
    this.dom = d3.select(this);
  }

  _renderInfoChanged() {
    if (this.renderInfo == null) {
      return;
    }
    if (this.renderInfo.thresholds) {
      // <linearGradient> is under dom-if so we should wait for it to be
      // inserted in the dom tree using async().
      this.async(() => this.setupLinearGradient());
    }
  }

  _getLastThreshold(): number {
    if (this.renderInfo == null || this.renderInfo.thresholds == null) {
      return;
    }
    return this.renderInfo.thresholds[this.renderInfo.thresholds.length - 1]
        .value;
  }

  private getOffset(value: number): string {
    let min = this.renderInfo.thresholds[0].value;
    let max =
        this.renderInfo.thresholds[this.renderInfo.thresholds.length - 1].value;
    return (100 * (value - min) / (max - min)).toFixed(2) + '%';
  }

  private setupLinearGradient() {
    let linearGradient = this.dom.select('#gradient');

    let width =
        (this.dom.select('svg.gradient').node() as SVGElement).clientWidth;

    // Set the svg <rect> to be the width of its <svg> parent.
    this.dom.select('svg.gradient rect').attr('width', width);

    // Remove all <stop> children from before.
    linearGradient.selectAll('*').remove();

    // Add a <stop> child in <linearGradient> for each gradient threshold.
    this.renderInfo.thresholds.forEach(t => {
      linearGradient.append('stop')
          .attr('offset', this.getOffset(t.value))
          .attr('stop-color', t.color);
    });
  }
}

document.registerElement(Legend.prototype.is, Legend);
