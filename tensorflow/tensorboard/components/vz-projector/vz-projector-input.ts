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
export let PolymerClass = PolymerElement(
    {is: 'vz-projector-input', properties: {label: String, message: String}});

export interface InputChangedListener {
  (value: string, inRegexMode: boolean): void;
}

/** Input control with custom capabilities (e.g. regex). */
export class ProjectorInput extends PolymerClass {
  private dom: d3.Selection<HTMLElement>;
  private inputChangedListeners: InputChangedListener[];
  private paperInput: HTMLInputElement;
  private inRegexMode: boolean;

  /** Message that will be displayed at the bottom of the input control. */
  message: string;
  /** Placeholder text for the input control. */
  label: string;

  /** Subscribe to be called everytime the input changes. */
  onInputChanged(listener: InputChangedListener, callImmediately = true) {
    this.inputChangedListeners.push(listener);
    if (callImmediately) {
      listener(this.paperInput.value, this.inRegexMode);
    }
  }

  ready() {
    this.inRegexMode = false;
    this.inputChangedListeners = [];
    this.dom = d3.select(this);
    this.paperInput = this.querySelector('paper-input') as HTMLInputElement;
    let paperButton = this.querySelector('paper-button') as HTMLButtonElement;
    this.paperInput.setAttribute('error-message', 'Invalid regex');

    this.paperInput.addEventListener('input', () => {
      this.inputChanged();
    });

    this.paperInput.addEventListener('keydown', event => {
      event.stopPropagation();
    });

    // Setup the regex mode button.
    paperButton.addEventListener('click', () => {
      this.inRegexMode = (paperButton as any).active;
      this.showHideSlashes();
      this.inputChanged();
    });
    this.showHideSlashes();
    this.inputChanged();
  }

  private notifyInputChanged(value: string, inRegexMode: boolean) {
    this.inputChangedListeners.forEach(l => l(value, inRegexMode));
  }

  private inputChanged() {
    try {
      if (this.inRegexMode) {
        new RegExp(this.paperInput.value);
      }
    } catch (invalidRegexException) {
      this.paperInput.setAttribute('invalid', 'true');
      this.message = '';
      this.notifyInputChanged(null, true);
      return;
    }
    this.paperInput.removeAttribute('invalid');
    this.notifyInputChanged(this.paperInput.value, this.inRegexMode);
  }

  private showHideSlashes() {
    d3.select(this.paperInput)
        .selectAll('.slash')
        .style('display', this.inRegexMode ? null : 'none');
  }
}

document.registerElement(ProjectorInput.prototype.is, ProjectorInput);
