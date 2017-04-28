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
  private textChangedListeners: InputChangedListener[];
  private paperInput: HTMLInputElement;
  private inRegexModeButton: HTMLButtonElement;
  private inRegexMode: boolean;

  /** Message that will be displayed at the bottom of the input control. */
  message: string;

  /** Subscribe to be called everytime the input changes. */
  registerInputChangedListener(listener: InputChangedListener) {
    this.textChangedListeners.push(listener);
  }

  ready() {
    this.inRegexMode = false;
    this.textChangedListeners = [];
    this.paperInput = this.querySelector('paper-input') as HTMLInputElement;
    this.inRegexModeButton =
        this.querySelector('paper-button') as HTMLButtonElement;
    this.paperInput.setAttribute('error-message', 'Invalid regex');

    this.paperInput.addEventListener('input', () => {
      this.onTextChanged();
    });

    this.paperInput.addEventListener('keydown', event => {
      event.stopPropagation();
    });

    this.inRegexModeButton.addEventListener(
        'click', () => this.onClickRegexModeButton());
    this.updateRegexModeDisplaySlashes();
    this.onTextChanged();
  }

  private onClickRegexModeButton() {
    this.inRegexMode = (this.inRegexModeButton as any).active;
    this.updateRegexModeDisplaySlashes();
    this.onTextChanged();
  }

  private notifyInputChanged(value: string, inRegexMode: boolean) {
    this.textChangedListeners.forEach(l => l(value, inRegexMode));
  }

  private onTextChanged() {
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

  private updateRegexModeDisplaySlashes() {
    const slashes = this.paperInput.querySelectorAll('.slash');
    const display = this.inRegexMode ? '' : 'none';

    for (let i = 0; i < slashes.length; i++) {
      (slashes[i] as HTMLDivElement).style.display = display;
    }
  }

  getValue(): string {
    return this.paperInput.value;
  }

  getInRegexMode(): boolean {
    return this.inRegexMode;
  }

  set(value: string, inRegexMode: boolean) {
    (this.inRegexModeButton as any).active = inRegexMode;
    this.paperInput.value = value;
    this.onClickRegexModeButton();
  }
}

document.registerElement(ProjectorInput.prototype.is, ProjectorInput);
