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
import {State} from './data';
import {Projector} from './vz-projector';
// tslint:disable-next-line:no-unused-variable
import {PolymerElement, PolymerHTMLElement} from './vz-projector-util';

// tslint:disable-next-line
export let BookmarkPanelPolymer = PolymerElement({
  is: 'vz-projector-bookmark-panel',
  properties: {savedStates: Object, selectedState: Number}
});

export class BookmarkPanel extends BookmarkPanelPolymer {
  private projector: Projector;

  // A list containing all of the saved states.
  private savedStates: State[];
  private selectedState: number;

  private dom: d3.Selection<any>;

  ready() {
    this.dom = d3.select(this);
    this.savedStates = [];
    this.setupUploadButton();
  }

  initialize(projector: Projector) {
    this.projector = projector;
  }

  /** Handles a click on show bookmarks tray button. */
  _expandMore() {
    this.$.panel.toggle();
    this.dom.select('#expand-more').style('display', 'none');
    this.dom.select('#expand-less').style('display', '');
  }

  /** Handles a click on hide bookmarks tray button. */
  _expandLess() {
    this.$.panel.toggle();
    this.dom.select('#expand-more').style('display', '');
    this.dom.select('#expand-less').style('display', 'none');
  }

  /** Handles a click on the add bookmark button. */
  _addBookmark() {
    let currentState = this.projector.getCurrentState();
    currentState.label = 'State ' + this.savedStates.length;
    currentState.isSelected = true;

    this.selectedState = this.savedStates.length;

    for (let i = 0; i < this.savedStates.length; i++) {
      this.savedStates[i].isSelected = false;
      // We have to call notifyPath so that polymer knows this element was
      // updated.
      this.notifyPath('savedStates.' + i + '.isSelected', false, false);
    }

    this.push('savedStates', currentState);
  }

  /** Handles a click on the download bookmarks button. */
  _downloadFile() {
    let serializedState = this.serializeAllSavedStates();
    let blob = new Blob([serializedState], {type: 'text/plain'});
    let textFile = window.URL.createObjectURL(blob);

    // Force a download.
    let a = document.createElement('a');
    document.body.appendChild(a);
    a.style.display = 'none';
    a.href = textFile;
    a.download = 'state';
    a.click();

    document.body.removeChild(a);
    window.URL.revokeObjectURL(textFile);
  }

  /** Handles a click on the upload bookmarks button. */
  _uploadFile() {
    let fileInput = this.dom.select('#state-file');
    (fileInput.node() as HTMLInputElement).click();
  }

  private setupUploadButton() {
    // Show and setup the load view button.
    let fileInput = this.dom.select('#state-file');
    fileInput.on('change', function() {
      let file: File = (d3.event as any).target.files[0];
      // Clear out the value of the file chooser. This ensures that if the user
      // selects the same file, we'll re-read it.
      (d3.event as any).target.value = '';
      let fileReader = new FileReader();
      fileReader.onload = function(evt) {
        let str: string = (evt.target as any).result;

        let savedStates = JSON.parse(str);
        for (let i = 0; i < savedStates.length; i++) {
          savedStates[i].isSelected = false;
          this.push('savedStates', savedStates[i]);
        }
      }.bind(this);
      fileReader.readAsText(file);
    }.bind(this));
  }

  /** Deselects any selected state selection. */
  clearStateSelection() {
    for (let i = 0; i < this.savedStates.length; i++) {
      if (this.savedStates[i].isSelected) {
        this.savedStates[i].isSelected = false;
        this.notifyPath('savedStates.' + i + '.isSelected', false, false);
        return;
      }
    }
  }

  /** Handles a radio button click on a saved state. */
  _radioButtonHandler(evt: Event) {
    let index =
        +(evt.target as Element).parentElement.getAttribute('data-index');

    for (let i = 0; i < this.savedStates.length; i++) {
      if (this.savedStates[i].isSelected) {
        this.savedStates[i].isSelected = false;
        this.notifyPath('savedStates.' + i + '.isSelected', false, false);
      } else if (index === i) {
        this.savedStates[i].isSelected = true;
        this.notifyPath('savedStates.' + i + '.isSelected', true, false);

        // Update the world to this state.
        this.projector.loadState(this.savedStates[i]);
      }
    }
  }

  /**
   * Crawls up the DOM to find an ancestor with a data-index attribute. This is
   * used to match events to their bookmark index.
   */
  _getParentDataIndex(evt: Event) {
    for (let i = 0; i < (evt as any).path.length; i++) {
      let dataIndex = (evt as any).path[i].getAttribute('data-index');
      if (dataIndex != null) {
        return +dataIndex;
      }
    }
    return -1;
  }

  /** Handles a clear button click on a bookmark. */
  _clearButtonHandler(evt: Event) {
    let index = this._getParentDataIndex(evt);
    this.splice('savedStates', index, 1);
  }

  /** Handles a label change event on a bookmark. */
  _labelChange(evt: Event) {
    let index = this._getParentDataIndex(evt);
    this.savedStates[index].label = (evt.target as any).value;
  }

  /**
   * Used to determine whether to select the radio button for a given bookmark.
   */
  _isSelectedState(index: number) {
    return index === this.selectedState;
  }
  _isNotSelectedState(index: number) {
    return index !== this.selectedState;
  }

  /**
   * Gets all of the saved states as a serialized string.
   */
  serializeAllSavedStates(): string {
    return JSON.stringify(this.savedStates);
  }

  /**
   * Loads all of the serialized states and shows them in the list of
   * viewable states.
   */
  loadSavedStates(serializedStates: string) {
    this.savedStates = JSON.parse(serializedStates);
  }
}
document.registerElement(BookmarkPanel.prototype.is, BookmarkPanel);
