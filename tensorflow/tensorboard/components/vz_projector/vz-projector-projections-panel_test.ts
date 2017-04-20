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
import {ProjectionsPanel} from './vz-projector-projections-panel';

const assert = chai.assert;

describe('restoreUIFromBookmark', () => {
  let projectionsPanel: ProjectionsPanel;
  beforeEach(() => {
    projectionsPanel = document.createElement(ProjectionsPanel.prototype.is) as
        ProjectionsPanel;

    // Set up some of the UI so the elements are found in the production code.
    const tsnePerplexityContainer = document.createElement('div');
    tsnePerplexityContainer.className = 'tsne-perplexity';
    const tsnePerplexity = document.createElement('span');
    tsnePerplexityContainer.appendChild(tsnePerplexity);
    projectionsPanel.appendChild(tsnePerplexityContainer);

    const tsneLearningRateContainer = document.createElement('div');
    tsneLearningRateContainer.className = 'tsne-learning-rate';
    const tsneLearningRate = document.createElement('span');
    tsneLearningRateContainer.appendChild(tsneLearningRate);
    projectionsPanel.appendChild(tsneLearningRateContainer);
  });

  it('sets the pcaX/Y properties when setting 2D component values', () => {
    spyOn(projectionsPanel, 'setZDropdownEnabled');

    const s = new State();
    s.pcaComponentDimensions = [0, 1];
    projectionsPanel.restoreUIFromBookmark(s);

    assert.equal(0, projectionsPanel.pcaX);
    assert.equal(1, projectionsPanel.pcaY);

    expect(projectionsPanel.setZDropdownEnabled).toHaveBeenCalledWith(false);
  });

  it('sets the pcaX/Y properties when setting 3D component values', () => {
    spyOn(projectionsPanel, 'setZDropdownEnabled');

    const s = new State();
    s.pcaComponentDimensions = [0, 1, 2];
    projectionsPanel.restoreUIFromBookmark(s);

    assert.equal(0, projectionsPanel.pcaX);
    assert.equal(1, projectionsPanel.pcaY);
    assert.equal(2, projectionsPanel.pcaZ);

    expect(projectionsPanel.setZDropdownEnabled).toHaveBeenCalledWith(true);
  });
});

describe('populateBookmarkFromUI', () => {
  let projectionsPanel: ProjectionsPanel;

  beforeEach(() => {
    projectionsPanel = document.createElement(ProjectionsPanel.prototype.is) as
        ProjectionsPanel;

    // Set up some of the UI so the elements are found in the production code.
    const tsnePerplexityContainer = document.createElement('div');
    tsnePerplexityContainer.className = 'tsne-perplexity';
    const tsnePerplexity = document.createElement('span');
    tsnePerplexityContainer.appendChild(tsnePerplexity);
    projectionsPanel.appendChild(tsnePerplexityContainer);

    const tsneLearningRateContainer = document.createElement('div');
    tsneLearningRateContainer.className = 'tsne-learning-rate';
    const tsneLearningRate = document.createElement('span');
    tsneLearningRateContainer.appendChild(tsneLearningRate);
    projectionsPanel.appendChild(tsneLearningRateContainer);
  });

  it('gets the PCA component UI values from a 2D PCA projection', () => {
    projectionsPanel.pcaX = 0;
    projectionsPanel.pcaY = 1;
    projectionsPanel.pcaIs3d = false;

    const s = new State();
    projectionsPanel.populateBookmarkFromUI(s);
    assert.deepEqual([0, 1], s.pcaComponentDimensions);
  });

  it('gets the PCA component UI values from a 3D PCA projection', () => {
    projectionsPanel.pcaX = 0;
    projectionsPanel.pcaY = 1;
    projectionsPanel.pcaZ = 2;
    projectionsPanel.pcaIs3d = true;

    const s = new State();
    projectionsPanel.populateBookmarkFromUI(s);
    assert.deepEqual([0, 1, 2], s.pcaComponentDimensions);
  });
});
