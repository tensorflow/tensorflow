/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the 'License');
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an 'AS IS' BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import * as backend_backend from '../../tf-backend/backend';
import {createRouter, setRouter} from '../../tf-backend/router';

// TODO(dandelion): Fix me.
declare function fixture(id: string): any;
declare function stub(x, y: any): void;

describe('audio dashboard tests', () => {
  let audioDash;
  let reloadCount = 0;
  beforeEach(() => {
    audioDash = fixture('testElementFixture');
    const router = createRouter('/data', true);
    setRouter(router);
    const backend = new backend_backend.Backend();
    audioDash.backend = backend;
    stub('tf-audio-loader', {
      reload: () => { reloadCount++; },
    });
  });

  it('calling reload on dashboard reloads the audio-loaders', (done) => {
    audioDash.backendReload().then(() => {
      reloadCount = 0;
      const loaders =
          [].slice.call(audioDash.getElementsByTagName('tf-audio-loader'));
      audioDash.frontendReload();
      setTimeout(() => {
        chai.assert.isTrue(reloadCount >= 2);
        done();
      });
    });
  });
});
