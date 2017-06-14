/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

import {demoify, queryEncoder} from './urlPathHelpers'

export type RunTagUrlFn = (tag: string, run: string) => string;

export interface Router {
  logdir: () => string;
  runs: () => string;
  isDemoMode: () => boolean;
  textRuns: () => string;
  text: RunTagUrlFn;
  healthPills: () => string;
  pluginRoute: (pluginName: string, route: string) => string;
  pluginRunTagRoute: (pluginName: string, route: string) => RunTagUrlFn;
}
;

/**
 * Create a router for communicating with the TensorBoard backend. You
 * can pass this to `setRouter` to make it the global router.
 *
 * @param dataDir {string} The base prefix for finding data on server.
 * @param demoMode {boolean} Whether to modify urls for filesystem demo usage.
 */
export function createRouter(dataDir = 'data', demoMode = false): Router {
  var clean = demoMode ? demoify : (x) => x;
  if (dataDir[dataDir.length - 1] === '/') {
    dataDir = dataDir.slice(0, dataDir.length - 1);
  }
  function standardRoute(route: string, demoExtension = '.json'):
      ((tag: string, run: string) => string) {
    return function(tag: string, run: string): string {
      var url =
          dataDir + '/' + route + clean(queryEncoder({tag: tag, run: run}));
      if (demoMode) {
        url += demoExtension;
      }
      return url;
    };
  }
  function pluginRoute(pluginName: string, route: string): string {
    return `${dataDir}/plugin/${pluginName}${route}`;
  }
  function pluginRunTagRoute(pluginName: string, route: string):
      ((tag: string, run: string) => string) {
    const base = pluginRoute(pluginName, route);
    return (tag, run) => base + clean(queryEncoder({tag, run}));
  }
  return {
    logdir: () => dataDir + '/logdir',
    runs: () => dataDir + '/runs' + (demoMode ? '.json' : ''),
    isDemoMode: () => demoMode,
    healthPills: () => dataDir + '/plugin/debugger/health_pills',
    textRuns: () => dataDir + '/plugin/text/runs' + (demoMode ? '.json' : ''),
    text: standardRoute('plugin/text/text'),
    pluginRoute,
    pluginRunTagRoute,
  };
};

let _router: Router = createRouter();

/**
 * @return {Router} the global router
 */
export function getRouter(): Router {
  return _router;
}

/**
 * Set the global router, to be returned by future calls to `getRouter`.
 * You may wish to invoke this if you are running a demo server with a
 * custom path prefix, or if you have customized the TensorBoard backend
 * to use a different path.
 *
 * @param {Router} router the new global router
 */
export function setRouter(router: Router): void {
  if (router == null) {
    throw new Error('Router required, but got: ' + router);
  }
  _router = router;
}
