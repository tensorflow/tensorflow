/* Copyright 2015 Google Inc. All Rights Reserved.

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

module TF.Backend {
  interface ResolveReject {resolve: Function; reject: Function; }
  /*
  * Manages many fetch requests. Launches up to nSimultaneousRequests
  * simultaneously, and maintains a LIFO queue of requests to process when
  * more urls are requested than can be handled at once. The queue can be cleared.
  *
  * When a request is made, a Promise is returned which resolves with the parsed
  * JSON rseult from the reqest.
  */

  export class RequestCancellationError extends Error {
    public name = "RequestCancellationError";

    constructor(message?: string) {
      super(message);
    }
  }

  export class RequestManager {
    private _queue: ResolveReject[];
    private _nActiveRequests: number;
    private _nSimultaneousRequests: number;

    constructor(nSimultaneousRequests = 10) {
      this._queue = [];
      this._nActiveRequests = 0;
      this._nSimultaneousRequests = nSimultaneousRequests;
    }
    /* Gives a promise that loads assets from given url (respects queuing) */
    public request(url: string): Promise<any> {
      var promise = new Promise((resolve, reject) => {
        var resolver = {resolve: resolve, reject: reject};
        this._queue.push(resolver);
        this.launchRequests();
      }).then(() => {
        return this._promiseFromUrl(url);
      }).then((response) => {
        this._nActiveRequests--;
        this.launchRequests(); // since we may have queued responses to launch
        return response;
      });
      return promise;
    }

    public clearQueue() {
      while (this._queue.length > 0) {
        this._queue.pop().reject(new RequestCancellationError("Request cancelled by clearQueue"));
      }
    }

    /* Return number of currently pending requests */
    public activeRequests(): number {
      return this._nActiveRequests;
    }

    /* Return total number of outstanding requests (includes queue) */
    public outstandingRequests(): number {
      return this._nActiveRequests + this._queue.length;
    }

    private launchRequests() {
      while (this._nActiveRequests < this._nSimultaneousRequests && this._queue.length > 0) {
        this._nActiveRequests++;
        this._queue.pop().resolve();
      }
    }

    /* Actually get promise from url using XMLHttpRequest */
    protected _promiseFromUrl(url) {
      return new Promise((resolve, reject) => {
        var req = new XMLHttpRequest();
        req.open("GET", url);
        req.onload = function() {
          if (req.status === 200) {
            resolve(JSON.parse(req.responseText));
          } else {
            reject(Error("Status: " + req.status + ":" + req.statusText + " at url: " + url));
          }
        };
        req.onerror = function() {
          reject(Error("Network error"));
        };
        req.send();
      });
    }
  }
}
