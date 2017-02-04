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

module TF.Backend {
  interface ResolveReject { resolve: Function; reject: Function; }
  /**
   * Manages many fetch requests. Launches up to nSimultaneousRequests
   * simultaneously, and maintains a LIFO queue of requests to process when
   * more urls are requested than can be handled at once. The queue can be
   * cleared.
   *
   * When a request is made, a Promise is returned which resolves with the
   * parsed JSON result from the request.
   */
  export class RequestCancellationError extends Error {
    public name = 'RequestCancellationError';
  }

  export class RequestNetworkError extends Error {
    public name: string;
    public req: XMLHttpRequest;
    public url: string;

    constructor(req: XMLHttpRequest, url) {
      super();
      this.message = `RequestNetworkError: ${req.status} at ${url}`;
      this.name = 'RequestNetworkError';
      this.req = req;
      this.url = url;
    }
  }

  export class RequestManager {
    private _queue: ResolveReject[];
    private _maxRetries: number;
    private _nActiveRequests: number;
    private _nSimultaneousRequests: number;

    constructor(nSimultaneousRequests = 10, maxRetries = 3) {
      this._queue = [];
      this._nActiveRequests = 0;
      this._nSimultaneousRequests = nSimultaneousRequests;
      this._maxRetries = maxRetries;
    }

    /**
     * Gives a promise that loads assets from given url (respects queuing). If
     * postData is provided, this request will use POST, not GET. This is an
     * object mapping POST keys to string values.
     */
    public request(
        url: string, postData?: {[key: string]: string}): Promise<any> {
      var promise = new Promise((resolve, reject) => {
                      var resolver = {resolve: resolve, reject: reject};
                      this._queue.push(resolver);
                      this.launchRequests();
                    })
                        .then(() => {
                          return this.promiseWithRetries(
                              url, this._maxRetries, postData);
                        })
                        .then(
                            (response) => {
                              // Success - Let's free space for another active
                              // reqest, and launch it
                              this._nActiveRequests--;
                              this.launchRequests();
                              return response;
                            },
                            (rejection) => {
                              if (rejection.name === 'RequestNetworkError') {
                                // If we failed due to network error, we should
                                // decrement
                                // _nActiveRequests because this request was
                                // active
                                this._nActiveRequests--;
                                this.launchRequests();
                              }
                              return Promise.reject(rejection);
                            });
      return promise;
    }

    public clearQueue() {
      while (this._queue.length > 0) {
        this._queue.pop().reject(
            new RequestCancellationError('Request cancelled by clearQueue'));
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
      while (this._nActiveRequests < this._nSimultaneousRequests &&
             this._queue.length > 0) {
        this._nActiveRequests++;
        this._queue.pop().resolve();
      }
    }

    /**
     * Try to request a given URL using overwritable _promiseFromUrl method.
     * If the request fails for any reason, we will retry up to maxRetries
     * times. In practice, this will help us paper over transient network issues
     * like '502 Bad Gateway'.
     * By default, Chrome displays network errors in console, so
     * the user will be able to tell when the requests are failing. I think this
     * is a feature, if the request failures and retries are causing any
     * pain to users, they can see it and file issues.
     */
    private promiseWithRetries(
        url: string,
        maxRetries: number,
        postData?: {[key: string]: string}) {
      var success = (x) =>  x;
      var failure = (x) => {
        if (maxRetries > 0) {
          return this.promiseWithRetries(url, maxRetries - 1, postData);
        } else {
          return Promise.reject(x);
        }
      };
      return this._promiseFromUrl(url, postData).then(success, failure);
    }

    /* Actually get promise from url using XMLHttpRequest */
    protected _promiseFromUrl(url:string, postData?: {[key: string]: string}) {
      return new Promise((resolve, reject) => {
        let req = new XMLHttpRequest();
        req.open(postData ? 'POST' : 'GET', url);

        let formData;
        if (postData) {
          // We are to make a POST request.
          formData = new FormData();
          for (let postKey in postData) {
            if (postKey) {
              // The linter requires 'for in' loops to be filtered by an if
              // condition.
              formData.append(postKey, postData[postKey]);
            }
          }
        }
        req.onload = function() {
          if (req.status === 200) {
            resolve(JSON.parse(req.responseText));
          } else {
            reject(new RequestNetworkError(req, url));
          }
        };
        req.onerror = function() {
          reject(new RequestNetworkError(req, url));
        };
        req.send(formData);
      });
    }
  }
}
