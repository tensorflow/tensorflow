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

import {RequestManager, RequestNetworkError} from '../requestManager';

interface MockRequest {
  resolve: Function;
  reject: Function;
  id: number;
  url: string;
}

class MockedRequestManager extends RequestManager {
  private resolvers: Function[];
  private rejectors: Function[];
  public requestsDispatched: number;
  constructor(maxRequests = 10, maxRetries = 3) {
    super(maxRequests, maxRetries);
    this.resolvers = [];
    this.rejectors = [];
    this.requestsDispatched = 0;
  }
  protected _promiseFromUrl(url) {
    return new Promise((resolve, reject) => {
      const mockJSON = {
        ok: true,
        json() {
          return url;
        },
        url,
        status: 200,
      };
      const mockFailedRequest: any = {
        ok: false,
        url,
        status: 502,
      };
      const mockFailure = new RequestNetworkError(mockFailedRequest, url);
      this.resolvers.push(() => {
        resolve(mockJSON);
      });
      this.rejectors.push(() => {
        reject(mockFailure);
      });
      this.requestsDispatched++;
    });
  }
  public resolveFakeRequest() {
    this.resolvers.pop()();
  }
  public rejectFakeRequest() {
    this.rejectors.pop()();
  }
  public dispatchAndResolve() {
    // Wait for at least one request to be dispatched, then resolve it.
    this.waitForDispatch(1).then(() => this.resolveFakeRequest());
  }
  public waitForDispatch(num) {
    return waitForCondition(() => {
      return this.requestsDispatched >= num;
    });
  }
}

/** Create a promise that returns when *check* returns true.
 * May cause a test timeout if check never becomes true.
 */

function waitForCondition(check: () => boolean): Promise<any> {
  return new Promise((resolve, reject) => {
    const go = () => {
      if (check()) {
        resolve();
      }
      setTimeout(go, 2);
    };
    go();
  });
}

describe('backend', () => {
  describe('request manager', () => {
    it('request loads JSON properly', (done) => {
      const rm = new RequestManager();
      const promise = rm.request('data/example.json');
      promise.then(
          (response) => {
            chai.assert.deepEqual(response, {foo: 3, bar: 'zoidberg'});
            done();
          },
          (reject) => {
            throw new Error(reject);
          });
    });

    it('rejects on bad url', (done) => {
      const rm = new RequestManager(5, 0);
      const badUrl = '_bad_url_which_doesnt_exist.json';
      const promise = rm.request(badUrl);
      promise.then(
          (success) => {
            done(new Error('the promise should have rejected'));
          },
          (reject: RequestNetworkError) => {
            chai.assert.include(reject.message, '404');
            chai.assert.include(reject.message, badUrl);
            chai.assert.equal(reject.req.status, 404);
            done();
          });
    });

    it('can retry if requests fail', (done) => {
      const rm = new MockedRequestManager(3, 5);
      const r = rm.request('foo');
      rm.waitForDispatch(1)
          .then(() => {
            rm.rejectFakeRequest();
            return rm.waitForDispatch(2);
          })
          .then(() => rm.resolveFakeRequest());
      r.then((success) => done());
    });

    it('retries at most maxRetries times', (done) => {
      const MAX_RETRIES = 2;
      const rm = new MockedRequestManager(3, MAX_RETRIES);
      const r = rm.request('foo');
      rm.waitForDispatch(1)
          .then(() => {
            rm.rejectFakeRequest();
            return rm.waitForDispatch(2);
          })
          .then(() => {
            rm.rejectFakeRequest();
            return rm.waitForDispatch(3);
          })
          .then(() => {
            rm.rejectFakeRequest();
          });

      r.then(
          (success) => done(new Error('The request should have failed')),
          (failure) => done());
    });

    it('requestManager only sends maxRequests requests at a time', (done) => {
      const rm = new MockedRequestManager(3);
      const r0 = rm.request('1');
      const r1 = rm.request('2');
      const r2 = rm.request('3');
      const r3 = rm.request('4');
      chai.assert.equal(rm.activeRequests(), 3, 'three requests are active');
      chai.assert.equal(
          rm.outstandingRequests(), 4, 'four requests are pending');
      rm.waitForDispatch(3)
          .then(() => {
            chai.assert.equal(
                rm.activeRequests(), 3, 'three requests are still active (1)');
            chai.assert.equal(
                rm.requestsDispatched, 3, 'three requests were dispatched');
            rm.resolveFakeRequest();
            return rm.waitForDispatch(4);
          })
          .then(() => {
            chai.assert.equal(
                rm.activeRequests(), 3, 'three requests are still active (2)');
            chai.assert.equal(
                rm.requestsDispatched, 4, 'four requests were dispatched');
            chai.assert.equal(
                rm.outstandingRequests(), 3, 'three requests are pending');
            rm.resolveFakeRequest();
            rm.resolveFakeRequest();
            rm.resolveFakeRequest();
            return r3;
          })
          .then(() => {
            chai.assert.equal(rm.activeRequests(), 0, 'all requests finished');
            chai.assert.equal(
                rm.outstandingRequests(), 0, 'no requests pending');
            done();
          });
    });

    it('queue continues after failures', (done) => {
      const rm = new MockedRequestManager(1, 0);
      const r0 = rm.request('1');
      const r1 = rm.request('2');
      rm.waitForDispatch(1).then(() => {
        rm.rejectFakeRequest();
      });

      r0.then(
            (success) => done(new Error('r0 should have failed')),
            (failure) => 'unused_argument')
          .then(() => rm.resolveFakeRequest());

      // When the first request rejects, it should decrement nActiveRequests
      // and then launch remaining requests in queue (i.e. this one)
      r1.then((success) => done(), (failure) => done(new Error(failure)));
    });

    it('queue is LIFO', (done) => {
      /* This test is a bit tricky.
       * We want to verify that the RequestManager queue has LIFO semantics.
       * So we construct three requests off the bat: A, B, C.
       * So LIFO semantics ensure these will resolve in order A, C, B.
       * (Because the A request launches immediately when we create it, it's
       * not in queue)
       * Then after resolving A, C moves out of queue, and we create X.
       * So expected final order is A, C, X, B.
       * We verify this with an external var that counts how many requests were
       * resolved.
       */
      const rm = new MockedRequestManager(1);
      let nResolved = 0;
      function assertResolutionOrder(expectedSpotInSequence) {
        return () => {
          nResolved++;
          chai.assert.equal(expectedSpotInSequence, nResolved);
        };
      }

      function launchThirdRequest() {
        rm.request('started late but goes third')
            .then(assertResolutionOrder(3))
            .then(() => rm.dispatchAndResolve());
      }

      rm.request('first')
          .then(
              assertResolutionOrder(1))  // Assert that this one resolved first
          .then(launchThirdRequest)
          .then(() => rm.dispatchAndResolve());  // then trigger the next one

      rm.request('this one goes fourth')  // created second, will go last
          .then(assertResolutionOrder(
              4))       // assert it was the fourth to get resolved
          .then(done);  // finish the test

      rm.request('second')
          .then(assertResolutionOrder(2))
          .then(() => rm.dispatchAndResolve());

      rm.dispatchAndResolve();
    });

    it('requestManager can clear queue', (done) => {
      const rm = new MockedRequestManager(1);
      let requestsResolved = 0;
      let requestsRejected = 0;
      const success = () => requestsResolved++;
      const failure = (err) => {
        chai.assert.equal(err.name, 'RequestCancellationError');
        requestsRejected++;
      };
      const finishTheTest = () => {
        chai.assert.equal(rm.activeRequests(), 0, 'no requests still active');
        chai.assert.equal(
            rm.requestsDispatched, 1, 'only one req was ever dispatched');
        chai.assert.equal(rm.outstandingRequests(), 0, 'no pending requests');
        chai.assert.equal(requestsResolved, 1, 'one request got resolved');
        chai.assert.equal(
            requestsRejected, 4, 'four were cancelled and threw errors');
        done();
      };
      rm.request('0').then(success, failure).then(finishTheTest);
      rm.request('1').then(success, failure);
      rm.request('2').then(success, failure);
      rm.request('3').then(success, failure);
      rm.request('4').then(success, failure);
      chai.assert.equal(rm.activeRequests(), 1, 'one req is active');
      rm.waitForDispatch(1).then(() => {
        chai.assert.equal(rm.activeRequests(), 1, 'one req is active');
        chai.assert.equal(rm.requestsDispatched, 1, 'one req was dispatched');
        chai.assert.equal(rm.outstandingRequests(), 5, 'five reqs outstanding');
        rm.clearQueue();
        rm.resolveFakeRequest();
        // resolving the first request triggers finishTheTest
      });
    });
  });
});
