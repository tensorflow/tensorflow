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
import {ProjectionType} from './data';

export class AnalyticsLogger {
  private eventLogging: boolean;
  private pageViewLogging: boolean;

  /**
   * Constructs an event logger using Google Analytics. It assumes there is a
   * Google Analytics script added to the page elsewhere. If there is no such
   * script, the logger acts as a no-op.
   *
   * @param pageViewLogging Whether to log page views.
   * @param eventLogging Whether to log user interaction.
   */
  constructor(pageViewLogging: boolean, eventLogging: boolean) {
    if (typeof ga === 'undefined' || ga == null) {
      this.eventLogging = false;
      this.pageViewLogging = false;
      return;
    }
    this.eventLogging = eventLogging;
    this.pageViewLogging = pageViewLogging;
  }

  logPageView(pageTitle: string) {
    if (this.pageViewLogging) {
      // Always send a page view.
      ga('send', {hitType: 'pageview', page: `/v/${pageTitle}`});
    }
  }

  logProjectionChanged(projection: ProjectionType) {
    if (this.eventLogging) {
      ga('send', {
        hitType: 'event',
        eventCategory: 'Projection',
        eventAction: 'click',
        eventLabel: projection
      });
    }
  }

  logWebGLDisabled() {
    if (this.eventLogging) {
      ga('send', {
        hitType: 'event',
        eventCategory: 'Error',
        eventAction: 'PageLoad',
        eventLabel: 'WebGL_disabled'
      });
    }
  }
}
