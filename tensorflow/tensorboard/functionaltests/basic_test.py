# Copyright 2016 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Basic TensorBoard functional tests using WebDriver."""

import os
import subprocess
import time
import unittest
from testing.web import browser
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support import wait
from selenium.webdriver.common import by


class BasicTest(unittest.TestCase):

  @classmethod
  def setUpClass(self):
    src_dir = os.environ["TEST_SRCDIR"]
    binary = os.path.join(src_dir,
                          "org_tensorflow/tensorflow/tensorboard/tensorboard")
    log_dir = os.path.join(
        src_dir,
        "org_tensorflow/tensorflow/tensorboard/functionaltests/testdata")
    self.process = subprocess.Popen(
        [binary, "--port", "8000", "--logdir", log_dir])

  @classmethod
  def tearDownClass(self):
    self.process.kill()
    self.process.wait()

  def setUp(self):
    self.driver = browser.Browser().new_session()
    self.driver.get("http://localhost:8000")
    self.wait = wait.WebDriverWait(self.driver, 2)

  def tearDown(self):
    try:
      self.driver.quit()
    finally:
      self.driver = None

  def testToolbarTitleDisplays(self):
    self.wait.until(
        expected_conditions.text_to_be_present_in_element((
            by.By.CLASS_NAME, "toolbar-title"), "TensorBoard"))

  def testChartAppearWhenHeadingClicks(self):
    self.wait.until(
        expected_conditions.element_to_be_clickable((
            by.By.CSS_SELECTOR,
            "tf-collapsable-pane.tf-collapsable-pane-0 button"))).click()
    self.wait.until(
        expected_conditions.visibility_of_element_located((
            by.By.CSS_SELECTOR,
            "tf-collapsable-pane.tf-collapsable-pane-0 vz-line-chart")))


if __name__ == "__main__":
  unittest.main()
