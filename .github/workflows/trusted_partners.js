/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

/** Get the domain of the user's email

  @param {!object}
    github enables querying for PR and also create issue using rest endpoint
    username has the login username of the Pull Request Event
  @return {string} Return the domain name of the user's email. Empty string if not found
*/

const get_email_domain = async ({github, username}) => {
  const user = await github.rest.users.getByUsername({
    username
  });
  if (user.status >= 400) {
    console.log(user);
    throw `Error Getting user data for ${username}`;
  }
  const email = user.data.email;
  let domain = "";
  if (email && email.lastIndexOf("@") != -1)
    domain = email.substring(email.lastIndexOf("@") +1);
  console.log(domain);
  return domain;
};

/** For trusted parters like Intel, we want to auto-run tests and mark the PR as ready to pull
    This allows us to reduce the delay to external partners
    Add Labels - kokoro:force-run, ready to pull
    The PR is also assigned to specific teams to fast track review
    Additional reviewers can be added manually based on PR contents
  @param {!object}
    github enables querying for PR and also create issue using rest endpoint
    context has the commit message details in the payload
  @return {string} Returns the message with labels attached and assignees added
*/
const filter_action = async ({github, context, domain}) => {
  const labels = ['kokoro:force-run', 'ready to pull'];

  let assignees = [];
  const title = context.payload.pull_request && context.payload.pull_request.title;
  const lowercased_title = (title || '').toLowerCase();
  const onednn_assignees = ['penpornk'];
  if (lowercased_title.includes('onednn')) assignees = onednn_assignees;
  const intel_windows_assignees = ['nitins17', 'learning-to-play'];
  if (lowercased_title.includes('intel') &&
      lowercased_title.includes('windows') && domain.includes('intel.com'))
    assignees = intel_windows_assignees;
  const apple_silicon_assignees = ['penpornk', 'nitins17'];
  if (lowercased_title.includes('apple') &&
      lowercased_title.includes('silicon') && domain.includes('apple.com'))
    assignees = apple_silicon_assignees;
  if (lowercased_title.includes('nvidia') && domain.includes('nvidia.com')) {
    if (lowercased_title.includes('jax')) {
      assignees.push('hawkinsp', 'yashk2810', 'skye');
    }
    if (lowercased_title.includes('xla') || lowercased_title.includes('gpu')) {
      assignees.push('cheshire', 'gcforster', 'reedwm', 'chsigg', 'xla-rotation');
    }
    if (lowercased_title.includes('tf-trt')) {
      assignees.push(
          'meena-at-work', 'nluehr', 'DEKHTIARJonathan', 'pjannaty', 'poulsbo');
    } else if (lowercased_title.includes('tf')) {
      assignees.push('rohan100jain', 'bfontain');
    }
  }
  if (lowercased_title.includes('linaro') && domain.includes('linaro.org')) {
    if (lowercased_title.includes('arm_ci')) {
      assignees.push('nitins17', 'penpornk');
    }
  }

  const resp_label = await github.rest.issues.addLabels({
    issue_number: context.issue.number,
    owner: context.repo.owner,
    repo: context.repo.repo,
    labels: labels
  });
  if (resp_label.status >= 400) {
    console.log(resp_label);
    throw "Error adding labels to PR";
  }
  if (assignees.length > 0) {
    const resp_assign = await github.rest.issues.addAssignees({
      issue_number: context.issue.number,
      owner: context.repo.owner,
      repo: context.repo.repo,
      assignees: assignees
    });
    if (resp_assign.status >= 400) {
      console.log(resp_assign);
      throw "Error adding assignee to PR";
    }
  }
  return `PR Updated successfully with Labels: ${labels} with Assignees: ${assignees}`;
};

module.exports = {
  filter: filter_action,
  get_email_domain
};
