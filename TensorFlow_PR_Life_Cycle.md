# TensorFlow Pull Request Life Cycle

![Screen Shot 2022-08-30 at 7 27 04 PM](https://user-images.githubusercontent.com/42785357/187579207-9924eb32-da31-47bb-99f9-d8bf1aa238ad.png)

### Typical Pull Request Workflow - 
**1. New PR**
   - As a contributor, you submit a New PR on GitHub.
   - We inspect every incoming PR and add certain labels to the PR such as `size:`, `comp:` etc. At this stage we check if the PR is valid and meets certain quality requirements. 
   - For example - We check if the CLA is signed, PR has sufficient description, if applicable unit tests are added, if it is a reasonable contribution meaning it is not a single liner cosmetic PR.

**2. Valid?**
   - If the PR passes all the quality checks then we go ahead and assign a reviewer.
      - If the PR didn't meet the validation criteria, we request for additional changes to be made to PR to pass quality checks and send it back or on a rare occassion we may reject it.
   
**3. Review**
   - For Valid PR, reviewer (person familiar with the code/functionality) checks if the PR looks good or needs additional changes.
      - If all looks good, reviewer would approve the PR. 
      - If a change is needed, the contributor is requested to make suggested change. 
        - You make the change and submit for the review again.
        - This cycle repeats itself till the PR gets approved.
        - Note: As a friendly reminder we may reach out to you if the PR is awaiting your response for more than 2 weeks.

**4. Approved**
   - Once the PR is approved, it gets `kokoro:force-run` label applied and it initiates CI/CD tests.
   - We can't move forward if theses tests fail. 
       - In such situations, we may request you to make further changes to your PR for the tests to pass. 
   - Once the tests pass, we now bring all the code in the internal code base, using a job called "copybara".

**5. Copy to G3**
   - Once the PR is in Google codebase, we make sure it integrates well with its dependencies and the rest of the system. 
   - Rarely, but If the tests fail at this stage, we cannot merge the code. 
      - If needed, we may come to you to make some changes. 
      - At times, it may not be you, it may be us who may have hit a snag. 
      - Please be patient while we work to fix this. 
   - Once the internal tests pass, we go ahead and merge the code internally as well as externally on GitHub.
