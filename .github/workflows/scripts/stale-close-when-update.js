/*
   Script will close all issues that have been labeled as stale for more than 7 days
   and have had at comment since being labeled, invoked from stale management.
*/

module.exports = async ({ github, context }) => {

    // fetch all the issues with stale label.
    let issues = await github.rest.issues.listForRepo({
        owner: context.repo.owner,
        repo: context.repo.repo,
        state: "open",
        labels: "stale"
    });
    if (issues.status != 200)
        return

    let issueList = issues.data
    for (let i = 0; i < issueList.length; i++) {
        let number = issueList[i].number;
        let nodeType = issueList[i].node_id
        let issueLabels = issueList[i].labels
        let stringLabel = JSON.stringify(issueLabels)
        console.log("issue list",issueList[i])
        let closeAfterStale = 7
        if(nodeType.startsWith('PR'))
           closeAfterStale = 14
        else if(stringLabel.indexOf('stat:contribution welcome') !=-1 || stringLabel.indexOf('stat:good first issue') !=-1 )
           closeAfterStale = 365
       
        //fetch all the events inside the issue.
        let resp = await github.rest.issues.listEventsForTimeline({
            owner: context.repo.owner,
            repo: context.repo.repo,
            issue_number: number,
        });

        let events = resp.data;
        let closeIssue = false
        for (let i = 0; i < events.length; i++) {
            let event_details = events[i];  
            // Check if issue is marked as stale.    
            if (event_details.event == 'labeled' && event_details.label && event_details.label.name == "stale") {  
                let currentDate = new Date();
                let labeledDate = new Date(event_details.created_at)
                let timeInDays = (currentDate - labeledDate) / 86400000
                console.log(`Issue ${number} stale label is ${timeInDays} days old.`)
                if (timeInDays > closeAfterStale)
                    closeIssue = true
            }
            if (event_details.event == 'unlabeled' && event_details.label && event_details.label.name == "stale"){
                console.log(`Stale is unlabel for issue ${number}.`)
                    closeIssue = false
            }
           
        }
        if(closeIssue){
            console.log(`Closing the issue ${number} more then 7 days old with stale label.`)
            let comment = "This issue was closed because it has been inactive for 7 days since being marked as stale. Please reopen if you'd like to work on this further."
            if(closeAfterStale == 14)
               comment = "This PR was closed because it has been inactive for 14 days since being marked as stale. Please reopen if you'd like to work on this further."
            else if(closeAfterStale == 365)
              comment = "This issue was closed because it has been inactive for 1 year"

            await github.rest.issues.update({
                owner: context.repo.owner,
                repo: context.repo.repo,
                issue_number: number,
                state:"closed"
              });
              
              await github.rest.issues.createComment({
                issue_number: number,
                owner: context.repo.owner,
                repo: context.repo.repo,
                body: comment
            });
        }
    }

}

