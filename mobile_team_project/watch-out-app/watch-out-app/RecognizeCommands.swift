// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import Foundation

struct RecognizedCommand {
  var score: Float
  var name: String
  var isNew: Bool
}

/**
 This class smoothes out the results by averaging them over a window duration and making sure the
 commands are not duplicated for display.
 */
class RecognizeCommands {
  // MARK: Structures that handles results.
  private struct Command {
    var score: Float
    let name: String
  }
  
  private struct ResultsAtTime {
    let time: TimeInterval
    let scores: [Float]
  }
  
  // MARK: Constants
  private let averageWindowDuration: Double
  private let suppressionTime: Double
  private let minimumCount: Int
  private let minimumTimeBetweenSamples: Double
  private let detectionThreshold: Float
  private let classLabels: [String]
  private let silenceLabel = "_silence_"
  private var previousTopLabel = "_silence_"
  
  
  private var previousTopScore: Float = 0.0
  private var previousTopLabelTime: TimeInterval = Date.distantPast.timeIntervalSince1970 * 1000
  private var previousResults: [ResultsAtTime] = []
  
  /**
   Initializes RecognizeCommands with specified parameters.
   */
  init(averageWindowDuration: Double, detectionThreshold: Float, minimumTimeBetweenSamples: Double, suppressionTime: Double, minimumCount: Int, classLabels: [String]) {
    self.averageWindowDuration = averageWindowDuration
    self.detectionThreshold = detectionThreshold
    self.minimumTimeBetweenSamples = minimumTimeBetweenSamples
    self.suppressionTime = suppressionTime
    self.minimumCount = minimumCount
    self.classLabels = classLabels
  }
  
  /**
   This function averages the results obtained over an average window duration and prunes out any
   old results.
   */
  func process(latestResults: [Float], currentTime: TimeInterval) -> RecognizedCommand? {
    
    guard latestResults.count == classLabels.count else {
      fatalError("There should be \(classLabels.count) in results. But there are \(latestResults.count) results")
    }
    
    // Checks if the new results were identified at a later time than the currently identified
    // results.
    if let first = previousResults.first, first.time > currentTime {
      fatalError("Results should be provided in increasing time order")
    }
    
    if let lastResult = previousResults.last {
      
      let timeSinceMostRecent = currentTime - previousResults[previousResults.count - 1].time
      
      // If not enough time has passed after the last inference, we return the previously identified
      // result as legitimate one.
      if timeSinceMostRecent < minimumTimeBetweenSamples {
        return RecognizedCommand(score: previousTopScore, name: previousTopLabel, isNew: false)
      }
    }
    
    // Appends the new results to the identified results
    let results: ResultsAtTime = ResultsAtTime(time: currentTime, scores: latestResults)
    
    previousResults.append(results)
    
    let timeLimit = currentTime - averageWindowDuration
    
    // Flushes out all the results currently held that less than the average window duration since
    // they are considered too old for averaging.
    while previousResults[0].time < timeLimit {
      previousResults.removeFirst()
      
      guard previousResults.count > 0 else {
        break
      }
    }
    
    // If number of results currently held to average is less than a minimum count, return the score
    // as zero so that no command is identified.
    if previousResults.count < minimumCount {
      return RecognizedCommand(score: 0.0, name: previousTopLabel, isNew: false)
    }
    
    // Creates an average of the scores of each classes currently held by this class.
    var averageScores:[Command] = []
    for i in 0...classLabels.count - 1 {
      
      let command = Command(score: 0.0, name: classLabels[i])
      averageScores.append(command)
      
    }
    
    for result in previousResults {
      
      let scores = result.scores
      for i in 0...scores.count - 1 {
        averageScores[i].score = averageScores[i].score + scores[i] / Float(previousResults.count)
        
      }
    }
    
    // Sorts scores in descending order of confidence.
    averageScores.sort { (first, second) -> Bool in
      return first.score > second.score
    }
    
    var timeSinceLastTop: Double = 0.0
    
    // If silence was detected previously, consider the current result with the best average as a
    // new command to be displayed.
    if (previousTopLabel == silenceLabel ||
      previousTopLabelTime == (Date.distantPast.timeIntervalSince1970 * 1000)) {
      
      timeSinceLastTop = Date.distantFuture.timeIntervalSince1970 * 1000
    }
    else {
      timeSinceLastTop = currentTime - previousTopLabelTime
    }
    
    // Return the results
    var isNew = false
    if (averageScores[0].score > detectionThreshold && timeSinceLastTop > suppressionTime) {
      
      previousTopScore = averageScores[0].score
      previousTopLabel = averageScores[0].name
      previousTopLabelTime = currentTime
      isNew = true
    }
    else {
      isNew = false
    }
    
    return RecognizedCommand(
      score: previousTopScore, name: previousTopLabel, isNew: isNew)
  }
}
