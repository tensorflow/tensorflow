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

/**
    - 작성자: 김창윤
    - 작성날짜: 2020-08-20
 */
import UIKit
import SwiftUI
import WatchConnectivity

class ViewController: UIViewController {

//  let contentView = UIHostingController(rootView: MainView())
  var session: WCSession?
  
  // MARK: Objects Handling Core Functionality
  private var modelDataHandler: ModelDataHandler? =
    ModelDataHandler(modelFileInfo: ConvActions.modelInfo, labelsFileInfo: ConvActions.labelsInfo)
  private var audioInputManager: AudioInputManager?

  // MARK: Instance Variables
  private var words: [String] = []
  private var result: Result?
  private var highlightedCommand: String?
  private var bufferSize: Int = 0

  // MARK: View Handling Methods
  override func viewDidLoad() {
    super.viewDidLoad()
    
//    addChild(contentView)
//    view.addSubview(contentView.view)
//    self.setupConstraints()

    guard let handler = modelDataHandler else {
      return
    }

    // Displays lables
    words = handler.offsetLabelsForDisplay()
    startAudioRecognition()
    
    self.configureWatchKitSesstion()
  }
  
//  fileprivate func setupConstraints() {
//    contentView.view.translatesAutoresizingMaskIntoConstraints = false
//    contentView.view.topAnchor.constraint(equalTo: view.topAnchor).isActive = true
//    contentView.view.bottomAnchor.constraint(equalTo: view.bottomAnchor).isActive = true
//    contentView.view.leftAnchor.constraint(equalTo: view.leftAnchor).isActive = true
//    contentView.view.rightAnchor.constraint(equalTo: view.rightAnchor).isActive = true
//  }
  
  func configureWatchKitSesstion() {
    
    if WCSession.isSupported() {//4.1
      session = WCSession.default//4.2
      session?.delegate = self//4.3
      session?.activate()//4.4
    }
  }
  
  @IBAction func tapSendDataToWatch(_ sender: Any) {
    
    print("?")
    
    if let validSession = self.session, validSession.isReachable {//5.1
      let data: [String: Any] = ["iPhone": self.highlightedCommand as Any] // Create your Dictionay as per uses
      
      print(data)
      validSession.sendMessage(data, replyHandler: nil, errorHandler: nil)
    }
  }
  
  override var preferredStatusBarStyle : UIStatusBarStyle {
    return .lightContent
  }

  // MARK: Storyboard Segue Handlers
  override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
    super.prepare(for: segue, sender: sender)
  }

  /**
   Initializes the AudioInputManager and starts recognizing on the output buffers.
   */
  private func startAudioRecognition() {

    guard let handler = modelDataHandler else {
      return
    }

    audioInputManager = AudioInputManager(sampleRate: handler.sampleRate)
    audioInputManager?.delegate = self

    guard let workingAudioInputManager = audioInputManager else {
      return
    }

    bufferSize = workingAudioInputManager.bufferSize

    workingAudioInputManager.checkPermissionsAndStartTappingMicrophone()
//    workingAudioInputManager.start { (channelDataArray) in
//
//      self.runModel(onBuffer: Array(channelDataArray[0..<handler.sampleRate]))
//      self.runModel(onBuffer: Array(channelDataArray[handler.sampleRate..<bufferSize]))
//    }
  }

  /**
   This method runs hands off inference to the ModelDataHandler by passing the audio buffer.
   */
  private func runModel(onBuffer buffer: [Int16]) {

    // buffer: 2차원 배열로 변환된 음성
    result = modelDataHandler?.runModel(onBuffer: buffer)

    // Updates the results on the screen.
    DispatchQueue.main.async {
      guard let recognizedCommand = self.result?.recognizedCommand else {
        return
      }
      // 인식이 잘되는지 console에 출력 합니다.
      print(self.result?.recognizedCommand)
      self.highlightedCommand =  recognizedCommand.name
      
      if let validSession = self.session, validSession.isReachable {//5.1
        let data: [String: Any] = ["title": self.highlightedCommand!, "content": self.highlightedCommand! + "!!!"] // Create your Dictionay as per uses
         validSession.sendMessage(data, replyHandler: nil, errorHandler: nil)
       }
    }
  }
}

// WCSession delegate functions
extension ViewController: WCSessionDelegate {
  
  func sessionDidBecomeInactive(_ session: WCSession) {
  }
  
  func sessionDidDeactivate(_ session: WCSession) {
  }
  
  func session(_ session: WCSession, activationDidCompleteWith activationState: WCSessionActivationState, error: Error?) {
  }
  
//  func session(_ session: WCSession, didReceiveMessage message: [String : Any]) {
//    print("received message: \(message)")
//    DispatchQueue.main.async { //6
//      if let value = message["watch"] as? String {
//        self.label.text = value
//      }
//    }
//  }
}


extension ViewController: AudioInputManagerDelegate {

  func didOutput(channelData: [Int16]) {

    guard let handler = modelDataHandler else {
      return
    }
    
//    print("didOutput");

    self.runModel(onBuffer: Array(channelData[0..<handler.sampleRate]))
    self.runModel(onBuffer: Array(channelData[handler.sampleRate..<bufferSize]))
  }

  func showCameraPermissionsDeniedAlert() {

    let alertController = UIAlertController(title: "Microphone Permissions Denied", message: "Microphone permissions have been denied for this app. You can change this by going to Settings", preferredStyle: .alert)

    let cancelAction = UIAlertAction(title: "Cancel", style: .cancel, handler: nil)
    let settingsAction = UIAlertAction(title: "Settings", style: .default) { (action) in
      UIApplication.shared.open(URL(string: UIApplication.openSettingsURLString)!, options: [:], completionHandler: nil)
    }

    alertController.addAction(cancelAction)
    alertController.addAction(settingsAction)

    present(alertController, animated: true, completion: nil)
  }
}
