//
//  HostingController.swift
//  TestProject2 WatchKit Extension
//
//  Created by yoonseok312 on 2020/08/29.
//  Copyright © 2020 riiid. All rights reserved.
//


import WatchKit
import Foundation
import SwiftUI
import WatchConnectivity

class HostingController: WKHostingController<AnyView> {
  
  let session = WCSession.default
  
  override func awake(withContext context: Any?) {
    super.awake(withContext: context)
    
    session.delegate = self
    session.activate()
  }
  var environment = WatchEnvironment(connectivityProvider: WatchConnectivityProvider())
  override var body: AnyView {
    return AnyView(ContainerView().environmentObject(environment))
  }
}

extension HostingController: WCSessionDelegate {
  
  func session(_ session: WCSession, activationDidCompleteWith activationState: WCSessionActivationState, error: Error?) {
  }
  
  func session(_ session: WCSession, didReceiveMessage message: [String : Any]) {
    
    print("received data: \(message)")
    if let t = message["title"] as? String {
      DispatchQueue.main.async {
        self.environment.changeWord(word:t)
      }
    }
  }
}
