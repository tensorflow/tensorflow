//
//  WatchConnectivityProvider.swift
//  watch-out-app
//
//  Created by yoonseok312 on 2020/08/29.
//  Copyright © 2020 Ryan Taylor. All rights reserved.
//

import Foundation
import SwiftUI
import WatchConnectivity


class WatchConnectivityProvider: NSObject, WCSessionDelegate {
  @EnvironmentObject var viewModel: WatchEnvironment
  
  let session: WCSession
  
  init(session: WCSession = .default) {
    self.session = session
    super.init()
    self.session.delegate = self
    self.session.activate()
  }
  
  func session(_ session: WCSession, activationDidCompleteWith activationState: WCSessionActivationState, error: Error?) {
  }
  
  func session(_ session: WCSession, didReceiveMessage message: [String : Any]) {
    
    print("received data: \(message)")
    if let t = message["title"] as? String {
      viewModel.word = t
      print("👺")
    }
    //    if let c = message["content"] as? String {
    //        self.content.setText(c)
    //    }
  }
}
