//
//  ConectivityProvider.swift
//  watch-out-app
//
//  Created by yoonseok312 on 2020/08/29.
//  Copyright © 2020 Ryan Taylor. All rights reserved.
//

import Foundation
import WatchConnectivity

class ConnectivityProvider: NSObject, WCSessionDelegate {
  
  let session: WCSession
  
  init(session: WCSession = .default) {
    self.session = session
    super.init()
    self.session.delegate = self
    self.session.activate()
  }
  
  func send(message: [String:Any]) -> Void {
    //toggle 상태에 따라 분기
    session.sendMessage(message, replyHandler: nil) { (error) in
      print("🗽")
      print(error.localizedDescription)
    }
  }
  
  func session(_ session: WCSession, activationDidCompleteWith activationState: WCSessionActivationState, error: Error?) {
    return
  }
  
  #if os(iOS)
  public func sessionDidBecomeInactive(_ session: WCSession) { }
  public func sessionDidDeactivate(_ session: WCSession) {
    session.activate()
  }
  #endif
}
