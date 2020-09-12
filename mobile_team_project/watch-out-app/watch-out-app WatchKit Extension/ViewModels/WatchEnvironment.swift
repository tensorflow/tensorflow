//
//  WatchViewModel.swift
//  watch-out-app
//
//  Created by yoonseok312 on 2020/08/29.
//  Copyright © 2020 Ryan Taylor. All rights reserved.
//

import UIKit
import Foundation
import SwiftUI
import WatchConnectivity

class WatchEnvironment: ObservableObject {
  
  @Published var word: String = "changed"
  @Published var isActive : Bool = false
    
    
    
  private(set) var connectivityProvider: WatchConnectivityProvider
  
  init(connectivityProvider: WatchConnectivityProvider) {
    self.connectivityProvider = connectivityProvider
    
  }
  
  func changeWord(word: String) {
    self.word = word
    self.isActive = true
  }
    
    func activated() {
        DispatchQueue.main.asyncAfter(deadline: .now() + 5) {
            self.isActive = false
       }
     }
  
  
}
