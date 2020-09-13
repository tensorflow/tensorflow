//
//  defaultView.swift
//  watch-out-app WatchKit Extension
//
//  Created by 이보성 on 2020/08/31.
//  Copyright © 2020 Ryan Taylor. All rights reserved.
//

import SwiftUI

struct ContainerView: View {

  @EnvironmentObject var viewModel: WatchEnvironment
  var body: some View {

    Group {
      if viewModel.word != "changed" && viewModel.isActive {
      WatchView()
    } else {
      DefaultView()
    }
    }
  }
}

struct DefaultView: View {
  var body: some View {
    VStack {
        Text("모든 설정은")
        Text("아이폰 앱에서")
        Text("해주세요!")
    }
  }
}
