//
//  Alert.swift
//  watch-out WatchKit Extension
//
//  Created by Ryan Taylor on 2020/08/13.
//  Copyright © 2020 yoonseok. All rights reserved.
//

import Foundation
import SwiftUI

struct Alert: View {
  
  struct FontStyle: ViewModifier {
    
    func body(content: Content) -> some View {
      return content
        .font(.headline)
        .font(Font.custom("Arial Rounded MT Bold", size: 18))
    }
  }
  var type: String
  
  var body: some View {
    VStack {
      VStack {
        HStack(alignment: .center) {
          Image("\(type)")
          Text("\(type == "fire" ? "불" : "자동차") 조심 하세요!!!")
            .modifier(FontStyle())
        }
        Text("근처에서\n \(type == "fire" ? "불" : "자동차") 관련 소리가\n 들렸습니다!")
          .fontWeight(.bold)
          .multilineTextAlignment(.center)
          .frame(width: 150.0, height: 120.0)
          .modifier(FontStyle())
          .background(
            RoundedRectangle(cornerRadius: 20)
              .fill(LinearGradient(
                gradient: Gradient(colors: [.pink, .blue]),
                startPoint: .leading,
                endPoint: .trailing))
        )
      }
    }
  }
}

struct Alert_Previews: PreviewProvider {
  static var previews: some View {
    Alert(type: "fire")
  }
}
