//
//  ContentView.swift
//  watch-out WatchKit Extension
//
//  Created by yoonseok312 on 2020/08/09.
//  Copyright © 2020 yoonseok. All rights reserved.
//

// 2019년 6월에 나온 친구...
import SwiftUI
// Pods - 의존성 관리 해주는 친구
// SwiftPackage Manager

struct SwiftUIView: View {
  
  @State private var animateStrokeStart = true
  @State private var animateStrokeEnd = true
  @State private var isRotating = true
  
  //    struct FontStyle: ViewModifier {
  //
  //        func body(content: Content) -> some View {
  //            return content
  //                .foregroundColor(Color.white)
  //                .font(Font.custom("Arial Rounded MT Bold", size: 15))
  //        }
  //    }
  
  var body: some View {
    VStack {
      ZStack {
        Image("microphone")
        
        Circle()
          .trim(from: animateStrokeStart ? 1/3 : 1/9, to: animateStrokeEnd ? 2/5 : 1)
          .stroke(lineWidth: 10)
          .frame(width: 150, height: 150)
          .foregroundColor(Color(red: 0.0, green: 0.588, blue: 1.0))
          .rotationEffect(.degrees(isRotating ? 360 : 0))
          .onAppear() {
            
            withAnimation(Animation.linear(duration: 1).repeatForever(autoreverses: false)) {
              self.isRotating.toggle()
            }
            
            withAnimation(Animation.linear(duration: 1).delay(0.5).repeatForever(autoreverses: true)) {
              self.animateStrokeStart.toggle()
            }
            
            withAnimation(Animation.linear(duration: 1).delay(0.5).repeatForever(autoreverses: true)) {
              self.animateStrokeEnd.toggle()
            }
        }
      }
      Spacer()
      //            Text("Watch-out이 듣고 있습니다...")
      //                .fontWeight(.bold)
      //                .modifier(FontStyle())
      
      //            HStack {
      //
      //                NavigationLink(destination: Alert(type: "car")) {
      //                    Text(/*@START_MENU_TOKEN@*/"자동차"/*@END_MENU_TOKEN@*/)
      //                }
      //
      //                NavigationLink(destination: Alert(type: "fire")) {
      //                    Text(/*@START_MENU_TOKEN@*/"불이야!"/*@END_MENU_TOKEN@*/)
      //                }
      //            }
    }
  }
}


struct SwiftUIView_Previews: PreviewProvider {
  static var previews: some View {
    /*@START_MENU_TOKEN@*/Text("Hello, World!")/*@END_MENU_TOKEN@*/
  }
}
