import SwiftUI
import UIKit
import Foundation

struct StartView: View {
  
  @State var nextScreenShown = false
  
  var body: some View {
    
    ZStack {
      Circle()
        .fill(Color(000099))
      Circle()
        .fill(Color(0x0000ff))
        .frame(width: 300, height: 300)
      Circle()
        .fill(Color(0x6666ff))
        .frame(width: 150, height: 150)
      VStack {
        NavigationLink(destination: SwiftUIView()) {
          Text("Start")
            .bold()
            .font(Font.custom("Arial Rounded MT Bold", size: 15))
          
        }
      }
      //.buttonStyle(PlainButtonStyle())
      
    }
    
  }
}


struct StartView_Previews: PreviewProvider {
  static var previews: some View {
    StartView()
  }
}

struct AlarmView: View {
  
  //@Binding var nextScreenShown : Bool
  
  var body: some View {
    Text("Alarm view")
  }
}

extension Color {
  init(_ hex: UInt32, opacity:Double = 1.0) {
    let red = Double((hex & 0xff0000) >> 16) / 255.0
    let green = Double((hex & 0xff00) >> 8) / 255.0
    let blue = Double((hex & 0xff) >> 0) / 255.0
    self.init(.sRGB, red: red, green: green, blue: blue, opacity: opacity)
  }
}

//func changeRootView<V>(to view: V, animated: Bool) where V: View {
//  let window = UIApplication.shared.application.windows.first!
//  let navigationView = NavigationView {
//    view
//  }.navigationViewStyle(StackNavigationViewStyle())
//  window.rootViewController = UIHostingController(rootView: navigationView)
//
//  if animated {
//    UIView.transition(with: window,
//                      duration: 0.3,
//                      options: .transitionCrossDissolve,
//                      animations: {},
//                      completion: nil)
//  }
//}
