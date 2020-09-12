//  SettingView.swift
//  watch-out-app
//
//  Created by Ryan Taylor on 2020/08/27.
//  Copyright © 2020 Ryan Taylor. All rights reserved.
//

import SwiftUI

struct SettingView: View {
  
  @State private var storeData = UserDefaultsManager()
  
  let appleGray3 = Color(red: 199.0 / 255.0, green: 199.0 / 255.0, blue: 204.0 / 255.0)
  let foreColor = Color.black.opacity(0.5)
  
  var body : some View {
    
    VStack(spacing: 15) {
      
      VStack(spacing: 15) {
        
        Form {
          Section(header: Text("알림 설정"))  {
            
            OptionView(image: "car", name: "자동차 소리", activate: $storeData.carToggle)
            OptionView(image: "fire", name: "불이야 소리", activate: $storeData.fireToggle)
            OptionView(image: "cone", name: "Yes", activate: $storeData.yesToggle)
            OptionView(image: "cone", name: "No", activate: $storeData.noToggle)
            OptionView(image: "cone", name: "Right", activate: $storeData.rightToggle)
          }
          
          Section(header: Text("추가 기능"))  {
            OptionView(image: "settings_other", name: "외부 API 사용", activate: .constant(false))
          }
          
          Section(header: Text("etc")) {
            NavigationLink(destination: Information()){
              HStack(spacing: 5) {
                Image("team").renderingMode(.original).resizable().frame(width: 40, height: 40)
                Text("About Watch-Out")
              }.padding()
            }
          }
        }
      }
      
      Spacer()
    }.navigationBarTitle("설정")
  }
}

struct SettingView_Previews: PreviewProvider {
  static var previews: some View {
    SettingView()
  }
}

struct OptionView: View {
  
  var image = ""
  var name = ""
  var activate: Binding<Bool>
  
  var body : some View {
    
    HStack {
      
      Image(image).renderingMode(.original).resizable().frame(width: 40, height: 40)
      Toggle(isOn: activate) {
        Text(name)
      }
      Spacer(minLength: 15)
      
    }.padding().foregroundColor(Color.black.opacity(0.5))
  }
}
