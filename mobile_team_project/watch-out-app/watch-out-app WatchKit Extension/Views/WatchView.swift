//
//  ContentView.swift
//  watch-out-app
//
//  Created by yoonseok312 on 2020/08/29.
//  Copyright © 2020 Ryan Taylor. All rights reserved.
//

import SwiftUI
import WatchKit
import Foundation
import SwiftUI
import WatchConnectivity

struct WatchView: View {
  @EnvironmentObject var viewModel: WatchEnvironment
  static let gradientStart = Color(red: 255.0 / 255, green: 60.0 / 255, blue: 0.0 / 255)
  static let gradientEnd =  Color(red: 255 / 255, green: 108 / 255, blue: 63 / 255)
  
  static let gradientStart_ = Color(red: 255.0 / 255, green: 168.0 / 255, blue: 0.0 / 255)
  static let gradientEnd_ =  Color(red: 255 / 255, green: 198 / 255, blue: 0 / 255)
  
  //    var detectedWord : String = viewModel.word
  var finalIcon : String = "map"

  
  
  var body: some View {
    VStack{
    //NavigationLink(destination: defaultView(), isActive: self.$viewModel.isActive) {EmptyView()}
            // viewModel.word = "changed"
            // 5초 지난 후 뷰 이동
            // 다른 뷰로 연결
        Spacer()
        HStack{
            
          // 아이콘 변경 코드
          if(viewModel.word == "suzy"){
            Image(systemName: "speaker.3.fill").foregroundColor(.white).font(.system(size: 23)).padding(.horizontal,2)
          }else if ( viewModel.word == "bulyiya"){
            Image(systemName: "flame.fill").foregroundColor(.white).font(.system(size: 23)).padding(.horizontal,2)
          }
          Text("소리 감지!!!").font(.system(size: 23, weight: .bold))
        }
        
        Spacer()
        
        HStack{
          ZStack{
            
            if( viewModel.word == "suzy"){
              //yes 소리만 노랑색으로 박스가 바뀜
              RoundedRectangle(cornerRadius: 23, style: .continuous)
                .fill(LinearGradient(
                  gradient: .init(colors: [Self.gradientStart_, Self.gradientEnd_]),
                  startPoint: .init(x: 0.0, y: 0.0),
                  endPoint: .init(x: 0.5, y: 0.6)
                ))
                .frame(width: 185, height: 120)
              
            }else if ( viewModel.word == "bulyiya"){
              // 나머지는 주황색
              RoundedRectangle(cornerRadius: 23, style: .continuous)
                .fill(LinearGradient(
                  gradient: .init(colors: [Self.gradientStart, Self.gradientEnd]),
                  startPoint: .init(x: 0.0, y: 0.0),
                  endPoint: .init(x: 0.5, y: 0.6)
                ))
                .frame(width: 185, height: 120)
                
                }
            
            
            VStack(alignment: .center) {
              if( viewModel.word == "suzy"){
                Text("근처에서 소리").font(.system(size: 18, weight: .black)).padding(.vertical,7).padding(.trailing,55).foregroundColor(Color.init(red: 255.0, green: 255.0, blue: 255.0))
                Text("수지 소리").font(.system(size: 37, weight: .black)).foregroundColor(Color.init(red: 0.0, green: 0.0, blue: 0.0))
                HStack(alignment:.lastTextBaseline){
                  Text("가 들렸습니다").font(.system(size: 18, weight: .black)).padding(.vertical,7).padding(.leading,60).foregroundColor(Color.init(red: 255.0, green: 255.0, blue: 255.0))
                }
                
              }else if ( viewModel.word == "bulyiya"){
                Text("근처에서 소리").font(.system(size: 18, weight: .black)).padding(.vertical,7).padding(.trailing,55).foregroundColor(Color.init(red: 0, green: 0, blue: 0))
                Text("불이야 소리").font(.system(size: 37, weight: .black)).foregroundColor(Color.init(red: 255.0, green: 255.0, blue: 255.0))
                HStack(alignment:.lastTextBaseline){
                  Text("가 들렸습니다").font(.system(size: 18, weight: .black)).padding(.vertical,7).padding(.leading,60).foregroundColor(Color.init(red: 0, green: 0, blue: 0))
                }
              }
              
            }

          }
          
        }
      
    }
    .onAppear{
        self.viewModel.activated()
      //self.viewModel.callNumber(phoneNumber: "01096872456")
    }
 }
}
