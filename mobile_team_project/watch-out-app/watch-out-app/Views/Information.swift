//
//  Information.swift
//  watch-out-app
//
//  Created by 공예슬 on 2020/08/30.
//  Copyright © 2020 Ryan Taylor. All rights reserved.
//

import SwiftUI

struct titleStyle: ViewModifier {
  func body(content: Content) -> some View {
    return content
      .foregroundColor(Color.black)
      .font(Font.custom("AppleSDGothicNeo-Bold", size: 30))
    
  }
}
struct textStyle: ViewModifier {
  func body(content: Content) -> some View {
    return content
      .foregroundColor(Color.black)
      .font(Font.custom("AppleSDGothicNeo-SemiBold", size: 18))
    
  }
}
struct textSmallStyle: ViewModifier {
  func body(content: Content) -> some View {
    return content
      .foregroundColor(Color.black)
      .font(Font.custom("AppleSDGothicNeo-Light", size: 15))
  }
}

struct Information: View {
  var body: some View {
    VStack(alignment: .leading, spacing: 20) {
      
      Text("Watch Out").modifier(titleStyle())
      
      Text("# Watch Out은 이런 사람을 위해서 만들었습니다.").modifier(textStyle())
      Text("화재나 교통사고 발생시 위험 상황을 소리로 제대로 인지하지 못해 큰 피해를 입을 수 있습니다. 비단 청각장애인 뿐만 아니라, 노이즈 캔슬링 등 외부 소음을 차단하는 이어폰들의 성능이 향상되면서 일반인들도 주변의 위험 상황을 인지하지 못해 사고를 당하는 사례가 빈번히 발생하고 있습니다. Watch Out은 이러한 문제를 해결해 드립니다.").modifier(textSmallStyle())
      
      Text("# Watch Out은 대신 들어드립니다.").modifier(textStyle())
      Text("이 앱은 위험한 소리를 인식하여 사용자에게 알려주는 어플입니다. 자동차의 경적 소리 또는 사이렌 소리와 같은 비언어적인 소리 분만 아니라 사용자의 이름을 학습하여 누군가 당신의 이름을 불렀다는 것을 바로 알 수 있고, \"불이야\" 와 같은 위험한 키워드의 소리도 들을 수 있습니다. 웨어러블 기기인 애플워치를 사용 중이시라면 더 편리하게 알림을 받아보실 수 있습니다.").modifier(textSmallStyle())
      
      Text("# Watch Out은 Tensorflow Lite를 시용합니다.").modifier(textStyle())
      Text("Tensorflow Lite를 사용하여 인터넷 접속이 없는 상황에서도 위험한 소리를 듣고 인식할 수 있습니다.").modifier(textSmallStyle())
      
      Spacer()
      Text("만든이").modifier(titleStyle())
      Text("강상훈, 공예슬, 김도연, 김창윤, 김하림, 맹윤호, 서미지, 송보영, 양윤석, 이보성").modifier(textStyle())
      
    }.navigationBarTitle(Text("더 알아보기"), displayMode: .inline)
      .padding(30)
  }
}

struct Information_Previews: PreviewProvider {
  static var previews: some View {
    Information()
  }
}
