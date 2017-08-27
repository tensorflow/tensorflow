//
//  caption.swift
//  tf_caption_example
//
//  Created by Liam Nakagawa on 8/27/17.
//  Copyright © 2017 Liam Nakagawa. All rights reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

import Foundation
import UIKit

extension ViewController {
    
    func changeCaption(text: String){
        caption.text = text
    }

}

@IBDesignable
class UIDescription: UILabel {
    override init(frame: CGRect) {
        super.init(frame: frame)
    }
    
    required init?(coder aDecoder: NSCoder) {
        super.init(coder: aDecoder)
    }
    
    override func drawText(in rect: CGRect) {
        let yMargin : CGFloat = 16 //Distance from text to bottom of label frame
        let height = self.sizeThatFits(rect.size).height
        let y = rect.origin.y + rect.height - height - yMargin
        super.drawText(in: CGRect(x: rect.origin.x, y: y, width: rect.width, height: height))
    }
}
