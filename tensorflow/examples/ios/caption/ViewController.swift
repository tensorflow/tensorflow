//
//  ViewController.swift
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

import UIKit

class ViewController: UIViewController {
    
    let v = vision()
    var open = false
    
    var torch = false
    var captureSession : AVCaptureSession!
    var sessionOutput : AVCapturePhotoOutput!
    var previewLayer : AVCaptureVideoPreviewLayer!
    var previewView = UIView()
    private let sessionQueue = DispatchQueue(label: "session queue",
                                             attributes: [],
                                             target: nil)
    
    let caption = UIDescription()
    
    override var preferredStatusBarStyle: UIStatusBarStyle {
        return .lightContent
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        
        startNet()
        startPreview()
        startCam()
        startCaption()
    }

    //Generate caption from image
    func scan(image: UIImage){
        let description = v.generate_caption(image) ?? "Did not run properly"
        self.changeCaption(text: description)
        NSLog("%@", description)
        open = true //Reset the scan flag
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        sessionQueue.async { [unowned self] in
            if self.captureSession.isRunning {
                self.captureSession.stopRunning() //Ending captureSession
            }
        }
        super.viewWillDisappear(animated)
    }
    
    // Camera + Caption + Init methods in the "view controller" folder
    
}

