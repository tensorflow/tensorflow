//
//  initializations.swift
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
    
    func startNet(){
        v.load_model()
        open = true
    }
    
    func startPreview(){
        let width = UIScreen.main.bounds.width
        let height = UIScreen.main.bounds.height
        previewView.frame = CGRect(x: 0, y: 0, width: width, height: height)
        self.view.addSubview(previewView)
    }

    func startCam(){
        let AVStatus = AVCaptureDevice.authorizationStatus(forMediaType: AVMediaTypeVideo)
        
        if  AVStatus ==  AVAuthorizationStatus.notDetermined {
            AVCaptureDevice.requestAccess(forMediaType: AVMediaTypeVideo, completionHandler: { (granted :Bool) -> Void in
                
                if granted == true
                {
                    print("Accepted Camera Permission")
                }
                else
                {
                    print("Rejected Camera Permission")
                    self.previewView.accessibilityLabel = "No Camera Permission"
                    self.changeCaption(text: "No Camera Permission")
                    self.previewView.isUserInteractionEnabled = false
                }
                
            });
        }
        else if AVStatus == AVAuthorizationStatus.denied || AVStatus == AVAuthorizationStatus.restricted {
            print("Rejected Camera Permission")
            self.previewView.accessibilityLabel = "No Camera Permission"
            self.changeCaption(text: "No Camera Permission \n")
            self.previewView.isUserInteractionEnabled = false
        }
        
        //Initialize AVCaptureSession and output
        captureSession = AVCaptureSession()
        captureSession.sessionPreset = AVCaptureSessionPresetPhoto
        sessionOutput = AVCapturePhotoOutput()
        sessionOutput.isHighResolutionCaptureEnabled = true
        
        //Add input device and intialize previewLayer
        do {
            var defaultVideoDevice: AVCaptureDevice?
            if #available(iOS 10.2, *) {
                if let dualCameraDevice = AVCaptureDevice.defaultDevice(withDeviceType: .builtInDualCamera, mediaType: AVMediaTypeVideo, position: .back) {
                    defaultVideoDevice = dualCameraDevice
                } else if let backCameraDevice = AVCaptureDevice.defaultDevice(withDeviceType: .builtInWideAngleCamera, mediaType: AVMediaTypeVideo, position: .back) {
                    defaultVideoDevice = backCameraDevice
                } else if let frontCameraDevice = AVCaptureDevice.defaultDevice(withDeviceType: .builtInWideAngleCamera, mediaType: AVMediaTypeVideo, position: .front) {
                    defaultVideoDevice = frontCameraDevice
                }
            } else {
                if let backCameraDevice = AVCaptureDevice.defaultDevice(withDeviceType: .builtInWideAngleCamera,
                                                                        mediaType: AVMediaTypeVideo, position: .back) {
                    defaultVideoDevice = backCameraDevice
                } else if let frontCameraDevice = AVCaptureDevice.defaultDevice(withDeviceType: .builtInWideAngleCamera, mediaType: AVMediaTypeVideo, position: .front) {
                    defaultVideoDevice = frontCameraDevice
                }
            }
            
            if let device = defaultVideoDevice {
                //Set torch value
                torch = device.hasFlash
                
                let input = try AVCaptureDeviceInput(device: device)
                
                //let input = try AVCaptureDeviceInput(device: device)
                if (captureSession.canAddInput(input)) {
                    captureSession.addInput(input)
                    if (captureSession.canAddOutput(sessionOutput)) {
                        print("Loading Preview Layer")
                        captureSession.addOutput(sessionOutput)
                        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
                        previewLayer.connection.videoOrientation = AVCaptureVideoOrientation.portrait
                        previewLayer.frame = previewView.bounds
                        previewLayer.videoGravity = AVLayerVideoGravityResizeAspectFill
                        previewView.layer.addSublayer(previewLayer)
                        
                        //Begin capture session
                        captureSession.startRunning()
                        
                        //Add tap gesture to PreviewView
                        let tap = UITapGestureRecognizer(target: self, action: #selector(self.takePicture))
                        previewView.addGestureRecognizer(tap)
                    }
                }
            }
        }
        catch{
            print("exception!");
        }
    }
    
    func startCaption(){
        caption.text = "Tap to generate a caption"
        caption.textColor = UIColor.white
        caption.font = UIFont.boldSystemFont(ofSize: 35)
        caption.textAlignment = NSTextAlignment.center
        caption.frame = CGRect(x: 8, y: 64, width: previewView.frame.width - 16, height: previewView.frame.height - 64)
        caption.numberOfLines = 0
        caption.isAccessibilityElement = true
        self.view.addSubview(caption)
    }


}
