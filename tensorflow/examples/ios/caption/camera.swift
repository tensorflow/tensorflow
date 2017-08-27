//
//  camera.swift
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

extension ViewController : AVCapturePhotoCaptureDelegate {
    
    //Capture a photo and generate a caption
    func takePicture(){
        if open { open = false
            NSLog("%@", "Tap Received")
            sessionOutput.capturePhoto(with: getSettings(), delegate: self)
        }
    }
    
    //AVKit delegate method for photo capture
    func capture(_ captureOutput: AVCapturePhotoOutput, didFinishProcessingPhotoSampleBuffer photoSampleBuffer: CMSampleBuffer?, previewPhotoSampleBuffer: CMSampleBuffer?, resolvedSettings: AVCaptureResolvedPhotoSettings, bracketSettings: AVCaptureBracketedStillImageSettings?, error: Error?) {
        if let error = error {
            print(error.localizedDescription)
        }
        if let sampleBuffer = photoSampleBuffer, let previewBuffer = previewPhotoSampleBuffer,
            let dataImage = AVCapturePhotoOutput.jpegPhotoDataRepresentation(forJPEGSampleBuffer: sampleBuffer,
                                                                             previewPhotoSampleBuffer: previewBuffer) {
            if let image = UIImage(data: dataImage){
                NSLog("%@", "Took Picture")
                scan(image: image) //Switch for different modes of inference
            }
            else {
                NSLog("%@", "Failed to scan image")
                open = true
            }
        }
    }
    
    //Generate capture settings
    func getSettings() -> AVCapturePhotoSettings {
        let settings = AVCapturePhotoSettings()
        
        settings.flashMode = torch ? .auto : .off
        settings.isAutoStillImageStabilizationEnabled = true
        settings.isHighResolutionPhotoEnabled = true
        let previewPixelType = settings.availablePreviewPhotoPixelFormatTypes.first!
        let previewFormat = [kCVPixelBufferPixelFormatTypeKey as String: previewPixelType,
                             kCVPixelBufferWidthKey as String: 160,
                             kCVPixelBufferHeightKey as String: 160,
                             ]
        settings.previewPhotoFormat = previewFormat
        return settings
    }

    
    
}
