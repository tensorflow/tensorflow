//
//  caption.h
//  tf_caption_example
//
//  Created by Liam Nakagawa on 1/6/17.
//  Copyright © 2017 Liam Nakagawa. All rights reserved.
//
//  Adapted from https://github.com/tensorflow/models/blob/master/im2txt/im2txt/inference_utils/caption_generator.py
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


#import <UIKit/UIKit.h>

#include <memory>
#include "tensorflow/core/public/session.h"

@interface caption : NSObject {
    
    @public NSMutableArray<NSNumber *> *sentence;
    tensorflow::Tensor state;

    double logprob;
    @public double score;
}

- (id)init;
- (id)initWithSentence:(NSMutableArray*)sentence withState:(tensorflow::Tensor)state withLogprob:(double)logprob withScore:(double)score;
- (NSComparisonResult)compare:(caption *)otherCaption;


@end
