//
//  caption.mm
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


#import "caption.h"

@implementation caption

- (id)init
{
    return [self initWithSentence:[[NSMutableArray alloc] init] withState:tensorflow::Tensor() withLogprob:0.0 withScore:0.0];
}

- (id)initWithSentence:(NSMutableArray*)withSentence withState:(tensorflow::Tensor)withState withLogprob:(double)withLogprob withScore:(double)withScore{
    self = [super init];
    sentence = withSentence;
    state = withState;
    logprob = withLogprob;
    score = withScore;
    return self;
}

- (NSComparisonResult)compare:(caption *)otherCaption {
    return [[NSNumber numberWithDouble:self->score]
            compare:[NSNumber numberWithDouble:otherCaption->score]];
};

@end
