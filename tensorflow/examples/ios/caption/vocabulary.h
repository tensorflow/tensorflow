//
//  vocabulary.h
//  tf_caption_example
//
//  Created by Liam Nakagawa on 1/6/17.
//  Copyright © 2017 Liam Nakagawa. All rights reserved.
//
//  Adapted from https://github.com/tensorflow/models/blob/master/im2txt/im2txt/inference_utils/vocabulary.py
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

@interface vocabulary : NSObject {
    
    @public int start_id;
    @public int end_id;
    @public int unk_id;
}
- (id)init;
- (id)initWithVocab:(NSString*)vocab_file Type:(NSString*)vocab_type;
- (int)word_to_id:(NSString*)word;
- (NSString*)id_to_word:(int)word_id;


@end
