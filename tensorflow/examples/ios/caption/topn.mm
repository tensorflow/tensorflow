//
//  topn.mm
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


#import "topn.h"

@implementation topn

- (id)init{
    return [self initWithN:3];
}

- (id)initWithN:(int)size{
    self = [super init];
    n = size;
    queue = [[PriorityQueue alloc] initWithCapacity:size+1];
    return self;
}

- (int)size{
    return (int)[queue size];
}

- (void)push:(caption*)cap{
    [queue add:cap];
    if ([queue size]>n){
        [queue poll];
    }
}

- (NSMutableArray*)extract:(bool)sort{
    if (sort){
        
        NSMutableArray *tmp =   [[NSMutableArray alloc] initWithArray: [[queue toArray] sortedArrayUsingComparator:^(caption* a, caption* b) {
            return [[NSNumber numberWithDouble:b->score] compare:[NSNumber numberWithDouble:a->score]];
        }]]; //This is working
        [queue clear];
        return tmp;
        
    }
    else{
        NSMutableArray *tmp =  [[NSMutableArray alloc] initWithArray: [queue toArray]];
        [queue clear];
        return tmp;
    }
}

- (void)reset{
    [queue clear];
}

@end
