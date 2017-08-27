//
//  vocabulary.mm
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


#import "vocabulary.h"
#include <fstream>

@implementation vocabulary



NSString *start_word = @"<S>";
NSString *end_word = @"</S>";
NSString *unk_word =@"<UNK>";
NSMutableArray<NSString *> *reverse_vocab = [NSMutableArray new];
NSMutableDictionary *_vocab = [NSMutableDictionary dictionary];

- (id)init
{
    return [self initWithVocab:@"words" Type:@"txt"];
}

//Assuming length of vocab file is 11519 for testing, works with any length
//Parses out "B'" and "' *score" (formatting differs for newer vocabs)
- (id)initWithVocab:(NSString*)vocab_file Type:(NSString*)vocab_type{
    self = [super init];
    NSString* labels_path = [[NSBundle mainBundle] pathForResource:vocab_file ofType:vocab_type];
    std::ifstream t;
    t.open([labels_path UTF8String]);
    std::string line;
    while (t) {
        std::getline(t, line);
        std::string delimiter = line.substr(1,1) + " ";
        auto pos = line.find(delimiter);
        line = line.substr(2,pos-2);
        [reverse_vocab addObject:[NSString stringWithCString:line.c_str() encoding: [NSString defaultCStringEncoding]]];
    }
    t.close();
    
    for (int i = 0; i < reverse_vocab.count; i++)
    {
        NSNumber *tempnum = [[NSNumber alloc] initWithInt:i];
        [_vocab setObject:tempnum forKey:reverse_vocab[i]];
    }

    //Testing
    NSLog(@"%@", reverse_vocab[0]);
    NSLog(@"%@", reverse_vocab[2]);
    NSLog(@"%@", reverse_vocab[11518]);
    NSLog(@"%@", reverse_vocab[128]);
    NSLog(@"%@", reverse_vocab[11519]);
    NSLog(@"%@", reverse_vocab[455]);
    NSLog(@"%@", reverse_vocab[5411]);
    NSLog(@"%@", _vocab[reverse_vocab[0]]);
    NSLog(@"%@", _vocab[reverse_vocab[2]]);
    NSLog(@"%@", _vocab[@"trudging"]);
    
    //Start and end IDs specific to the pre-trained model
    start_id = 2;
    end_id = 1;
    unk_id = 11519;
    return self;
};
- (int)word_to_id:(NSString*)word{
    return [_vocab[word] intValue];
};
- (NSString*)id_to_word:(int)word_id{
    return reverse_vocab[word_id];
};

@end
