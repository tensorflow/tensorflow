//
//  vision.mm
//  tf_caption_example
//
//  Created by Liam Nakagawa on 1/6/17.
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

#import "vision.h"
#import "CaptionGenerator.h"

@interface vision()
{
    CaptionGenerator *eye;
}
@end

@implementation vision

-(id)init
{
    eye = [[CaptionGenerator alloc] init];
    return self;
}

-(void)load_model
{
    [eye load_model];
}

-(NSString*)generate_caption:(UIImage*)image
{
    return [eye generate_caption:image];
}

@end
