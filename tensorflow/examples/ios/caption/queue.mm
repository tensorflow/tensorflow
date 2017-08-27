//
//  qsueue.m
//  queue
//
//  Created by Jeremy Fox on 12/29/14.
//  Copyright (c) 2014 Jeremy Fox. All rights reserved.
//

#import "queue.h"

@interface PriorityQueue ()
@property (nonatomic, strong) NSMutableArray<caption *>* queue;
@end

@implementation PriorityQueue

- (id)init {
    if (self = [super init]) {
        _queue = [@[] mutableCopy];
    }
    return self;
}

- (id)initWithObjects:(NSSet*)objects
{
    if (self = [super init]) {
        _queue = [@[] mutableCopy];
        for (id<NSObject> object in objects) {
            [self add:object];
        }
    }
    return self;
}

- (id)initWithCapacity:(int)capacity
{
    if (self = [super init]) {
        _queue = [NSMutableArray arrayWithCapacity:capacity];
    }
    return self;
}

- (BOOL)isEmpty
{
    return (self.size > 0);
}

- (NSUInteger)size
{
    return self.queue.count;
}

- (BOOL)contains:(id<NSObject>)object
{
    return [self.queue containsObject:object];
}

- (void)clear
{
    [self.queue removeAllObjects];
}

- (void)add:(id<NSObject>)object
{
    [self insert:object];
}

- (void)remove:(id<NSObject>)object
{
    [self.queue removeObject:object];
}

- (id<NSObject>)peek
{
    return self.queue[0];
}

- (id<NSObject>)poll
{
    id<NSObject> object = self.queue[0];
    [self.queue removeObject:object];
    return object;
}

- (NSMutableArray<caption *>*)toArray
{
    return self.queue.copy;
}

#pragma mark ----------------------
#pragma mark Private
#pragma mark ----------------------

- (NSComparator)comparator
{
    if (_comparator) {
        return _comparator;
    } else {
        return ^NSComparisonResult(caption* obj1, caption* obj2) {
            NSComparisonResult result = NSOrderedSame;
            if (obj1->score < obj2->score) {
                result = NSOrderedAscending;
            } else if (obj1->score > obj2->score) {
                result = NSOrderedDescending;
            }
            return result;
        };
    }
}

- (void)insert:(id<NSObject>)object
{
    if (self.size == 0) {
        [self.queue addObject:object];
        return;
    }
    
    NSUInteger mid = 0;
    NSUInteger min = 0;
    NSUInteger max = self.queue.count - 1;
    BOOL found = NO;
    
    while (min <= max) {
        
        mid = (min + max) / 2;
        
        NSComparisonResult result = self.comparator(object, self.queue[mid]);
        
        if (result == NSOrderedSame) {
            mid++;
            found = YES;
            break;
        } else if (result == NSOrderedAscending) {
            max = mid - 1;
            if (max == NSUIntegerMax) {
                found = YES;
                break;
            }
        } else if (result == NSOrderedDescending) {
            min = mid + 1;
        }
    }
    
    if (found) {
        // Index found at mid
        [self.queue insertObject:object atIndex:mid];
    } else {
        // Index not found, use min
        [self.queue insertObject:object atIndex:min];
    }
}

@end
