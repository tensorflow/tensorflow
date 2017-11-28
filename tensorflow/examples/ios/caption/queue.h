//
//  queue.h
//  queue
//
//  Created by Jeremy Fox on 12/29/14.
//  Copyright (c) 2014 Jeremy Fox. All rights reserved.
//

#import <Foundation/Foundation.h>
#import "caption.h"

@interface PriorityQueue : NSObject

/**
 * Can be used to set a custom comparator. Should be used when using a PriorityQueue with custom objects. If this is not set, natural ordering will be used.
 */
@property (nonatomic, assign) NSComparator comparator;

- (id)initWithObjects:(NSSet*)objects;
- (id)initWithCapacity:(int)capacity;

/**
 * Can be used to determine if the queue is empty.
 *
 * @return YES if the queue is empty. NO if the queue is not empty.
 */
- (BOOL)isEmpty;

/**
 * Can be used to determine if an object already exists in the queue.
 *
 * @return YES if the object exists in the queue. NO if the object does not exist in the queue.
 */
- (BOOL)contains:(id<NSObject>)object;

/**
 * Can be used to determine the number of objects in the queue.
 *
 * @return The number of objects in the queue.
 */
- (NSUInteger)size;

/**
 * Removes all the elements of the priority queue.
 */
- (void)clear;

/**
 * Adds the specified object to the priority queue.
 *
 * @param object The object to be added.
 * @throws NSInternalInconsistencyException If the element cannot be compared with the elements in the priority queue using the ordering of the priority queue.
 * @throws NSInvalidArgumentException If object is null.
 */
- (void)add:(id<NSObject>)object;

/**
 * Removes the specified object from the priority queue.
 *
 * @param object The object to be removed.
 */
- (void)remove:(id<NSObject>)object;

/**
 * Gets but does not remove the head of the queue.
 *
 * @return the head of the queue or null if the queue is empty.
 */
- (id<NSObject>)peek;

/**
 * Gets and removes the head of the queue.
 *
 * @return the head of the queue or null if the queue is empty.
 */
- (id<NSObject>)poll;

/**
 * Creates and returns an NSArray from the contents of the queue.
 *
 * @return An array of all elements in the queue.
 */
- (NSMutableArray<caption *>*)toArray;

@end
