#ifndef __SYS_LOCK_H__
#define __SYS_LOCK_H__

#if defined(__cplusplus) || defined(__XC__)
extern "C" {
#endif

typedef int _LOCK_SIMPLE_T;

typedef struct {
  // Counter used to allocated ticket numbers.
  unsigned _counter;
  // Ticket number of the lock owner (if the lock is held) or the next owner
  // (if lock isn't currently held).
  unsigned _owner;
} _LOCK_FAIR_T;

typedef struct {
  int _owner;
  int _count;
} _LOCK_RECURSIVE_T;

#define __LOCK_SIMPLE_INIT(class, lock) class _LOCK_SIMPLE_T lock = 0;
#define __LOCK_FAIR_INIT(class, lock) class _LOCK_FAIR_T lock = {0, 0};
#define __LOCK_RECURSIVE_INIT(class, lock) class _LOCK_RECURSIVE_T lock = {-1, 0};

#define __LOCK_SIMPLE_INIT_ACQUIRED(class, lock) class _LOCK_SIMPLE_T lock = 1;
#define __LOCK_FAIR_INIT_ACQUIRED(class, lock) class _LOCK_FAIR_T lock = {1, 0};

void __lock_simple_init(volatile _LOCK_SIMPLE_T *);
void __lock_simple_close(volatile _LOCK_SIMPLE_T *);
void __lock_simple_acquire(volatile _LOCK_SIMPLE_T *);
int  __lock_simple_try_acquire(volatile _LOCK_SIMPLE_T *);
void __lock_simple_release(volatile _LOCK_SIMPLE_T *);

void __lock_fair_init(volatile _LOCK_FAIR_T *);
void __lock_fair_close(volatile _LOCK_FAIR_T *);
void __lock_fair_acquire(volatile _LOCK_FAIR_T *);
int  __lock_fair_try_acquire(volatile _LOCK_FAIR_T *);
void __lock_fair_release(volatile _LOCK_FAIR_T *);

void __lock_recursive_init(volatile _LOCK_RECURSIVE_T *);
void __lock_recursive_close(volatile _LOCK_RECURSIVE_T *);
void __lock_recursive_acquire(volatile _LOCK_RECURSIVE_T *);
int  __lock_recursive_try_acquire(volatile _LOCK_RECURSIVE_T *);
void __lock_recursive_release(volatile _LOCK_RECURSIVE_T *);

typedef _LOCK_FAIR_T _LOCK_T;

#define __LOCK_INIT(class, lock) __LOCK_FAIR_INIT(class, lock)
#define __LOCK_INIT_ACQUIRED(class, lock) __LOCK_FAIR_INIT_ACQUIRED(class, lock)

#define __lock_init(lock) __lock_fair_init(&(lock))
#define __lock_close(lock) __lock_fair_close(&(lock))
#define __lock_acquire(lock) __lock_fair_acquire(&(lock))
#define __lock_try_acquire(lock) __lock_fair_try_acquire(&(lock))
#define __lock_release(lock) __lock_fair_release(&(lock))

#define __LOCK_INIT_RECURSIVE(class, lock) __LOCK_RECURSIVE_INIT(class, lock)

#define __lock_init_recursive(lock) __lock_recursive_init(&(lock))
#define __lock_close_recursive(lock) __lock_recursive_close(&(lock))
#define __lock_acquire_recursive(lock) __lock_recursive_acquire(&(lock))
#define __lock_try_acquire_recursive(lock) __lock_recursive_try_acquire(&(lock))
#define __lock_release_recursive(lock) __lock_recursive_release(&(lock))

#if defined(__cplusplus) || defined(__XC__)
};
#endif

#endif /* __SYS_LOCK_H__ */
