#ifndef XCORE_ATOMIC_H_INCLUDED
#define XCORE_ATOMIC_H_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif

extern int xcore_sync_fetch_and_add (int *ptr, int val);
extern int xcore_sync_fetch_and_sub (int *ptr, int val);
extern int xcore_sync_fetch_and_or (int *ptr, int val);
extern int xcore_sync_fetch_and_and (int *ptr, int val);
extern int xcore_sync_fetch_and_xor (int *ptr, int val);
extern int xcore_sync_fetch_and_nand (int *ptr, int val);
extern int xcore_sync_add_and_fetch (int *ptr, int val);
extern int xcore_sync_sub_and_fetch (int *ptr, int val);
extern int xcore_sync_or_and_fetch (int *ptr, int val);
extern int xcore_sync_and_and_fetch (int *ptr, int val);
extern int xcore_sync_xor_and_fetch (int *ptr, int val);
extern int xcore_sync_nand_and_fetch (int *ptr, int val);
extern int xcore_sync_bool_compare_and_swap (int *ptr, int oldval, int newval);
extern int xcore_sync_val_compare_and_swap (int *ptr, int oldval, int newval);
extern void xcore_sync_synchronize (void);
extern int xcore_sync_lock_test_and_set (int *ptr, int val);
extern void xcore_sync_lock_release (int *ptr);

#ifndef sched_yield
#define sched_yield() /* do-nothing */
#endif

#ifdef __cplusplus
}
#endif

#endif // XCORE_ATOMIC_H_INCLUDED
