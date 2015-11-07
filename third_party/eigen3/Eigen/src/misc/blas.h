#ifndef BLAS_H
#define BLAS_H

#ifdef __cplusplus
extern "C"
{
#endif

#define BLASFUNC(FUNC) FUNC##_

#ifdef __WIN64__
typedef long long BLASLONG;
typedef unsigned long long BLASULONG;
#else
typedef long BLASLONG;
typedef unsigned long BLASULONG;
#endif

int    BLASFUNC(xerbla)(const char *, int *info, int);

float  BLASFUNC(sdot)  (int *, float  *, int *, float  *, int *);
float  BLASFUNC(sdsdot)(int *, float  *,        float  *, int *, float  *, int *);

double BLASFUNC(dsdot) (int *, float  *, int *, float  *, int *);
double BLASFUNC(ddot)  (int *, double *, int *, double *, int *);
double BLASFUNC(qdot)  (int *, double *, int *, double *, int *);

int  BLASFUNC(cdotuw)  (int *, float  *, int *, float  *, int *, float*);
int  BLASFUNC(cdotcw)  (int *, float  *, int *, float  *, int *, float*);
int  BLASFUNC(zdotuw)  (int *, double  *, int *, double  *, int *, double*);
int  BLASFUNC(zdotcw)  (int *, double  *, int *, double  *, int *, double*);

int    BLASFUNC(saxpy) (int *, float  *, float  *, int *, float  *, int *);
int    BLASFUNC(daxpy) (int *, double *, double *, int *, double *, int *);
int    BLASFUNC(qaxpy) (int *, double *, double *, int *, double *, int *);
int    BLASFUNC(caxpy) (int *, float  *, float  *, int *, float  *, int *);
int    BLASFUNC(zaxpy) (int *, double *, double *, int *, double *, int *);
int    BLASFUNC(xaxpy) (int *, double *, double *, int *, double *, int *);
int    BLASFUNC(caxpyc)(int *, float  *, float  *, int *, float  *, int *);
int    BLASFUNC(zaxpyc)(int *, double *, double *, int *, double *, int *);
int    BLASFUNC(xaxpyc)(int *, double *, double *, int *, double *, int *);

int    BLASFUNC(scopy) (int *, float  *, int *, float  *, int *);
int    BLASFUNC(dcopy) (int *, double *, int *, double *, int *);
int    BLASFUNC(qcopy) (int *, double *, int *, double *, int *);
int    BLASFUNC(ccopy) (int *, float  *, int *, float  *, int *);
int    BLASFUNC(zcopy) (int *, double *, int *, double *, int *);
int    BLASFUNC(xcopy) (int *, double *, int *, double *, int *);

int    BLASFUNC(sswap) (int *, float  *, int *, float  *, int *);
int    BLASFUNC(dswap) (int *, double *, int *, double *, int *);
int    BLASFUNC(qswap) (int *, double *, int *, double *, int *);
int    BLASFUNC(cswap) (int *, float  *, int *, float  *, int *);
int    BLASFUNC(zswap) (int *, double *, int *, double *, int *);
int    BLASFUNC(xswap) (int *, double *, int *, double *, int *);

float  BLASFUNC(sasum) (int *, float  *, int *);
float  BLASFUNC(scasum)(int *, float  *, int *);
double BLASFUNC(dasum) (int *, double *, int *);
double BLASFUNC(qasum) (int *, double *, int *);
double BLASFUNC(dzasum)(int *, double *, int *);
double BLASFUNC(qxasum)(int *, double *, int *);

int    BLASFUNC(isamax)(int *, float  *, int *);
int    BLASFUNC(idamax)(int *, double *, int *);
int    BLASFUNC(iqamax)(int *, double *, int *);
int    BLASFUNC(icamax)(int *, float  *, int *);
int    BLASFUNC(izamax)(int *, double *, int *);
int    BLASFUNC(ixamax)(int *, double *, int *);

int    BLASFUNC(ismax) (int *, float  *, int *);
int    BLASFUNC(idmax) (int *, double *, int *);
int    BLASFUNC(iqmax) (int *, double *, int *);
int    BLASFUNC(icmax) (int *, float  *, int *);
int    BLASFUNC(izmax) (int *, double *, int *);
int    BLASFUNC(ixmax) (int *, double *, int *);

int    BLASFUNC(isamin)(int *, float  *, int *);
int    BLASFUNC(idamin)(int *, double *, int *);
int    BLASFUNC(iqamin)(int *, double *, int *);
int    BLASFUNC(icamin)(int *, float  *, int *);
int    BLASFUNC(izamin)(int *, double *, int *);
int    BLASFUNC(ixamin)(int *, double *, int *);

int    BLASFUNC(ismin)(int *, float  *, int *);
int    BLASFUNC(idmin)(int *, double *, int *);
int    BLASFUNC(iqmin)(int *, double *, int *);
int    BLASFUNC(icmin)(int *, float  *, int *);
int    BLASFUNC(izmin)(int *, double *, int *);
int    BLASFUNC(ixmin)(int *, double *, int *);

float  BLASFUNC(samax) (int *, float  *, int *);
double BLASFUNC(damax) (int *, double *, int *);
double BLASFUNC(qamax) (int *, double *, int *);
float  BLASFUNC(scamax)(int *, float  *, int *);
double BLASFUNC(dzamax)(int *, double *, int *);
double BLASFUNC(qxamax)(int *, double *, int *);

float  BLASFUNC(samin) (int *, float  *, int *);
double BLASFUNC(damin) (int *, double *, int *);
double BLASFUNC(qamin) (int *, double *, int *);
float  BLASFUNC(scamin)(int *, float  *, int *);
double BLASFUNC(dzamin)(int *, double *, int *);
double BLASFUNC(qxamin)(int *, double *, int *);

float  BLASFUNC(smax)  (int *, float  *, int *);
double BLASFUNC(dmax)  (int *, double *, int *);
double BLASFUNC(qmax)  (int *, double *, int *);
float  BLASFUNC(scmax) (int *, float  *, int *);
double BLASFUNC(dzmax) (int *, double *, int *);
double BLASFUNC(qxmax) (int *, double *, int *);

float  BLASFUNC(smin)  (int *, float  *, int *);
double BLASFUNC(dmin)  (int *, double *, int *);
double BLASFUNC(qmin)  (int *, double *, int *);
float  BLASFUNC(scmin) (int *, float  *, int *);
double BLASFUNC(dzmin) (int *, double *, int *);
double BLASFUNC(qxmin) (int *, double *, int *);

int    BLASFUNC(sscal) (int *,  float  *, float  *, int *);
int    BLASFUNC(dscal) (int *,  double *, double *, int *);
int    BLASFUNC(qscal) (int *,  double *, double *, int *);
int    BLASFUNC(cscal) (int *,  float  *, float  *, int *);
int    BLASFUNC(zscal) (int *,  double *, double *, int *);
int    BLASFUNC(xscal) (int *,  double *, double *, int *);
int    BLASFUNC(csscal)(int *,  float  *, float  *, int *);
int    BLASFUNC(zdscal)(int *,  double *, double *, int *);
int    BLASFUNC(xqscal)(int *,  double *, double *, int *);

float  BLASFUNC(snrm2) (int *, float  *, int *);
float  BLASFUNC(scnrm2)(int *, float  *, int *);

double BLASFUNC(dnrm2) (int *, double *, int *);
double BLASFUNC(qnrm2) (int *, double *, int *);
double BLASFUNC(dznrm2)(int *, double *, int *);
double BLASFUNC(qxnrm2)(int *, double *, int *);

int    BLASFUNC(srot)  (int *, float  *, int *, float  *, int *, float  *, float  *);
int    BLASFUNC(drot)  (int *, double *, int *, double *, int *, double *, double *);
int    BLASFUNC(qrot)  (int *, double *, int *, double *, int *, double *, double *);
int    BLASFUNC(csrot) (int *, float  *, int *, float  *, int *, float  *, float  *);
int    BLASFUNC(zdrot) (int *, double *, int *, double *, int *, double *, double *);
int    BLASFUNC(xqrot) (int *, double *, int *, double *, int *, double *, double *);

int    BLASFUNC(srotg) (float  *, float  *, float  *, float  *);
int    BLASFUNC(drotg) (double *, double *, double *, double *);
int    BLASFUNC(qrotg) (double *, double *, double *, double *);
int    BLASFUNC(crotg) (float  *, float  *, float  *, float  *);
int    BLASFUNC(zrotg) (double *, double *, double *, double *);
int    BLASFUNC(xrotg) (double *, double *, double *, double *);

int    BLASFUNC(srotmg)(float  *, float  *, float  *, float  *, float  *);
int    BLASFUNC(drotmg)(double *, double *, double *, double *, double *);

int    BLASFUNC(srotm) (int *, float  *, int *, float  *, int *, float  *);
int    BLASFUNC(drotm) (int *, double *, int *, double *, int *, double *);
int    BLASFUNC(qrotm) (int *, double *, int *, double *, int *, double *);

/* Level 2 routines */

int BLASFUNC(sger)(int *,    int *, float *,  float *, int *,
		   float *,  int *, float *,  int *);
int BLASFUNC(dger)(int *,    int *, double *, double *, int *,
		   double *, int *, double *, int *);
int BLASFUNC(qger)(int *,    int *, double *, double *, int *,
		   double *, int *, double *, int *);
int BLASFUNC(cgeru)(int *,    int *, float *,  float *, int *,
		    float *,  int *, float *,  int *);
int BLASFUNC(cgerc)(int *,    int *, float *,  float *, int *,
		    float *,  int *, float *,  int *);
int BLASFUNC(zgeru)(int *,    int *, double *, double *, int *,
		    double *, int *, double *, int *);
int BLASFUNC(zgerc)(int *,    int *, double *, double *, int *,
		    double *, int *, double *, int *);
int BLASFUNC(xgeru)(int *,    int *, double *, double *, int *,
		    double *, int *, double *, int *);
int BLASFUNC(xgerc)(int *,    int *, double *, double *, int *,
		    double *, int *, double *, int *);

int BLASFUNC(sgemv)(char *, int *, int *, float  *, float  *, int *,
		    float  *, int *, float  *, float  *, int *);
int BLASFUNC(dgemv)(char *, int *, int *, double *, double *, int *,
		    double *, int *, double *, double *, int *);
int BLASFUNC(qgemv)(char *, int *, int *, double *, double *, int *,
		    double *, int *, double *, double *, int *);
int BLASFUNC(cgemv)(char *, int *, int *, float  *, float  *, int *,
		    float  *, int *, float  *, float  *, int *);
int BLASFUNC(zgemv)(char *, int *, int *, double *, double *, int *,
		    double *, int *, double *, double *, int *);
int BLASFUNC(xgemv)(char *, int *, int *, double *, double *, int *,
		    double *, int *, double *, double *, int *);

int BLASFUNC(strsv) (char *, char *, char *, int *, float  *, int *,
		     float  *, int *);
int BLASFUNC(dtrsv) (char *, char *, char *, int *, double *, int *,
		     double *, int *);
int BLASFUNC(qtrsv) (char *, char *, char *, int *, double *, int *,
		     double *, int *);
int BLASFUNC(ctrsv) (char *, char *, char *, int *, float  *, int *,
		     float  *, int *);
int BLASFUNC(ztrsv) (char *, char *, char *, int *, double *, int *,
		     double *, int *);
int BLASFUNC(xtrsv) (char *, char *, char *, int *, double *, int *,
		     double *, int *);

int BLASFUNC(stpsv) (char *, char *, char *, int *, float  *, float  *, int *);
int BLASFUNC(dtpsv) (char *, char *, char *, int *, double *, double *, int *);
int BLASFUNC(qtpsv) (char *, char *, char *, int *, double *, double *, int *);
int BLASFUNC(ctpsv) (char *, char *, char *, int *, float  *, float  *, int *);
int BLASFUNC(ztpsv) (char *, char *, char *, int *, double *, double *, int *);
int BLASFUNC(xtpsv) (char *, char *, char *, int *, double *, double *, int *);

int BLASFUNC(strmv) (char *, char *, char *, int *, float  *, int *,
		     float  *, int *);
int BLASFUNC(dtrmv) (char *, char *, char *, int *, double *, int *,
		     double *, int *);
int BLASFUNC(qtrmv) (char *, char *, char *, int *, double *, int *,
		     double *, int *);
int BLASFUNC(ctrmv) (char *, char *, char *, int *, float  *, int *,
		     float  *, int *);
int BLASFUNC(ztrmv) (char *, char *, char *, int *, double *, int *,
		     double *, int *);
int BLASFUNC(xtrmv) (char *, char *, char *, int *, double *, int *,
		     double *, int *);

int BLASFUNC(stpmv) (char *, char *, char *, int *, float  *, float  *, int *);
int BLASFUNC(dtpmv) (char *, char *, char *, int *, double *, double *, int *);
int BLASFUNC(qtpmv) (char *, char *, char *, int *, double *, double *, int *);
int BLASFUNC(ctpmv) (char *, char *, char *, int *, float  *, float  *, int *);
int BLASFUNC(ztpmv) (char *, char *, char *, int *, double *, double *, int *);
int BLASFUNC(xtpmv) (char *, char *, char *, int *, double *, double *, int *);

int BLASFUNC(stbmv) (char *, char *, char *, int *, int *, float  *, int *, float  *, int *);
int BLASFUNC(dtbmv) (char *, char *, char *, int *, int *, double *, int *, double *, int *);
int BLASFUNC(qtbmv) (char *, char *, char *, int *, int *, double *, int *, double *, int *);
int BLASFUNC(ctbmv) (char *, char *, char *, int *, int *, float  *, int *, float  *, int *);
int BLASFUNC(ztbmv) (char *, char *, char *, int *, int *, double *, int *, double *, int *);
int BLASFUNC(xtbmv) (char *, char *, char *, int *, int *, double *, int *, double *, int *);

int BLASFUNC(stbsv) (char *, char *, char *, int *, int *, float  *, int *, float  *, int *);
int BLASFUNC(dtbsv) (char *, char *, char *, int *, int *, double *, int *, double *, int *);
int BLASFUNC(qtbsv) (char *, char *, char *, int *, int *, double *, int *, double *, int *);
int BLASFUNC(ctbsv) (char *, char *, char *, int *, int *, float  *, int *, float  *, int *);
int BLASFUNC(ztbsv) (char *, char *, char *, int *, int *, double *, int *, double *, int *);
int BLASFUNC(xtbsv) (char *, char *, char *, int *, int *, double *, int *, double *, int *);

int BLASFUNC(ssymv) (char *, int *, float  *, float *, int *,
		     float  *, int *, float *, float *, int *);
int BLASFUNC(dsymv) (char *, int *, double  *, double *, int *,
		     double  *, int *, double *, double *, int *);
int BLASFUNC(qsymv) (char *, int *, double  *, double *, int *,
		     double  *, int *, double *, double *, int *);
int BLASFUNC(csymv) (char *, int *, float  *, float *, int *,
		     float  *, int *, float *, float *, int *);
int BLASFUNC(zsymv) (char *, int *, double  *, double *, int *,
		     double  *, int *, double *, double *, int *);
int BLASFUNC(xsymv) (char *, int *, double  *, double *, int *,
		     double  *, int *, double *, double *, int *);

int BLASFUNC(sspmv) (char *, int *, float  *, float *,
		     float  *, int *, float *, float *, int *);
int BLASFUNC(dspmv) (char *, int *, double  *, double *,
		     double  *, int *, double *, double *, int *);
int BLASFUNC(qspmv) (char *, int *, double  *, double *,
		     double  *, int *, double *, double *, int *);
int BLASFUNC(cspmv) (char *, int *, float  *, float *,
		     float  *, int *, float *, float *, int *);
int BLASFUNC(zspmv) (char *, int *, double  *, double *,
		     double  *, int *, double *, double *, int *);
int BLASFUNC(xspmv) (char *, int *, double  *, double *,
		     double  *, int *, double *, double *, int *);

int BLASFUNC(ssyr) (char *, int *, float   *, float  *, int *,
		    float  *, int *);
int BLASFUNC(dsyr) (char *, int *, double  *, double *, int *,
		    double *, int *);
int BLASFUNC(qsyr) (char *, int *, double  *, double *, int *,
		    double *, int *);
int BLASFUNC(csyr) (char *, int *, float   *, float  *, int *,
		    float  *, int *);
int BLASFUNC(zsyr) (char *, int *, double  *, double *, int *,
		    double *, int *);
int BLASFUNC(xsyr) (char *, int *, double  *, double *, int *,
		    double *, int *);

int BLASFUNC(ssyr2) (char *, int *, float   *,
		     float  *, int *, float  *, int *, float  *, int *);
int BLASFUNC(dsyr2) (char *, int *, double  *,
		     double *, int *, double *, int *, double *, int *);
int BLASFUNC(qsyr2) (char *, int *, double  *,
		     double *, int *, double *, int *, double *, int *);
int BLASFUNC(csyr2) (char *, int *, float   *,
		     float  *, int *, float  *, int *, float  *, int *);
int BLASFUNC(zsyr2) (char *, int *, double  *,
		     double *, int *, double *, int *, double *, int *);
int BLASFUNC(xsyr2) (char *, int *, double  *,
		     double *, int *, double *, int *, double *, int *);

int BLASFUNC(sspr) (char *, int *, float   *, float  *, int *,
		    float  *);
int BLASFUNC(dspr) (char *, int *, double  *, double *, int *,
		    double *);
int BLASFUNC(qspr) (char *, int *, double  *, double *, int *,
		    double *);
int BLASFUNC(cspr) (char *, int *, float   *, float  *, int *,
		    float  *);
int BLASFUNC(zspr) (char *, int *, double  *, double *, int *,
		    double *);
int BLASFUNC(xspr) (char *, int *, double  *, double *, int *,
		    double *);

int BLASFUNC(sspr2) (char *, int *, float   *,
		     float  *, int *, float  *, int *, float  *);
int BLASFUNC(dspr2) (char *, int *, double  *,
		     double *, int *, double *, int *, double *);
int BLASFUNC(qspr2) (char *, int *, double  *,
		     double *, int *, double *, int *, double *);
int BLASFUNC(cspr2) (char *, int *, float   *,
		     float  *, int *, float  *, int *, float  *);
int BLASFUNC(zspr2) (char *, int *, double  *,
		     double *, int *, double *, int *, double *);
int BLASFUNC(xspr2) (char *, int *, double  *,
		     double *, int *, double *, int *, double *);

int BLASFUNC(cher) (char *, int *, float   *, float  *, int *,
		    float  *, int *);
int BLASFUNC(zher) (char *, int *, double  *, double *, int *,
		    double *, int *);
int BLASFUNC(xher) (char *, int *, double  *, double *, int *,
		    double *, int *);

int BLASFUNC(chpr) (char *, int *, float   *, float  *, int *, float  *);
int BLASFUNC(zhpr) (char *, int *, double  *, double *, int *, double *);
int BLASFUNC(xhpr) (char *, int *, double  *, double *, int *, double *);

int BLASFUNC(cher2) (char *, int *, float   *,
		     float  *, int *, float  *, int *, float  *, int *);
int BLASFUNC(zher2) (char *, int *, double  *,
		     double *, int *, double *, int *, double *, int *);
int BLASFUNC(xher2) (char *, int *, double  *,
		     double *, int *, double *, int *, double *, int *);

int BLASFUNC(chpr2) (char *, int *, float   *,
		     float  *, int *, float  *, int *, float  *);
int BLASFUNC(zhpr2) (char *, int *, double  *,
		     double *, int *, double *, int *, double *);
int BLASFUNC(xhpr2) (char *, int *, double  *,
		     double *, int *, double *, int *, double *);

int BLASFUNC(chemv) (char *, int *, float  *, float *, int *,
		     float  *, int *, float *, float *, int *);
int BLASFUNC(zhemv) (char *, int *, double  *, double *, int *,
		     double  *, int *, double *, double *, int *);
int BLASFUNC(xhemv) (char *, int *, double  *, double *, int *,
		     double  *, int *, double *, double *, int *);

int BLASFUNC(chpmv) (char *, int *, float  *, float *,
		     float  *, int *, float *, float *, int *);
int BLASFUNC(zhpmv) (char *, int *, double  *, double *,
		     double  *, int *, double *, double *, int *);
int BLASFUNC(xhpmv) (char *, int *, double  *, double *,
		     double  *, int *, double *, double *, int *);

int BLASFUNC(snorm)(char *, int *, int *, float  *, int *);
int BLASFUNC(dnorm)(char *, int *, int *, double *, int *);
int BLASFUNC(cnorm)(char *, int *, int *, float  *, int *);
int BLASFUNC(znorm)(char *, int *, int *, double *, int *);

int BLASFUNC(sgbmv)(char *, int *, int *, int *, int *, float  *, float  *, int *,
		    float  *, int *, float  *, float  *, int *);
int BLASFUNC(dgbmv)(char *, int *, int *, int *, int *, double *, double *, int *,
		    double *, int *, double *, double *, int *);
int BLASFUNC(qgbmv)(char *, int *, int *, int *, int *, double *, double *, int *,
		    double *, int *, double *, double *, int *);
int BLASFUNC(cgbmv)(char *, int *, int *, int *, int *, float  *, float  *, int *,
		    float  *, int *, float  *, float  *, int *);
int BLASFUNC(zgbmv)(char *, int *, int *, int *, int *, double *, double *, int *,
		    double *, int *, double *, double *, int *);
int BLASFUNC(xgbmv)(char *, int *, int *, int *, int *, double *, double *, int *,
		    double *, int *, double *, double *, int *);

int BLASFUNC(ssbmv)(char *, int *, int *, float  *, float  *, int *,
		    float  *, int *, float  *, float  *, int *);
int BLASFUNC(dsbmv)(char *, int *, int *, double *, double *, int *,
		    double *, int *, double *, double *, int *);
int BLASFUNC(qsbmv)(char *, int *, int *, double *, double *, int *,
		    double *, int *, double *, double *, int *);
int BLASFUNC(csbmv)(char *, int *, int *, float  *, float  *, int *,
		    float  *, int *, float  *, float  *, int *);
int BLASFUNC(zsbmv)(char *, int *, int *, double *, double *, int *,
		    double *, int *, double *, double *, int *);
int BLASFUNC(xsbmv)(char *, int *, int *, double *, double *, int *,
		    double *, int *, double *, double *, int *);

int BLASFUNC(chbmv)(char *, int *, int *, float  *, float  *, int *,
		    float  *, int *, float  *, float  *, int *);
int BLASFUNC(zhbmv)(char *, int *, int *, double *, double *, int *,
		    double *, int *, double *, double *, int *);
int BLASFUNC(xhbmv)(char *, int *, int *, double *, double *, int *,
		    double *, int *, double *, double *, int *);

/* Level 3 routines */

int BLASFUNC(sgemm)(char *, char *, int *, int *, int *, float *,
	   float  *, int *, float  *, int *, float  *, float  *, int *);
int BLASFUNC(dgemm)(char *, char *, int *, int *, int *, double *,
	   double *, int *, double *, int *, double *, double *, int *);
int BLASFUNC(qgemm)(char *, char *, int *, int *, int *, double *,
	   double *, int *, double *, int *, double *, double *, int *);
int BLASFUNC(cgemm)(char *, char *, int *, int *, int *, float *,
	   float  *, int *, float  *, int *, float  *, float  *, int *);
int BLASFUNC(zgemm)(char *, char *, int *, int *, int *, double *,
	   double *, int *, double *, int *, double *, double *, int *);
int BLASFUNC(xgemm)(char *, char *, int *, int *, int *, double *,
	   double *, int *, double *, int *, double *, double *, int *);

int BLASFUNC(cgemm3m)(char *, char *, int *, int *, int *, float *,
	   float  *, int *, float  *, int *, float  *, float  *, int *);
int BLASFUNC(zgemm3m)(char *, char *, int *, int *, int *, double *,
	   double *, int *, double *, int *, double *, double *, int *);
int BLASFUNC(xgemm3m)(char *, char *, int *, int *, int *, double *,
	   double *, int *, double *, int *, double *, double *, int *);

int BLASFUNC(sge2mm)(char *, char *, char *, int *, int *,
		     float *, float  *, int *, float  *, int *,
		     float *, float  *, int *);
int BLASFUNC(dge2mm)(char *, char *, char *, int *, int *,
		     double *, double  *, int *, double  *, int *,
		     double *, double  *, int *);
int BLASFUNC(cge2mm)(char *, char *, char *, int *, int *,
		     float *, float  *, int *, float  *, int *,
		     float *, float  *, int *);
int BLASFUNC(zge2mm)(char *, char *, char *, int *, int *,
		     double *, double  *, int *, double  *, int *,
		     double *, double  *, int *);

int BLASFUNC(strsm)(char *, char *, char *, char *, int *, int *,
	   float *,  float *, int *, float *, int *);
int BLASFUNC(dtrsm)(char *, char *, char *, char *, int *, int *,
	   double *,  double *, int *, double *, int *);
int BLASFUNC(qtrsm)(char *, char *, char *, char *, int *, int *,
	   double *,  double *, int *, double *, int *);
int BLASFUNC(ctrsm)(char *, char *, char *, char *, int *, int *,
	   float *,  float *, int *, float *, int *);
int BLASFUNC(ztrsm)(char *, char *, char *, char *, int *, int *,
	   double *,  double *, int *, double *, int *);
int BLASFUNC(xtrsm)(char *, char *, char *, char *, int *, int *,
	   double *,  double *, int *, double *, int *);

int BLASFUNC(strmm)(char *, char *, char *, char *, int *, int *,
	   float *,  float *, int *, float *, int *);
int BLASFUNC(dtrmm)(char *, char *, char *, char *, int *, int *,
	   double *,  double *, int *, double *, int *);
int BLASFUNC(qtrmm)(char *, char *, char *, char *, int *, int *,
	   double *,  double *, int *, double *, int *);
int BLASFUNC(ctrmm)(char *, char *, char *, char *, int *, int *,
	   float *,  float *, int *, float *, int *);
int BLASFUNC(ztrmm)(char *, char *, char *, char *, int *, int *,
	   double *,  double *, int *, double *, int *);
int BLASFUNC(xtrmm)(char *, char *, char *, char *, int *, int *,
	   double *,  double *, int *, double *, int *);

int BLASFUNC(ssymm)(char *, char *, int *, int *, float  *, float  *, int *,
	   float  *, int *, float  *, float  *, int *);
int BLASFUNC(dsymm)(char *, char *, int *, int *, double *, double *, int *,
	   double *, int *, double *, double *, int *);
int BLASFUNC(qsymm)(char *, char *, int *, int *, double *, double *, int *,
	   double *, int *, double *, double *, int *);
int BLASFUNC(csymm)(char *, char *, int *, int *, float  *, float  *, int *,
	   float  *, int *, float  *, float  *, int *);
int BLASFUNC(zsymm)(char *, char *, int *, int *, double *, double *, int *,
	   double *, int *, double *, double *, int *);
int BLASFUNC(xsymm)(char *, char *, int *, int *, double *, double *, int *,
	   double *, int *, double *, double *, int *);

int BLASFUNC(csymm3m)(char *, char *, int *, int *, float  *, float  *, int *,
	   float  *, int *, float  *, float  *, int *);
int BLASFUNC(zsymm3m)(char *, char *, int *, int *, double *, double *, int *,
	   double *, int *, double *, double *, int *);
int BLASFUNC(xsymm3m)(char *, char *, int *, int *, double *, double *, int *,
	   double *, int *, double *, double *, int *);

int BLASFUNC(ssyrk)(char *, char *, int *, int *, float  *, float  *, int *,
	   float  *, float  *, int *);
int BLASFUNC(dsyrk)(char *, char *, int *, int *, double *, double *, int *,
	   double *, double *, int *);
int BLASFUNC(qsyrk)(char *, char *, int *, int *, double *, double *, int *,
	   double *, double *, int *);
int BLASFUNC(csyrk)(char *, char *, int *, int *, float  *, float  *, int *,
	   float  *, float  *, int *);
int BLASFUNC(zsyrk)(char *, char *, int *, int *, double *, double *, int *,
	   double *, double *, int *);
int BLASFUNC(xsyrk)(char *, char *, int *, int *, double *, double *, int *,
	   double *, double *, int *);

int BLASFUNC(ssyr2k)(char *, char *, int *, int *, float  *, float  *, int *,
	   float *, int *, float  *, float  *, int *);
int BLASFUNC(dsyr2k)(char *, char *, int *, int *, double *, double *, int *,
	   double*, int *, double *, double *, int *);
int BLASFUNC(qsyr2k)(char *, char *, int *, int *, double *, double *, int *,
	   double*, int *, double *, double *, int *);
int BLASFUNC(csyr2k)(char *, char *, int *, int *, float  *, float  *, int *,
	   float *, int *, float  *, float  *, int *);
int BLASFUNC(zsyr2k)(char *, char *, int *, int *, double *, double *, int *,
	   double*, int *, double *, double *, int *);
int BLASFUNC(xsyr2k)(char *, char *, int *, int *, double *, double *, int *,
	   double*, int *, double *, double *, int *);

int BLASFUNC(chemm)(char *, char *, int *, int *, float  *, float  *, int *,
	   float  *, int *, float  *, float  *, int *);
int BLASFUNC(zhemm)(char *, char *, int *, int *, double *, double *, int *,
	   double *, int *, double *, double *, int *);
int BLASFUNC(xhemm)(char *, char *, int *, int *, double *, double *, int *,
	   double *, int *, double *, double *, int *);

int BLASFUNC(chemm3m)(char *, char *, int *, int *, float  *, float  *, int *,
	   float  *, int *, float  *, float  *, int *);
int BLASFUNC(zhemm3m)(char *, char *, int *, int *, double *, double *, int *,
	   double *, int *, double *, double *, int *);
int BLASFUNC(xhemm3m)(char *, char *, int *, int *, double *, double *, int *,
	   double *, int *, double *, double *, int *);

int BLASFUNC(cherk)(char *, char *, int *, int *, float  *, float  *, int *,
	   float  *, float  *, int *);
int BLASFUNC(zherk)(char *, char *, int *, int *, double *, double *, int *,
	   double *, double *, int *);
int BLASFUNC(xherk)(char *, char *, int *, int *, double *, double *, int *,
	   double *, double *, int *);

int BLASFUNC(cher2k)(char *, char *, int *, int *, float  *, float  *, int *,
	   float *, int *, float  *, float  *, int *);
int BLASFUNC(zher2k)(char *, char *, int *, int *, double *, double *, int *,
	   double*, int *, double *, double *, int *);
int BLASFUNC(xher2k)(char *, char *, int *, int *, double *, double *, int *,
	   double*, int *, double *, double *, int *);
int BLASFUNC(cher2m)(char *, char *, char *, int *, int *, float  *, float  *, int *,
	   float *, int *, float  *, float  *, int *);
int BLASFUNC(zher2m)(char *, char *, char *, int *, int *, double *, double *, int *,
	   double*, int *, double *, double *, int *);
int BLASFUNC(xher2m)(char *, char *, char *, int *, int *, double *, double *, int *,
	   double*, int *, double *, double *, int *);

int BLASFUNC(sgemt)(char *, int *, int *, float  *, float  *, int *,
		    float  *, int *);
int BLASFUNC(dgemt)(char *, int *, int *, double *, double *, int *,
		    double *, int *);
int BLASFUNC(cgemt)(char *, int *, int *, float  *, float  *, int *,
		    float  *, int *);
int BLASFUNC(zgemt)(char *, int *, int *, double *, double *, int *,
		    double *, int *);

int BLASFUNC(sgema)(char *, char *, int *, int *, float  *,
		    float  *, int *, float *, float  *, int *, float *, int *);
int BLASFUNC(dgema)(char *, char *, int *, int *, double *,
		    double *, int *, double*, double *, int *, double*, int *);
int BLASFUNC(cgema)(char *, char *, int *, int *, float  *,
		    float  *, int *, float *, float  *, int *, float *, int *);
int BLASFUNC(zgema)(char *, char *, int *, int *, double *,
		    double *, int *, double*, double *, int *, double*, int *);

int BLASFUNC(sgems)(char *, char *, int *, int *, float  *,
		    float  *, int *, float *, float  *, int *, float *, int *);
int BLASFUNC(dgems)(char *, char *, int *, int *, double *,
		    double *, int *, double*, double *, int *, double*, int *);
int BLASFUNC(cgems)(char *, char *, int *, int *, float  *,
		    float  *, int *, float *, float  *, int *, float *, int *);
int BLASFUNC(zgems)(char *, char *, int *, int *, double *,
		    double *, int *, double*, double *, int *, double*, int *);

int BLASFUNC(sgetf2)(int *, int *, float  *, int *, int *, int *);
int BLASFUNC(dgetf2)(int *, int *, double *, int *, int *, int *);
int BLASFUNC(qgetf2)(int *, int *, double *, int *, int *, int *);
int BLASFUNC(cgetf2)(int *, int *, float  *, int *, int *, int *);
int BLASFUNC(zgetf2)(int *, int *, double *, int *, int *, int *);
int BLASFUNC(xgetf2)(int *, int *, double *, int *, int *, int *);

int BLASFUNC(sgetrf)(int *, int *, float  *, int *, int *, int *);
int BLASFUNC(dgetrf)(int *, int *, double *, int *, int *, int *);
int BLASFUNC(qgetrf)(int *, int *, double *, int *, int *, int *);
int BLASFUNC(cgetrf)(int *, int *, float  *, int *, int *, int *);
int BLASFUNC(zgetrf)(int *, int *, double *, int *, int *, int *);
int BLASFUNC(xgetrf)(int *, int *, double *, int *, int *, int *);

int BLASFUNC(slaswp)(int *, float  *, int *, int *, int *, int *, int *);
int BLASFUNC(dlaswp)(int *, double *, int *, int *, int *, int *, int *);
int BLASFUNC(qlaswp)(int *, double *, int *, int *, int *, int *, int *);
int BLASFUNC(claswp)(int *, float  *, int *, int *, int *, int *, int *);
int BLASFUNC(zlaswp)(int *, double *, int *, int *, int *, int *, int *);
int BLASFUNC(xlaswp)(int *, double *, int *, int *, int *, int *, int *);

int BLASFUNC(sgetrs)(char *, int *, int *, float  *, int *, int *, float  *, int *, int *);
int BLASFUNC(dgetrs)(char *, int *, int *, double *, int *, int *, double *, int *, int *);
int BLASFUNC(qgetrs)(char *, int *, int *, double *, int *, int *, double *, int *, int *);
int BLASFUNC(cgetrs)(char *, int *, int *, float  *, int *, int *, float  *, int *, int *);
int BLASFUNC(zgetrs)(char *, int *, int *, double *, int *, int *, double *, int *, int *);
int BLASFUNC(xgetrs)(char *, int *, int *, double *, int *, int *, double *, int *, int *);

int BLASFUNC(sgesv)(int *, int *, float  *, int *, int *, float *, int *, int *);
int BLASFUNC(dgesv)(int *, int *, double *, int *, int *, double*, int *, int *);
int BLASFUNC(qgesv)(int *, int *, double *, int *, int *, double*, int *, int *);
int BLASFUNC(cgesv)(int *, int *, float  *, int *, int *, float *, int *, int *);
int BLASFUNC(zgesv)(int *, int *, double *, int *, int *, double*, int *, int *);
int BLASFUNC(xgesv)(int *, int *, double *, int *, int *, double*, int *, int *);

int BLASFUNC(spotf2)(char *, int *, float  *, int *, int *);
int BLASFUNC(dpotf2)(char *, int *, double *, int *, int *);
int BLASFUNC(qpotf2)(char *, int *, double *, int *, int *);
int BLASFUNC(cpotf2)(char *, int *, float  *, int *, int *);
int BLASFUNC(zpotf2)(char *, int *, double *, int *, int *);
int BLASFUNC(xpotf2)(char *, int *, double *, int *, int *);

int BLASFUNC(spotrf)(char *, int *, float  *, int *, int *);
int BLASFUNC(dpotrf)(char *, int *, double *, int *, int *);
int BLASFUNC(qpotrf)(char *, int *, double *, int *, int *);
int BLASFUNC(cpotrf)(char *, int *, float  *, int *, int *);
int BLASFUNC(zpotrf)(char *, int *, double *, int *, int *);
int BLASFUNC(xpotrf)(char *, int *, double *, int *, int *);

int BLASFUNC(slauu2)(char *, int *, float  *, int *, int *);
int BLASFUNC(dlauu2)(char *, int *, double *, int *, int *);
int BLASFUNC(qlauu2)(char *, int *, double *, int *, int *);
int BLASFUNC(clauu2)(char *, int *, float  *, int *, int *);
int BLASFUNC(zlauu2)(char *, int *, double *, int *, int *);
int BLASFUNC(xlauu2)(char *, int *, double *, int *, int *);

int BLASFUNC(slauum)(char *, int *, float  *, int *, int *);
int BLASFUNC(dlauum)(char *, int *, double *, int *, int *);
int BLASFUNC(qlauum)(char *, int *, double *, int *, int *);
int BLASFUNC(clauum)(char *, int *, float  *, int *, int *);
int BLASFUNC(zlauum)(char *, int *, double *, int *, int *);
int BLASFUNC(xlauum)(char *, int *, double *, int *, int *);

int BLASFUNC(strti2)(char *, char *, int *, float  *, int *, int *);
int BLASFUNC(dtrti2)(char *, char *, int *, double *, int *, int *);
int BLASFUNC(qtrti2)(char *, char *, int *, double *, int *, int *);
int BLASFUNC(ctrti2)(char *, char *, int *, float  *, int *, int *);
int BLASFUNC(ztrti2)(char *, char *, int *, double *, int *, int *);
int BLASFUNC(xtrti2)(char *, char *, int *, double *, int *, int *);

int BLASFUNC(strtri)(char *, char *, int *, float  *, int *, int *);
int BLASFUNC(dtrtri)(char *, char *, int *, double *, int *, int *);
int BLASFUNC(qtrtri)(char *, char *, int *, double *, int *, int *);
int BLASFUNC(ctrtri)(char *, char *, int *, float  *, int *, int *);
int BLASFUNC(ztrtri)(char *, char *, int *, double *, int *, int *);
int BLASFUNC(xtrtri)(char *, char *, int *, double *, int *, int *);

int BLASFUNC(spotri)(char *, int *, float  *, int *, int *);
int BLASFUNC(dpotri)(char *, int *, double *, int *, int *);
int BLASFUNC(qpotri)(char *, int *, double *, int *, int *);
int BLASFUNC(cpotri)(char *, int *, float  *, int *, int *);
int BLASFUNC(zpotri)(char *, int *, double *, int *, int *);
int BLASFUNC(xpotri)(char *, int *, double *, int *, int *);

#ifdef __cplusplus
}
#endif

#endif
