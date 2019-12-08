#pragma once

#include <iostream>
#include "stdafx.h"
#include <armadillo>
#include <complex>
#include <iomanip>
#include <cmath>
#include <fstream>
#include "estimation.h"
#include <nlopt.h>

using namespace std;
using namespace arma;

typedef Col<double> column_vector;
typedef Col<cx_double> column_vector_cx;
#define  j1 cx_double(0,1)

class PID{
public:
	PID(int axisFlag);
	static column_vector rigParamsDis;
	static column_vector resParamsDis;
	static column_vector_cx rigParamsCon;
	static column_vector_cx resParamsCon;
	static int hasRes;
	static int gonnaOverDamped;
	static int axFlag;
	static int checkRes();
	static int checkRes2();
	static int checkDamping();
	static column_vector PIDparams;
	static void initPIDparams();
	static column_vector_cx calcRigParamsCon(column_vector rigParamsDis);
	static column_vector_cx calcResParamsCon(column_vector resParamsDis);
	static double phaseRig(double w);
	static double phaseRigNew(double w);
	static double phaseRes(double w);
	static double phaseResWorst(double w);
	static double phasePID(double w);
	static double totPhase(double w);
	static double totPhaseNew(double w);
	static double totPhasePID(double w);
	static double totPhasePIDNew(double w);
	static void checkPhase();
	static double phase180Obj(unsigned numberOfParams, const double* params, double* grad, void* errorResCData);
	static double phase170OLObj(unsigned numberOfParams, const double* params, double* grad, void* errorResCData);
	static double phaseMaxPhaseOLObj(unsigned numberOfParams, const double* params, double* grad, void* errorResCData);
	static double phaseMaxPhaseOLResWorstObj(unsigned numberOfParams, const double* params, double* grad, void* errorResCData);
	static double magRig(double w);
	static double magRigNew(double w);
	static double magRes(double w);
	static double magResWorst(double w);
	static double magPID(double w);
	static double totMag(double w);
	static double totMagNew(double w);
	static double totMagPID(double w);
	static double totMagPIDNew(double w);
	static double magResWorstObj(unsigned numberOfParams, const double* params, double* grad, void* errorResCData);
	static double magResObj(unsigned numberOfParams, const double* params, double* grad, void* errorResCData);
	static void checkMag();
	static double modfn(double a, double b);
	static double constraintOpt(unsigned numberOfParams, const double* params, double* grad, void* ConstraintErrorData);
	static double find180Freq();
	static double find170FreqOL();
	static double findMaxPhaseFreqOL();
	static double findMaxPhaseFreqResWorstFreq();
	static double findMaxMagResWorstFreq();
	static double findMaxMagResFreq();
	static double findkdmin();
	static double findkdmax();
	static double a;
	static double b;
	static double p1;
	static double p2;
	static double k;
	static double dkv;
	static double z1;
	static double z2;
	static double kd;
	static double kdmin;
	static double kdmax;
	static double kdstart;
	static int tachFailFlag;
	static double omegar;
	static double omegad;
	static double *rigParamsAzConC;
	static double *resParamsAzConC;
	static double *rigParamsElConC;
	static double *resParamsElConC;
	static void setkd(double kd);
	static void setTuneParams();
	static column_vector tunePID();

	static column_vector rigParamsConAz;
	static column_vector resParamsConAz;
	static column_vector rigParamsConEl;
	static column_vector resParamsConEl;
};