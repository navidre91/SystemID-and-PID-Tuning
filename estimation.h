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

typedef Col<double> column_vector_rig;

class estimation{
public:
	int a = 0;
	static double *input;
	static double *output;
	static double *paramCRig;
	static double *paramCRes;
	static double errorSqSum;
	column_vector_rig startingPointRig;
	static column_vector_rig finalPointRig;
	column_vector_rig startingPointRes;
	static column_vector_rig finalPointRes;
	column_vector_rig targetVector;
	static column_vector_rig u;
	static column_vector_rig uShift;
	static column_vector_rig uInt;
	static column_vector_rig y;
	static column_vector_rig yDiff;
	static column_vector_rig yDiffFilt;
	static column_vector_rig yRig;
	static column_vector_rig yRigyDiff;
	static column_vector_rig yRigyDiffFilt;
	estimation(double* inputFromTrackerAz, double* outputFromTrackerAz, double* inputFromTrackerEl, double* outputFromTrackerEl);
	void importInputData();
	void setInputData(double* inputFromTracker);
	void importOutputData();
	void setOutputData(double* outputFromTracker);
	void inputIntegration(double dt);
	void outputDifferentation(double dt);
	void inputShift();
	static double errorRig(const column_vector_rig& params);
	static double errorRigC(unsigned numberOfParamsC, const double* paramsC, double* grad, void* errorRigCData);
	static double errorRigCyDiff(unsigned numberOfParamsC, const double* paramsC, double* grad, void* errorRigCData);
	static double errorResC(unsigned numberOfParamsC, const double* paramsC, double* grad, void* errorResCData);
	static double errorResCyDiff(unsigned numberOfParamsC, const double* paramsC, double* grad, void* errorResCData);
	static double constraintErrorRigC(unsigned numberOfParamsC, const double* paramsC, double* grad, void* ConstraintErrorRigCData);
	static double constraintErrorResC(unsigned numberOfParamsC, const double* paramsC, double* grad, void* ConstraintErrorRigCData);
	static column_vector_rig errorRigVect(const column_vector_rig &params);
	void setStartingPointRig();
	void setStartingPointRigyDiff();
	void setStartingPointRes();
	void setStartingPointResyDiff();
	static column_vector_rig derivativeRig(const column_vector_rig &params);
	static Mat<double> hessianRig(const column_vector_rig &params);
	void optimizeRig();
	void optimizeRes();
	void setTargetVector();
	static void convertA2C(double *&c, column_vector_rig a);
	static void convertC2A(double *c, column_vector_rig &a);
	void calculateYRig();
	void calculateYRigyDiff();
	void highpassFilter();
	void highpassFilteryDiff();
	static column_vector_rig yRigFilt;
	static column_vector_rig yTotFilt;
	static int rowCount;
	static int axis;
	static int disturbance;
};