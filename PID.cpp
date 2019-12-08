#pragma once

#include <iostream>
#include "stdafx.h"
#include <armadillo>
#include <complex>
#include <iomanip>
#include <cmath>
#include <fstream>
#include "PID.h"
#include <nlopt.h>

using namespace std;
using namespace arma;

typedef Col<double> column_vector;
typedef Col<cx_double> column_vector_cx;
#define  j1 cx_double(0,1)


column_vector PID::rigParamsDis = zeros<column_vector>(8);
column_vector PID::resParamsDis = zeros<column_vector>(9);
column_vector_cx PID::rigParamsCon = zeros<column_vector_cx>(3);
column_vector_cx PID::resParamsCon = zeros<column_vector_cx>(5);


column_vector PID::PIDparams = zeros<column_vector>(5);


int PID::hasRes = 0;
int PID::axFlag = 0;
int PID::gonnaOverDamped = 0;
double PID::a = 0;
double PID::b = 0;
double PID::p1=0;
double PID::p2=0;
double PID::k = 0;
double PID::dkv = 0;
double PID::z1 = 0;
double PID::z2 = 0;
double PID::kd = 0;
double PID::kdmin = 0;
double PID::kdmax = 0;
double PID::kdstart = 0;
int PID::tachFailFlag = 1;
double PID::omegar = 0;
double PID::omegad = 0;


column_vector PID::rigParamsConAz = zeros<column_vector>(3);
column_vector PID::resParamsConAz = zeros<column_vector>(5);
column_vector PID::rigParamsConEl = zeros<column_vector>(3);
column_vector PID::resParamsConEl = zeros<column_vector>(5);


PID::PID(int axisFlag){
	PID::axFlag = axisFlag;
	PID::initPIDparams();
	if (axisFlag == 1){
		PID::rigParamsDis.load("RigTFAz.txt");
		PID::resParamsDis.load("ResTFAz.txt");
	}
	if (axisFlag == 2){
		PID::rigParamsDis.load("RigTFEl.txt");
		PID::resParamsDis.load("ResTFEl.txt");
	}
	PID::initPIDparams();
	PID::hasRes=PID::checkRes();
	PID::rigParamsCon = calcRigParamsCon(rigParamsDis);
	PID::resParamsCon = calcResParamsCon(resParamsDis);
	PID::hasRes = PID::checkRes2();
	cout << endl << "checkres:" << hasRes<<endl;
	PID::tunePID();

}
int PID::checkRes(){
	double num;
	double den;
	int hasRes=0;
	num = pow(PID::resParamsDis(1), 2) - 4 * resParamsDis(0) * resParamsDis(2);
	den = pow(PID::resParamsDis(7), 2) - 4 * resParamsDis(8);

	if (num < 0 && den < 0){
		hasRes = 1;
	}
	return hasRes;
}
int PID::checkRes2(){
	if (hasRes==1){

		if (sqrt(double(real(resParamsCon(4)))) > 624){

			return 0;
		}


		double w=0;
		w=findMaxMagResFreq();
		cout << endl << "magres:" << w << endl;
		if (magRes(w)>0.5){
			return 1;
		}
		else{
			return 0;
		}
		
	}
	else{
		return 0;
	}
}
int PID::checkDamping(){
	double b = double(real(PID::rigParamsCon(1)));
	double c = double(real(PID::rigParamsCon(2)));
	if ((b/2)>100 && (b*b-4*c)>0){
		return 1;
	}
	else{
		return 0;
	}
}
void PID::initPIDparams(){
	PID::PIDparams(0) = 320;
	PID::PIDparams(1) = 6;
	PID::PIDparams(2) = 1800;
	PID::PIDparams(3) = 1000;
	PID::PIDparams(4) = 4.1;
}
column_vector_cx PID::calcRigParamsCon(column_vector rigParamsDis){
	column_vector_cx rigParamsCon = zeros<column_vector_cx>(3);
	column_vector_cx rigParamsDisCx = zeros<column_vector_cx>(8);
	column_vector_cx pzRig = zeros<column_vector_cx>(2);
	column_vector_cx pcRig = zeros<column_vector_cx>(2);
	for (int i = 0; i < 8; i++){
		rigParamsDisCx(i) = cx_double(rigParamsDis(i), 0);
	}
	pzRig(0) = (-rigParamsDisCx(6) + sqrt(pow(rigParamsDisCx(6), 2) - cx_double(4, 0)*rigParamsDisCx(7))) / cx_double(2,0);
	pzRig(1) = (-rigParamsDisCx(6) - sqrt(pow(rigParamsDisCx(6), 2) - cx_double(4, 0)*rigParamsDisCx(7))) / cx_double(2,0);
	pcRig(0) = cx_double(1000, 0)*log(pzRig(0));
	pcRig(1) = cx_double(1000, 0)*log(pzRig(1));
	rigParamsCon(1) = -(pcRig(0) + pcRig(1));
	rigParamsCon(2) = (pcRig(0) * pcRig(1));
	rigParamsCon(0) = abs((rigParamsDis(0)*exp(j1*0.001) + rigParamsDis(1)) / (pow(exp(j1*0.001), 2) + rigParamsDis(6)*exp(j1*0.001) + rigParamsDis(7))) / abs(cx_double(1, 0) / (pow(j1, 2) + rigParamsCon(1)*j1 + rigParamsCon(2)));
	if (PID::axFlag == 1){
		rigParamsCon.save("rigParamsConAz.txt", raw_ascii);
		for (int i = 0; i < 3; i++){
		//	rigParamsAzConC[i] = double(real(rigParamsCon(i)));
		rigParamsConAz(i) = double(real(rigParamsCon(i)));
		}
	}
	else{
		rigParamsCon.save("rigParamsConEl.txt", raw_ascii);
		for (int i = 0; i < 3; i++){
		//	rigParamsElConC[i] = double(real(rigParamsCon(i)));
			rigParamsConEl(i) = double(real(rigParamsCon(i)));
		}

	}
	return (rigParamsCon);
}
column_vector_cx PID::calcResParamsCon(column_vector resParamsDis){
	column_vector_cx resParamsCon = zeros<column_vector_cx>(5);
	column_vector_cx resParamsDisCx = zeros<column_vector_cx>(9);
	column_vector_cx pzRes = zeros<column_vector_cx>(2);
	column_vector_cx pcRes = zeros<column_vector_cx>(2);
	column_vector_cx zzRes = zeros<column_vector_cx>(2);
	column_vector_cx zcRes = zeros<column_vector_cx>(2);

	for (int i = 0; i < 9; i++){
		resParamsDisCx(i) = cx_double(resParamsDis(i), 0);
	}
	 
	pzRes(0) = (-resParamsDisCx(7) + sqrt(pow(resParamsDisCx(7), 2) - cx_double(4, 0)*resParamsDisCx(8))) / cx_double(2, 0);
	pzRes(1) = (-resParamsDisCx(7) - sqrt(pow(resParamsDisCx(7), 2) - cx_double(4, 0)*resParamsDisCx(8))) / cx_double(2, 0);
	pcRes(0) = cx_double(1000, 0)*log(pzRes(0));
	pcRes(1) = cx_double(1000, 0)*log(pzRes(1));
	resParamsCon(3) = -(pcRes(0) + pcRes(1));
	resParamsCon(4) = (pcRes(0) * pcRes(1));

	zzRes(0) = (-resParamsDisCx(1) / resParamsDisCx(0) + sqrt(pow(resParamsDisCx(1) / resParamsDisCx(0), 2) - cx_double(4, 0)*resParamsDisCx(2) / resParamsDisCx(0))) / cx_double(2, 0);
	zzRes(1) = (-resParamsDisCx(1) / resParamsDisCx(0) - sqrt(pow(resParamsDisCx(1) / resParamsDisCx(0), 2) - cx_double(4, 0)*resParamsDisCx(2) / resParamsDisCx(0))) / cx_double(2, 0);
	zcRes(0) = cx_double(1000, 0)*log(zzRes(0));
	zcRes(1) = cx_double(1000, 0)*log(zzRes(1));
	resParamsCon(1) = -(zcRes(0) + zcRes(1));
	resParamsCon(2) = (zcRes(0) * zcRes(1));
	resParamsCon(0) = resParamsDisCx(0);
	if (PID::axFlag == 1){
		resParamsCon.save("resParamsConAz.txt", raw_ascii);
		for (int i = 0; i < 5; i++){
		//	resParamsAzConC[i] = double(real(resParamsCon(i)));
			resParamsConAz(i) = double(real(resParamsCon(i)));
		}
	}
	else{
		resParamsCon.save("resParamsConEl.txt", raw_ascii);
		for (int i = 0; i < 5; i++){
		//	resParamsElConC[i] = double(real(resParamsCon(i)));
		    resParamsConEl(i) = double(real(resParamsCon(i)));
		}
	}
	return (resParamsCon);
}
double PID::phaseRig(double w){
	cx_double wc = cx_double(w, 0);
	cx_double cplx = rigParamsCon(0) / (pow(j1*wc, 2) + rigParamsCon(1)*j1*wc + rigParamsCon(2)) / j1*wc;
	return (atan2(double(imag(cplx)), double(real(cplx))));
}
double PID::phaseRigNew(double w){
	cx_double wc = cx_double(w, 0);
	cx_double k = cx_double(PID::k, 0);
	cx_double a = cx_double(PID::p1, 0) + cx_double(PID::p2, 0);
	cx_double b = cx_double(PID::p1, 0) * cx_double(PID::p2, 0);
	cx_double cplx = k / (pow(j1*wc, 2) + a*j1*wc + b) / (j1*wc);
	return (atan2(double(imag(cplx)), double(real(cplx))));
}
double PID::phaseRes(double w){
	cx_double wc = cx_double(w, 0);
	cx_double cplx = resParamsCon(0)*(pow(j1*wc, 2) + resParamsCon(1)*j1*wc + resParamsCon(2)) / (pow(j1*wc, 2) + resParamsCon(3)*j1*wc + resParamsCon(4));
	return (atan2(double(imag(cplx)), double(real(cplx))));
}
double PID::phaseResWorst(double w){
	cx_double wc = cx_double(w, 0);
	cx_double cplx = resParamsCon(0)*(cx_double(0,0)*pow(j1*wc, 2) + resParamsCon(1)*j1*wc + resParamsCon(2)) / (pow(j1*wc, 2) + resParamsCon(3)*j1*wc + resParamsCon(4));
	return (atan2(double(imag(cplx)), double(real(cplx))));
}
double PID::phasePID(double w){
	cx_double wc = cx_double(w, 0);
	cx_double cplx = (cx_double((PID::z1 + PID::z2)*PID::kd, 0) + cx_double(PID::z1 * PID::z2*PID::kd, 0) / (j1*wc) + cx_double(PID::kd, 0) * (j1*wc));
	return (atan2(double(imag(cplx)), double(real(cplx))));
}
double PID::totPhase(double w){
	if (hasRes == 0){
		return modfn(phaseRig(w), -2 * 3.14);
	}
	else{
		return modfn(phaseRig(w) + phaseResWorst(w), -2 * 3.14);
	}
}
double PID::totPhaseNew(double w){
	if (hasRes == 0){
		return modfn(phaseRigNew(w), -2 * 3.14);
	}
	else{
		return modfn(phaseRigNew(w) + phaseResWorst(w), -2 * 3.14);
	}
}
double PID::totPhasePID(double w){
	if (hasRes == 0){
		return modfn(PID::phaseRig(w) + PID::phasePID(w), -2 * 3.14);
	}
	else{
		return modfn(PID::phaseRig(w) + PID::phaseResWorst(w) + PID::phasePID(w), -2 * 3.14);
	}
}
double PID::totPhasePIDNew(double w){
	if (hasRes == 0){
		return modfn(PID::phaseRigNew(w) + PID::phasePID(w), -2 * 3.14);
	}
	else{
		return modfn(PID::phaseRigNew(w) + PID::phaseRes(w), -2 * 3.14);
	}
}
void PID::checkPhase(){
	column_vector phase = zeros<column_vector>(5000);
	for (int i = 0; i < 5000; i++){
		phase(i) = totPhasePIDNew(i + 1);
	}
	(phase).save("phaseTest.txt", raw_ascii);
}
void PID::checkMag(){
	column_vector mag = zeros<column_vector>(5000);
	for (int i = 0; i < 5000; i++){
		mag(i) = magRigNew(i + 1);
	}
	(mag).save("magTest.txt", raw_ascii);
}
double PID::phase180Obj(unsigned numberOfParams, const double* params, double* grad, void* errorResCData){
	return(pow(PID::totPhase(params[0])+3.14,2));
}
double PID::phase170OLObj(unsigned numberOfParams, const double* params, double* grad, void* errorResCData){
	return(pow(PID::totPhasePIDNew(params[0]) + 2.96, 2));
}
double PID::phaseMaxPhaseOLObj(unsigned numberOfParams, const double* params, double* grad, void* errorResCData){
	return(-PID::totPhasePIDNew(params[0]));
}
double PID::phaseMaxPhaseOLResWorstObj(unsigned numberOfParams, const double* params, double* grad, void* errorResCData){
	return(-PID::totPhasePID(params[0]));
}
double PID::magRig(double w){
	cx_double wc = cx_double(w, 0);
	cx_double cplx = rigParamsCon(0) / (pow(j1*wc, 2) + rigParamsCon(1)*j1*wc + rigParamsCon(2)) / (j1*wc);
	return (abs(cplx));
}
double PID::magRigNew(double w){
	cx_double wc = cx_double(w, 0);
	cx_double k = cx_double(PID::k, 0);
	cx_double a = cx_double(PID::p1, 0) + cx_double(PID::p2, 0);
	cx_double b = cx_double(PID::p1, 0) * cx_double(PID::p2, 0);

	cx_double cplx = k / (pow(j1*wc, 2) + a*j1*wc + b) / (j1*wc);
	return (abs(cplx));
}
double PID::magRes(double w){
	cx_double wc = cx_double(w, 0);
	cx_double cplx = resParamsCon(0)*(pow(j1*wc, 2) + resParamsCon(1)*j1*wc + resParamsCon(2)) / (pow(j1*wc, 2) + resParamsCon(3)*j1*wc + resParamsCon(4));
	return (abs(cplx));
}
double PID::magResWorst(double w){
	cx_double wc = cx_double(w, 0);
	cx_double cplx = resParamsCon(0)*(cx_double(0, 0)*pow(j1*wc, 2) + resParamsCon(1)*j1*wc + resParamsCon(2)) / (pow(j1*wc, 2) + resParamsCon(3)*j1*wc + resParamsCon(4));
	return (abs(cplx));
}
double PID::magPID(double w){
	cx_double wc = cx_double(w, 0);
	cx_double cplx = (cx_double((PID::z1 + PID::z2)*PID::kd, 0) + cx_double(PID::z1 * PID::z2*PID::kd, 0) / (j1*wc) + cx_double(PID::kd, 0) * (j1*wc));
	return (abs(cplx));
}
double PID::totMag(double w){
	if (PID::axFlag == 1){
		if (hasRes == 0){
			return (1200*magRig(w));
		}
		else{
			return (1200 * magRig(w) * magResWorst(w));
		}
	}
	else{
		if (hasRes == 0){
			return (1400 * magRig(w));
		}
		else{
			return (1400 * magRig(w) * magResWorst(w));
		}
	
	}
}
double PID::totMagNew(double w){
	if (PID::axFlag == 1){
		if (hasRes == 0){
			return (1200 * magRigNew(w));
		}
		else{
			return (1200 * magRigNew(w) * magResWorst(w));
		}
	}
	else{
		if (hasRes == 0){
			return (1400 * magRigNew(w));
		}
		else{
			return (1400 * magRigNew(w) * magResWorst(w));
		}

	}
}
double PID::totMagPID(double w){


	return (PID::totMag(w) * PID::magPID(w));
	
	
}
double PID::totMagPIDNew(double w){


	return (PID::totMagNew(w) * PID::magPID(w));


}
double PID::magResObj(unsigned numberOfParams, const double* params, double* grad, void* errorResCData){

	return(-log(PID::magRes(params[0])));
}
double PID::magResWorstObj(unsigned numberOfParams, const double* params, double* grad, void* errorResCData){

	return(-log(PID::magResWorst(params[0])));
}
double PID::modfn(double a, double b){
	int i = 0;
	while (i*b <= a){
		i = i - 1;
	}
	return(a - (i)*b);
}
double PID::constraintOpt(unsigned numberOfParams, const double* params, double* grad, void* ConstraintErrorData){

	return(-params[0] + 1);
}
double PID::find180Freq(){

	double lb[1] = { 0.001 };
	double ub[1] = { 2000 };
	nlopt_opt optG;
	nlopt_opt optL;
	optL = nlopt_create(NLOPT_LN_BOBYQA, 1);
	optG = nlopt_create(NLOPT_GN_CRS2_LM, 1);
	nlopt_set_lower_bounds(optG, lb);
	nlopt_set_upper_bounds(optG, ub);

	nlopt_set_min_objective(optG, PID::phase180Obj, NULL);

	nlopt_add_inequality_constraint(optG, PID::constraintOpt, NULL, 1e-8);

	nlopt_set_ftol_abs(optG, 1e-8);
	nlopt_set_population(optG, 500);
	nlopt_set_maxeval(optG, 100000);

	double minError;
	double w0[1] = { 1 };
	if (nlopt_optimize(optG, w0, &minError) < 0) {
		cout << endl;
		printf("nlopt failed at find180Freq!\n");
	}
	else {
		printf("found minimum at find180Freq = %0.10g\n", minError);
		for (int i = 0; i < 1; i++){
			cout << "---->" << w0[i] << "\n";
		}
	}


	return w0[0];
}
double PID::find170FreqOL(){

	double lb[1] = { sqrt(0.35)*(PID::p1) };
	double ub[1] = { 10*(PID::p1) };

	nlopt_opt optG;
	nlopt_opt optL;
	optL = nlopt_create(NLOPT_LN_COBYLA, 1);
	optG = nlopt_create(NLOPT_GN_CRS2_LM, 1);
	nlopt_set_lower_bounds(optG, lb);
	nlopt_set_upper_bounds(optG, ub);

	nlopt_set_min_objective(optG, PID::phase170OLObj, NULL);

	nlopt_add_inequality_constraint(optG, PID::constraintOpt, NULL, 1e-8);

	nlopt_set_ftol_abs(optL, 1e-3);
	nlopt_set_ftol_abs(optG, 1e-8);
	nlopt_set_population(optG, 500);
	nlopt_set_maxeval(optG, 100000);

	double minError;
	double w0[1] = { sqrt(0.35)*(PID::p1) };
	if (nlopt_optimize(optG, w0, &minError) < 0) {
		printf("nlopt failed at find170FreqOL!\n");
	}
	else {
		printf("found minimum at find170FreqOL = %0.10g\n", minError);
		for (int i = 0; i < 1; i++){
			cout << "---->" << w0[i] << "\n";
		}
	}


	return w0[0];
}
double PID::findMaxPhaseFreqOL(){

	double lb[1] = { 0.1 };
	double ub[1] = { 100 * PID::p1 };

	nlopt_opt optG;
	nlopt_opt optL;
	optL = nlopt_create(NLOPT_LN_COBYLA, 1);
	optG = nlopt_create(NLOPT_GN_CRS2_LM, 1);
	nlopt_set_lower_bounds(optG, lb);
	nlopt_set_upper_bounds(optG, ub);

	nlopt_set_min_objective(optG, PID::phaseMaxPhaseOLObj, NULL);

	nlopt_add_inequality_constraint(optG, PID::constraintOpt, NULL, 1e-8);

	nlopt_set_ftol_abs(optL, 1e-3);
	nlopt_set_ftol_abs(optG, 1e-8);
	nlopt_set_population(optG, 500);
	nlopt_set_maxeval(optG, 100000);

	double minError;
	double w0[1] = { 1 };
	if (nlopt_optimize(optG, w0, &minError) < 0) {
		printf("nlopt failed at findMaxPhaseFreqOL!\n");
	}
	else {
		printf("found minimum at findMaxPhaseFreqOL = %0.10g\n", minError);
		for (int i = 0; i < 1; i++){
			cout << "---->" << w0[i] << "\n";
		}
	}


	return w0[0];
}

double PID::findMaxPhaseFreqResWorstFreq(){

	double lb[1] = { 0.1 };

	double omegaMax = PID::omegad;
	if (omegaMax < PID::omegar){
		omegaMax = PID::omegar;
	}

	double ub[1] = { 2*omegaMax };

	nlopt_opt optG;
	nlopt_opt optL;
	optL = nlopt_create(NLOPT_LN_BOBYQA, 1);
	optG = nlopt_create(NLOPT_GN_CRS2_LM, 1);
	nlopt_set_lower_bounds(optG, lb);
	nlopt_set_upper_bounds(optG, ub);
	nlopt_set_ftol_abs(optG, 1e-8);
	nlopt_set_population(optG, 500);
	nlopt_set_maxeval(optG, 100000);


	nlopt_set_min_objective(optG, PID::phaseMaxPhaseOLResWorstObj, NULL);

	nlopt_add_inequality_constraint(optG, PID::constraintOpt, NULL, 1e-8);



	double minError;
	double w0[1] = { 1 };
	if (nlopt_optimize(optG, w0, &minError) < 0) {
		printf("nlopt failed at findMaxPhaseFreqResWorstFreq!\n");
	}
	else {
		printf("found minimum at findMaxPhaseFreqResWorstFreq = %0.10g\n", minError);
		for (int i = 0; i < 1; i++){
			cout << "---->" << w0[i] << "\n";
		}
	}


	return w0[0];
}
double PID::findMaxMagResWorstFreq(){

	double lb[1] = { 0.001 };
	double ub[1] = { 2000 };
	nlopt_opt optG;
	nlopt_opt optL;
	optL = nlopt_create(NLOPT_LN_BOBYQA, 1);
	optG = nlopt_create(NLOPT_GN_CRS2_LM, 1);

	nlopt_set_lower_bounds(optG, lb);
	nlopt_set_upper_bounds(optG, ub);
	nlopt_set_lower_bounds(optL, lb);
	nlopt_set_upper_bounds(optL, ub);

	nlopt_set_min_objective(optL, PID::magResWorstObj, NULL);
	nlopt_add_inequality_constraint(optL, PID::constraintOpt, NULL, 1e-8);
	nlopt_set_xtol_rel(optL, 1e-3);
	nlopt_set_population(optG, 2000);

	double minError;
	double w0[1] = { 10 };
	if (nlopt_optimize(optL, w0, &minError) < 0) {
		printf("nlopt failed at findMaxMagResWorstFreq!\n");
	}
	else {
		printf("found minimum at findMaxMagResWorstFreq = %0.10g\n", minError);
		for (int i = 0; i < 1; i++){
			cout << "---->" << w0[i] << "\n";
		}
	}

	return w0[0];
}
double PID::findMaxMagResFreq(){

	double lb[1] = { 0.001 };
	double ub[1] = { 2000 };
	nlopt_opt optG;
	nlopt_opt optL;
	optL = nlopt_create(NLOPT_LN_BOBYQA, 1);
	optG = nlopt_create(NLOPT_GN_CRS2_LM, 1);

	nlopt_set_lower_bounds(optG, lb);
	nlopt_set_upper_bounds(optG, ub);
	nlopt_set_lower_bounds(optL, lb);
	nlopt_set_upper_bounds(optL, ub);

	nlopt_set_min_objective(optG, PID::magResObj, NULL);
	nlopt_add_inequality_constraint(optG, PID::constraintOpt, NULL, 1e-8);
	nlopt_set_xtol_rel(optG, 1e-3);
	nlopt_set_population(optG, 500);
	nlopt_set_maxeval(optG, 100000);

	double minError;
	double w0[1] = { 1 };
	if (nlopt_optimize(optG, w0, &minError) < 0) {
		printf("nlopt failed at findMaxMagResFreq!\n");
	}
	else {
		printf("found minimum at findMaxMagResFreq = %0.10g\n", minError);
		for (int i = 0; i < 1; i++){
			cout << "---->" <<w0[i] << "\n";
		}
	}

	return w0[0];
}
void PID::setkd(double kd){
	PID::kd = kd;
}
double PID::findkdmax(){
	double w = PID::find170FreqOL();
	return (1 / PID::totMagPIDNew(w));
}

double PID::findkdmin(){
	double w = PID::findMaxPhaseFreqOL();
	return (1 / PID::totMagPIDNew(w));
}

void PID::setTuneParams(){

	if (PID::hasRes == 0 && PID::gonnaOverDamped==1){
		PID::k = real(PID::rigParamsCon(0));
		PID::a = real(PID::rigParamsCon(1));
		PID::b = real(PID::rigParamsCon(2));
		PID::p1 = 100.0;
		PID::p2 = a - 100;
		PID::z1 = 0.35*PID::p1;
		PID::z2 = 0.9*PID::p2;
		PID::dkv = (PID::p1*PID::p2 - PID::b)/PID::k;
		PID::setkd(1);
		PID::kdmin = PID::findkdmin();
		PID::kdmax = PID::findkdmax();
		cout << endl << "max kd:" << PID::kdmax;
		cout << endl << "min kd:" << PID::kdmin;
	}

	if (PID::hasRes == 0 && PID::gonnaOverDamped == 0){
		PID::k = real(PID::rigParamsCon(0));
		PID::a = real(PID::rigParamsCon(1));
		PID::b = real(PID::rigParamsCon(2));
		PID::p1 = a/2;
		PID::p2 = a/2;
		PID::z1 = 0.35*PID::p1;
		PID::z2 = 0.9*PID::p2;
		PID::dkv = (PID::p1*PID::p2 - PID::b) / PID::k;
		PID::setkd(1);
		PID::kdmin = PID::findkdmin();
		PID::kdmax = PID::findkdmax();
		cout << endl << "max kd:" << PID::kdmax;
		cout << endl << "min kd:" << PID::kdmin;
	}

	if (PID::hasRes == 1 && PID::gonnaOverDamped == 1){
		double w180=0;
		double wpmax = 0;
		PID::omegar = sqrt(double(real(PID::rigParamsCon(2))));
		PID::omegad = sqrt(double(real(PID::resParamsCon(2))));
		double zetar = double(real(PID::rigParamsCon(1))) / 2 / omegar;
		w180 = find180Freq();
		PID::z1 = w180*0.2;
		//PID::z2 = 0.7*omegar*(zetar + sqrt(pow(zetar, 2) - 1));
		PID::z2 = w180;
		PID::dkv = 0;
		PID::setkd(1);
		wpmax = findMaxPhaseFreqResWorstFreq();

		kdstart = 1 / (totMagPID(wpmax));

		if (kdstart > (1 / (totMagPID(omegar)))){
			kdstart = 1 / (totMagPID(omegar));
		}
		if (kdstart > (1 / (totMagPID(omegad)))){
			kdstart = 1 / (totMagPID(omegad));
		}

		cout << endl << "kstart:" << kdstart;
	}


	if (PID::hasRes == 1 && PID::gonnaOverDamped == 0){

		double w180 = 0;
		double wpmax = 0;
		PID::omegar = sqrt(double(real(PID::rigParamsCon(2))));
		PID::omegad = sqrt(double(real(PID::resParamsCon(2))));
		double zetar = double(real(PID::rigParamsCon(1))) / 2 / omegar;
		w180 = find180Freq();
		PID::z1 = w180*0.2;
		//PID::z2 = 0.7*omegar;
		PID::z2 = w180;
		PID::dkv = 0;
		PID::setkd(1);
		wpmax = findMaxPhaseFreqResWorstFreq();
		kdstart = 1 / (totMagPID(wpmax));
		if (kdstart > (1 / (totMagPID(omegar)))){
			kdstart = 1 / (totMagPID(omegar));
		}
		if (kdstart > (1 / (totMagPID(omegad)))){
			kdstart = 1 / (totMagPID(omegad));
		}


	}



}
column_vector PID::tunePID(){

	PID::gonnaOverDamped = PID::checkDamping();
	PID::setTuneParams();


	return PID::PIDparams;
}

