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


double estimation::errorSqSum = 0;
column_vector_rig estimation::finalPointRig;
column_vector_rig estimation::finalPointRes;
column_vector_rig estimation::u;
column_vector_rig estimation::uShift;
column_vector_rig estimation::uInt;
column_vector_rig estimation::y;
column_vector_rig estimation::yDiff;
column_vector_rig estimation::yDiffFilt;
column_vector_rig estimation::yRig;
column_vector_rig estimation::yRigyDiff;
column_vector_rig estimation::yRigyDiffFilt;
column_vector_rig estimation::yRigFilt;
column_vector_rig estimation::yTotFilt;

double* estimation::input;
double* estimation::output;
double* estimation::paramCRig;
double* estimation::paramCRes;

int estimation::rowCount = 10000;
int estimation::axis = 0;
int estimation::disturbance = 0;

estimation::estimation(double* inputFromTrackerAz, double* outputFromTrackerAz, double* inputFromTrackerEl, double* outputFromTrackerEl){
	startingPointRig.set_size(8);
	finalPointRig.set_size(8);
	startingPointRes.set_size(9);
	finalPointRes.set_size(9);
	importInputData();
	importOutputData();
	//inputIntegration(0.001);
	outputDifferentation(0.001);
	inputShift();
	axis = 0;
	//Azimtuh
	//setInputData(inputFromTrackerAz);
	//setOutputData(outputFromTrackerAz);
	//setStartingPointRig();
	setStartingPointRigyDiff();
	optimizeRig();
	//calculateYRig();
	calculateYRigyDiff();
	highpassFilteryDiff();
	setStartingPointResyDiff();
	optimizeRes();
	/////Elevation
	//axis = 1;
	//setInputData(inputFromTrackerEl);
	//setOutputData(outputFromTrackerEl);
	//setStartingPointRig();
	//optimizeRig();
	//calculateYRig();
	//calculateYRigyDiff();
	//highpassFilter();
	//setStartingPointRes();
	//optimizeRes();
}

void estimation::importInputData(){
	(u).load("input.txt", raw_ascii);
}

void estimation::setInputData(double* inputFromTracker){

	u.set_size(rowCount);
	
	for (int i = 0; i < rowCount; i++){
		u(i) = inputFromTracker[i];
	}

}

void estimation::importOutputData(){
	(y).load("output.txt", raw_ascii);
}

void estimation::setOutputData(double* outputFromTracker){

	y.set_size(rowCount);

	for (int i = 0; i < rowCount; i++){
		y(i) = outputFromTracker[i];
	}

}

void estimation::inputIntegration(double dt){
	Mat<double> temp = arma::mat(1,1);
	for (int i = 0; i < (u).n_elem; i++){
		if (i == 0){
			//temp(0,0) = (((u)(i)) / 2)*dt;
			temp(0, 0) = 0;
			(uInt) = join_vert((uInt), temp);

		}
		else{
			//temp(0,0) = ( ((u)(i - 1) + (u)(i)) / 2)*dt;
			temp(0, 0) = ((u)(i - 1) )*dt;
			temp(0,0) = temp(0, 0) + (uInt)(i - 1);
			(uInt) = join_vert((uInt), temp);
		}
	}
	(uInt).save("inputInt.txt", raw_ascii);
}

void estimation::outputDifferentation(double dt){
	Mat<double> temp = arma::mat(1, 1);
	for (int i = 1; i < (u).n_elem; i++){
			temp(0, 0) = (y(i)-y(i-1))/dt;
			(yDiff) = join_vert((yDiff), temp);
		}
	(yDiff).save("outputDiff.txt", raw_ascii);
}

void estimation::inputShift(){

	uShift = u.rows(1, u.n_rows-1);
	(uShift).save("inputShift.txt", raw_ascii);
}

double estimation::errorRig(const column_vector_rig &params){
	errorSqSum = 0;
	Mat<double> error = arma::mat((u).n_elem, 1,fill::zeros);
	for (int k = 4; k < (u).n_elem; k++){

		error(k) = (-error(k - 1)*(1 * params(6) + params(2) * 1) - error(k - 2)*(1 * params(7) + params(2)*params(6) + params(3) * 1) - error(k - 3)*(params(2)*params(7) + params(3)*params(6)) - error(k - 4)*(params(3)*params(7)) + y(k)*(1 * 1) + y(k - 1)*(1 * params(6) + params(4) * 1) + y(k - 2)*(1 * params(7) + params(4)*params(6) + params(5) * 1) + y(k - 3)*(params(4)*params(7) + params(5)*params(6)) + y(k - 4)*(params(5)*params(7)) - (uInt(k)*(params(0) * 1) + uInt(k - 1)*(params(0)*params(4) + params(1) * 1) + uInt(k - 2)*(params(0)*params(5) + params(1)*params(4)) + uInt(k - 3)*(params(1)*params(5)))) / (1 * 1);
	}

	for (int j = 0; j < (u).n_elem; j++){
		errorSqSum = error(j)*error(j) + errorSqSum;
	}
	cout << errorSqSum<<"\n";
	estimation::finalPointRig = params;
	if (errorSqSum < 1e100){
		return sqrt(errorSqSum);
	}
	else{
		return 1e100;
	}
	//return errorSqSum;
	
}

double estimation::errorRigC(unsigned numberOfParamsC, const double* paramsC, double* grad, void* errorRigCData){
	errorSqSum = 0;
	Mat<double> error = arma::mat((u).n_elem, 1, fill::zeros);
	for (int k = 4; k < (u).n_elem; k++){

		//error(k) = (-error(k - 1)*(1 * paramsC[6] + paramsC[2] * 1) - error(k - 2)*(1 * paramsC[7] + paramsC[2]*paramsC[6] + paramsC[3] * 1) - error(k - 3)*(paramsC[2]*paramsC[7] + paramsC[3]*paramsC[6]) - error(k - 4)*(paramsC[3]*paramsC[7]) + y(k)*(1 * 1) + y(k - 1)*(1 * paramsC[6] + paramsC[4] * 1) + y(k - 2)*(1 * paramsC[7] + paramsC[4]*paramsC[6] + paramsC[5] * 1) + y(k - 3)*(paramsC[4]*paramsC[7] + paramsC[5]*paramsC[6]) + y(k - 4)*(paramsC[5]*paramsC[7]) - (uInt(k)*(paramsC[0] * 1) + uInt(k - 1)*(paramsC[0]*paramsC[4] + paramsC[1] * 1) + uInt(k - 2)*(paramsC[0]*paramsC[5] + paramsC[1]*paramsC[4]) + uInt(k - 3)*(paramsC[1]*paramsC[5]))) / (1 * 1);
		//error(k) = (-error(k - 1)*(1 * paramsC[6] + paramsC[2] * 1) - error(k - 2)*(1 * paramsC[7] + paramsC[2] * paramsC[6] + paramsC[3] * 1) - error(k - 3)*(paramsC[2] * paramsC[7] + paramsC[3] * paramsC[6]) - error(k - 4)*(paramsC[3] * paramsC[7]) + y(k)*(1 * 1) + y(k - 1)*(1 * paramsC[6] + paramsC[4] * 1) + y(k - 2)*(1 * paramsC[7] + paramsC[4] * paramsC[6] + paramsC[5] * 1) + y(k - 3)*(paramsC[4] * paramsC[7] + paramsC[5] * paramsC[6]) + y(k - 4)*(paramsC[5] * paramsC[7]) - (uInt(k-1)*(paramsC[0] * 1) + uInt(k - 2)*(paramsC[0] * paramsC[4] + paramsC[1] * 1) + uInt(k - 3)*(paramsC[0] * paramsC[5] + paramsC[1] * paramsC[4]) + uInt(k - 4)*(paramsC[1] * paramsC[5]))) / (1 * 1);
		error(k) = (-error(k - 1)*(1 * paramsC[6] + paramsC[2] * 1) - error(k - 2)*(1 * paramsC[7] + paramsC[2] * paramsC[6] + paramsC[3] * 1) - error(k - 3)*(paramsC[2] * paramsC[7] + paramsC[3] * paramsC[6]) - error(k - 4)*(paramsC[3] * paramsC[7]) + y(k)*(1 * 1) + y(k - 1)*(1 * paramsC[6] + paramsC[4] * 1) + y(k - 2)*(1 * paramsC[7] + paramsC[4] * paramsC[6] + paramsC[5] * 1) + y(k - 3)*(paramsC[4] * paramsC[7] + paramsC[5] * paramsC[6]) + y(k - 4)*(paramsC[5] * paramsC[7]) - (uInt(k-1)*(paramsC[0] * 1) + uInt(k - 2)*(paramsC[0] * paramsC[4] + paramsC[1] * 1) + uInt(k - 3)*(paramsC[0] * paramsC[5] + paramsC[1] * paramsC[4]) + uInt(k - 4)*(paramsC[1] * paramsC[5]))) / (1 * 1);

	}

	for (int j = 0; j < (u).n_elem; j++){
		errorSqSum = error(j)*error(j)+ errorSqSum;
	}
	//cout << errorSqSum << "\n";
	//estimation::finalPointRig = params;
	if (errorSqSum < 1e200){
		return errorSqSum;
	}
	else{
		return 1e200;
	}
	//return errorSqSum;
}


double estimation::errorRigCyDiff(unsigned numberOfParamsC, const double* paramsC, double* grad, void* errorRigCData){

	errorSqSum = 0;
	Mat<double> error = arma::mat((u).n_elem, 1, fill::zeros);
	for (int k = 4; k < ((u).n_elem-1); k++){

		//error(k) = (-error(k - 1)*(1 * paramsC[6] + paramsC[2] * 1) - error(k - 2)*(1 * paramsC[7] + paramsC[2]*paramsC[6] + paramsC[3] * 1) - error(k - 3)*(paramsC[2]*paramsC[7] + paramsC[3]*paramsC[6]) - error(k - 4)*(paramsC[3]*paramsC[7]) + y(k)*(1 * 1) + y(k - 1)*(1 * paramsC[6] + paramsC[4] * 1) + y(k - 2)*(1 * paramsC[7] + paramsC[4]*paramsC[6] + paramsC[5] * 1) + y(k - 3)*(paramsC[4]*paramsC[7] + paramsC[5]*paramsC[6]) + y(k - 4)*(paramsC[5]*paramsC[7]) - (uInt(k)*(paramsC[0] * 1) + uInt(k - 1)*(paramsC[0]*paramsC[4] + paramsC[1] * 1) + uInt(k - 2)*(paramsC[0]*paramsC[5] + paramsC[1]*paramsC[4]) + uInt(k - 3)*(paramsC[1]*paramsC[5]))) / (1 * 1);
		error(k) = (-error(k - 1)*(1 * paramsC[6] + paramsC[2] * 1) - error(k - 2)*(1 * paramsC[7] + paramsC[2] * paramsC[6] + paramsC[3] * 1) - error(k - 3)*(paramsC[2] * paramsC[7] + paramsC[3] * paramsC[6]) - error(k - 4)*(paramsC[3] * paramsC[7]) + yDiff(k)*(1 * 1) + yDiff(k - 1)*(1 * paramsC[6] + paramsC[4] * 1) + yDiff(k - 2)*(1 * paramsC[7] + paramsC[4] * paramsC[6] + paramsC[5] * 1) + yDiff(k - 3)*(paramsC[4] * paramsC[7] + paramsC[5] * paramsC[6]) + yDiff(k - 4)*(paramsC[5] * paramsC[7]) - (uShift(k - 1)*(paramsC[0] * 1) + uShift(k - 2)*(paramsC[0] * paramsC[4] + paramsC[1] * 1) + uShift(k - 3)*(paramsC[0] * paramsC[5] + paramsC[1] * paramsC[4]) + uShift(k - 4)*(paramsC[1] * paramsC[5]))) / (1 * 1);


	}

	for (int j = 0; j < ((u).n_elem-1); j++){
		errorSqSum = error(j)*error(j) + errorSqSum;
	}
	//cout << errorSqSum << "\n";
	//estimation::finalPointRig = params;
	if (errorSqSum < 1e100){
		return sqrt(errorSqSum);
	}
	else{
		return 1e100;
	}
	//return errorSqSum;
}


double estimation::errorResC(unsigned numberOfParamsC, const double* paramsC, double* grad, void* errorResCData){

	errorSqSum = 0;
	Mat<double> error = arma::mat((u).n_elem, 1, fill::zeros);
	for (int k = 4; k < (u).n_elem; k++){

		error(k) = (-error(k - 1)*(1 * paramsC[7]+ paramsC[3] * 1) - error(k - 2)*(1 * paramsC[8]+ paramsC[3]*paramsC[7] + paramsC[4] * 1) - error(k - 3)*(paramsC[3]*paramsC[8] + paramsC[4]*paramsC[7]) - error(k - 4)*(paramsC[4]*paramsC[8]) + yTotFilt(k)*(1 * 1) + yTotFilt(k - 1)*(1 * paramsC[7] + paramsC[5] * 1) + yTotFilt(k - 2)*(1 * paramsC[8] + paramsC[5]*paramsC[7] + paramsC[6] * 1) + yTotFilt(k - 3)*(paramsC[5]*paramsC[8] + paramsC[6]*paramsC[7]) + yTotFilt(k - 4)*(paramsC[6]*paramsC[8]) - (yRigFilt(k)*(paramsC[0] * 1) + yRigFilt(k - 1)*(paramsC[0]*paramsC[5] + paramsC[1] * 1) + yRigFilt(k - 2)*(paramsC[2] + paramsC[0]*paramsC[6] + paramsC[1]*paramsC[5]) + yRigFilt(k - 3)*(paramsC[1]*paramsC[6] + paramsC[2]*paramsC[5]) + yRigFilt(k - 4)*(paramsC[6]*paramsC[2]))) / (1 * 1);

	}

	for (int j = 0; j < (u).n_elem; j++){
		errorSqSum = error(j)*error(j) + errorSqSum;
	}
	//cout << errorSqSum << "\n";
	//estimation::finalPointRig = params;
	if (errorSqSum < 1e100){
		return sqrt(errorSqSum);
	}
	else{
		return 1e100;
	}
	//return errorSqSum;
}


double estimation::errorResCyDiff(unsigned numberOfParamsC, const double* paramsC, double* grad, void* errorResCData){

	errorSqSum = 0;
	Mat<double> error = arma::mat((u).n_elem, 1, fill::zeros);
	for (int k = 4; k < ((u).n_elem-1); k++){

		//error(k) = (-error(k - 1)*(1 * paramsC[7] + paramsC[3] * 1) - error(k - 2)*(1 * paramsC[8] + paramsC[3] * paramsC[7] + paramsC[4] * 1) - error(k - 3)*(paramsC[3] * paramsC[8] + paramsC[4] * paramsC[7]) - error(k - 4)*(paramsC[4] * paramsC[8]) + yDiff(k)*(1 * 1) + yDiff(k - 1)*(1 * paramsC[7] + paramsC[5] * 1) + yDiff(k - 2)*(1 * paramsC[8] + paramsC[5] * paramsC[7] + paramsC[6] * 1) + yDiff(k - 3)*(paramsC[5] * paramsC[8] + paramsC[6] * paramsC[7]) + yDiff(k - 4)*(paramsC[6] * paramsC[8]) - (yRigyDiff(k)*(paramsC[0] * 1) + yRigyDiff(k - 1)*(paramsC[0] * paramsC[5] + paramsC[1] * 1) + yRigyDiff(k - 2)*(paramsC[2] + paramsC[0] * paramsC[6] + paramsC[1] * paramsC[5]) + yRigyDiff(k - 3)*(paramsC[1] * paramsC[6] + paramsC[2] * paramsC[5]) + yRigyDiff(k - 4)*(paramsC[6] * paramsC[2]))) / (1 * 1);
		error(k) = (-error(k - 1)*(1 * paramsC[7] + paramsC[3] * 1) - error(k - 2)*(1 * paramsC[8] + paramsC[3] * paramsC[7] + paramsC[4] * 1) - error(k - 3)*(paramsC[3] * paramsC[8] + paramsC[4] * paramsC[7]) - error(k - 4)*(paramsC[4] * paramsC[8]) + yDiffFilt(k)*(1 * 1) + yDiffFilt(k - 1)*(1 * paramsC[7] + paramsC[5] * 1) + yDiffFilt(k - 2)*(1 * paramsC[8] + paramsC[5] * paramsC[7] + paramsC[6] * 1) + yDiffFilt(k - 3)*(paramsC[5] * paramsC[8] + paramsC[6] * paramsC[7]) + yDiffFilt(k - 4)*(paramsC[6] * paramsC[8]) - (yRigyDiffFilt(k)*(paramsC[0] * 1) + yRigyDiffFilt(k - 1)*(paramsC[0] * paramsC[5] + paramsC[1] * 1) + yRigyDiffFilt(k - 2)*(paramsC[2] + paramsC[0] * paramsC[6] + paramsC[1] * paramsC[5]) + yRigyDiffFilt(k - 3)*(paramsC[1] * paramsC[6] + paramsC[2] * paramsC[5]) + yRigyDiffFilt(k - 4)*(paramsC[6] * paramsC[2]))) / (1 * 1);

	}

	for (int j = 0; j < ((u).n_elem-1); j++){
		errorSqSum = error(j)*error(j) + errorSqSum;
	}
	//cout << errorSqSum << "\n";
	//estimation::finalPointRig = params;
	if (errorSqSum < 1e100){
		return sqrt(errorSqSum);
	}
	else{
		return 1e100;
	}
	//return errorSqSum;
}


double estimation::constraintErrorRigC(unsigned numberOfParamsC, const double* paramsC, double* grad, void* ConstraintErrorRigCData){

	//return(paramsC[0] + paramsC[1] + paramsC[2] + paramsC[3] + paramsC[4] + paramsC[5] + paramsC[6] + paramsC[7] - 20);
	return(-paramsC[0] + paramsC[1]);
}

double estimation::constraintErrorResC(unsigned numberOfParamsC, const double* paramsC, double* grad, void* ConstraintErrorRigCData){

	return(paramsC[0] + paramsC[1] + paramsC[2] + paramsC[3] + paramsC[4] + paramsC[5] + paramsC[6] + paramsC[7] + paramsC[8] - 30);
}


column_vector_rig(estimation::errorRigVect)(const column_vector_rig &params){
	Mat<double> error = arma::mat((u).n_elem, 1, fill::zeros);
	column_vector_rig errorVect((u).n_elem,1);
	for (int k = 4; k < (u).n_elem; k++){

		error(k) = (-error(k - 1)*(1 * params(6) + params(2) * 1) - error(k - 2)*(1 * params(7) + params(2)*params(6) + params(3) * 1) - error(k - 3)*(params(2)*params(7) + params(3)*params(6)) - error(k - 4)*(params(3)*params(7)) + y(k)*(1 * 1) + y(k - 1)*(1 * params(6) + params(4) * 1) + y(k - 2)*(1 * params(7) + params(4)*params(6) + params(5) * 1) + y(k - 3)*(params(4)*params(7) + params(5)*params(6)) + y(k - 4)*(params(5)*params(7)) - (uInt(k)*(params(0) * 1) + uInt(k - 1)*(params(0)*params(4) + params(1) * 1) + uInt(k - 2)*(params(0)*params(5) + params(1)*params(4)) + uInt(k - 3)*(params(1)*params(5)))) / (1 * 1);
	}
	for (int k = 0; k < (u).n_elem; k++){

		errorVect(k) = error(k);
	}
	return errorVect;

}

column_vector_rig estimation::derivativeRig(const column_vector_rig &params){
	double h = 1e-7;
	column_vector_rig temp1;
	column_vector_rig temp2;
	column_vector_rig derivativeVector;
	temp1.set_size(8);
	temp2.set_size(8);
	derivativeVector.set_size(8);
	for (int i = 0; i < 8; i++){
		temp1 = params;
		temp2 = params;
		temp1(i) = temp1(i) + h;
		temp2(i) = temp2(i) - h;
		derivativeVector(i) = (estimation::errorRig(temp1) - estimation::errorRig(temp2)) / 2 / h;

	}
	return derivativeVector;
}


Mat<double> estimation::hessianRig(const column_vector_rig &params){
		double h = 1e-7;
		column_vector_rig tempi1;
		column_vector_rig tempi2;
		column_vector_rig tempi3;
		column_vector_rig tempi4;
		column_vector_rig tempj1;
		column_vector_rig tempj2;
		column_vector_rig tempj3;
		column_vector_rig tempj4;
		Mat<double> res(8, 8);
		for (int i = 0; i < 8; i++){
			tempi1 = params;
			tempi2 = params;
			tempi3 = params;
			tempi4 = params;
			tempi1(i) = params(i) + h;
			tempi2(i) = params(i) - h;
			tempi3(i) = params(i) + h;
			tempi4(i) = params(i) - h;
			for (int j = i; j < 8; j++){
				tempj1 = tempi1;
				tempj2 = tempi2;
				tempj3 = tempi3;
				tempj4 = tempi4;
				tempj1(j) = tempj1(j) + h;
				tempj2(j) = tempj2(j) + h;
				tempj3(j) = tempj3(j) - h;
				tempj4(j) = tempj4(j) - h;
				res(i, j) = res(j, i) = ((estimation::errorRig(tempj1) - estimation::errorRig(tempj2)) / 2 / h - (estimation::errorRig(tempj3) - estimation::errorRig(tempj4)) / 2 / h) / 2 / h;

			}
		
		}
		return res;
	}


void estimation::setStartingPointRig(){
	column_vector_rig Barx = column_vector_rig((uInt.n_elem - 2), fill::zeros);
	Mat<double> Aarx;
	column_vector_rig arxAns = column_vector_rig(4, fill::zeros);

	//Barx = y.rows(2, y.n_elem - 1);
	//Aarx = join_horiz(y.rows(1, y.n_elem - 2), y.rows(0, y.n_elem - 3));
	//Aarx = join_horiz(Aarx, uInt.rows(2, y.n_elem - 1));
	//Aarx = join_horiz(Aarx, uInt.rows(1, y.n_elem - 2));

	Barx = y.rows(2, uInt.n_elem - 1);
	Aarx = join_horiz(y.rows(1, uInt.n_elem - 2), y.rows(0, uInt.n_elem - 3));
	Aarx = join_horiz(Aarx, uInt.rows(1, uInt.n_elem - 2));
	Aarx = join_horiz(Aarx, uInt.rows(0, uInt.n_elem - 3));


	arxAns = (((Aarx.t())*Aarx).i())*((Aarx.t()))*Barx;
	

	startingPointRig(0) = arxAns(2);
	startingPointRig(1) = arxAns(3);
	startingPointRig(2) = 0;
	startingPointRig(3) = 0;
	startingPointRig(4) = -arxAns(0);
	startingPointRig(5) = -arxAns(1);
	startingPointRig(6) = -arxAns(0);
	startingPointRig(7) = -arxAns(1);
	


}

void estimation::setStartingPointRigyDiff(){
	column_vector_rig Barx = column_vector_rig((yDiff.n_elem - 2), fill::zeros);
	Mat<double> Aarx;
	column_vector_rig arxAns = column_vector_rig(4, fill::zeros);

	//Barx = y.rows(2, y.n_elem - 1);
	//Aarx = join_horiz(y.rows(1, y.n_elem - 2), y.rows(0, y.n_elem - 3));
	//Aarx = join_horiz(Aarx, uInt.rows(2, y.n_elem - 1));
	//Aarx = join_horiz(Aarx, uInt.rows(1, y.n_elem - 2));

	Barx = yDiff.rows(2, yDiff.n_elem - 1);
	Aarx = join_horiz(yDiff.rows(1, yDiff.n_elem - 2), yDiff.rows(0, yDiff.n_elem - 3));
	Aarx = join_horiz(Aarx, uShift.rows(1, yDiff.n_elem - 2));
	Aarx = join_horiz(Aarx, uShift.rows(0, yDiff.n_elem - 3));


	arxAns = (((Aarx.t())*Aarx).i())*((Aarx.t()))*Barx;


	startingPointRig(0) = arxAns(2);
	startingPointRig(1) = arxAns(3);
	startingPointRig(2) = 0;
	startingPointRig(3) = 0;
	startingPointRig(4) = -arxAns(0);
	startingPointRig(5) = -arxAns(1);
	startingPointRig(6) = -arxAns(0);
	startingPointRig(7) = -arxAns(1);


}

void estimation::setStartingPointRes(){

	column_vector_rig Barx = column_vector_rig((uInt.n_elem - 2), fill::zeros);
	Mat<double> Aarx;
	column_vector_rig arxAns = column_vector_rig(5, fill::zeros);

	Barx = yTotFilt.rows(2, uInt.n_elem - 1);
	Aarx = join_horiz(yRigFilt.rows(2, uInt.n_elem - 1), yRigFilt.rows(1, uInt.n_elem - 2));
	Aarx = join_horiz(Aarx, yRigFilt.rows(0, uInt.n_elem - 3));
	Aarx = join_horiz(Aarx, yTotFilt.rows(1, uInt.n_elem - 2));
	Aarx = join_horiz(Aarx, yTotFilt.rows(0, uInt.n_elem - 3));

	arxAns = (((Aarx.t())*Aarx).i())*((Aarx.t()))*Barx;


	startingPointRes(0) = arxAns(0);
	startingPointRes(1) = arxAns(1);
	startingPointRes(2) = arxAns(2);
	startingPointRes(3) = 0;
	startingPointRes(4) = 0;
	startingPointRes(5) = -arxAns(3);
	startingPointRes(6) = -arxAns(4);
	startingPointRes(7) = -arxAns(3);
	startingPointRes(8) = -arxAns(4);

}

void estimation::setStartingPointResyDiff(){


	/*column_vector_rig Barx = column_vector_rig((yDiff.n_elem - 2), fill::zeros);
	Mat<double> Aarx;
	column_vector_rig arxAns = column_vector_rig(5, fill::zeros);

	Barx = yDiff.rows(2, yDiff.n_elem - 1);
	Aarx = join_horiz(yRigyDiff.rows(2, yDiff.n_elem - 1), yRigyDiff.rows(1, yDiff.n_elem - 2));
	Aarx = join_horiz(Aarx, yRigyDiff.rows(0, yDiff.n_elem - 3));
	Aarx = join_horiz(Aarx, yDiff.rows(1, yDiff.n_elem - 2));
	Aarx = join_horiz(Aarx, yDiff.rows(0, yDiff.n_elem - 3));

	arxAns = (((Aarx.t())*Aarx).i())*((Aarx.t()))*Barx;*/


	column_vector_rig Barx = column_vector_rig((yDiffFilt.n_elem - 2), fill::zeros);
	Mat<double> Aarx;
	column_vector_rig arxAns = column_vector_rig(5, fill::zeros);

	Barx = yDiffFilt.rows(2, yDiffFilt.n_elem - 1);
	Aarx = join_horiz(yRigyDiffFilt.rows(2, yDiffFilt.n_elem - 1), yRigyDiffFilt.rows(1, yDiffFilt.n_elem - 2));
	Aarx = join_horiz(Aarx, yRigyDiffFilt.rows(0, yDiffFilt.n_elem - 3));
	Aarx = join_horiz(Aarx, yDiffFilt.rows(1, yDiffFilt.n_elem - 2));
	Aarx = join_horiz(Aarx, yDiffFilt.rows(0, yDiffFilt.n_elem - 3));

	arxAns = (((Aarx.t())*Aarx).i())*((Aarx.t()))*Barx;



	startingPointRes(0) = arxAns(0);
	startingPointRes(1) = arxAns(1);
	startingPointRes(2) = arxAns(2);
	startingPointRes(3) = 0;
	startingPointRes(4) = 0;
	startingPointRes(5) = -arxAns(3);
	startingPointRes(6) = -arxAns(4);
	startingPointRes(7) = -arxAns(3);
	startingPointRes(8) = -arxAns(4);

}

void estimation::setTargetVector(){
	targetVector.set_size((estimation::u).n_elem);
	for (int i = 0; i < (estimation::u).n_elem; i++){
		targetVector(i) = 0;
	}

}

void estimation::convertA2C(double *&c, column_vector_rig a){
	c = new double((a).n_elem);

	for (int i = 0; i < (a).n_elem; i++){
		c[i] = a(i);
	}
}

void estimation::convertC2A(double *c, column_vector_rig &a){
	
	for (int i = 0; i < (a).n_elem; i++){
		a(i) = c[i];
	}
}

void estimation::optimizeRig(){
	cout << " -------------- startingPointRig" << endl;
	cout << startingPointRig;

	double lb[8] = { -3, -3, -3, -3, -3, -3, -3, -3 };
	double ub[8] = { 3, 3, 3, 3, 3, 3, 3, 3 };
	//double lb[8] = { startingPointRig(0) - startingPointRig(0) / 10, startingPointRig(1) - startingPointRig(1) / 10, startingPointRig(2) - startingPointRig(2) / 10, startingPointRig(3) - startingPointRig(3) / 10, startingPointRig(4) - startingPointRig(4) / 10, startingPointRig(5) - startingPointRig(5) / 10, startingPointRig(6) - startingPointRig(6) / 10, startingPointRig(7) - startingPointRig(7) / 10 };
	//double ub[8] = { startingPointRig(0) + startingPointRig(0) / 10, startingPointRig(1) + startingPointRig(1) / 10, startingPointRig(2) + startingPointRig(2) / 10, startingPointRig(3) + startingPointRig(3) / 10, startingPointRig(4) + startingPointRig(4) / 10, startingPointRig(5) + startingPointRig(5) / 10, startingPointRig(6) + startingPointRig(6) / 10, startingPointRig(7) + startingPointRig(7) / 10 };
	//double lbp[8] = { startingPointRig(0) - 0.5, startingPointRig(1) - 0.5, startingPointRig(2) - 0.5, startingPointRig(3) - 0.5, startingPointRig(4) - 0.5, startingPointRig(5) - 0.5, startingPointRig(6) - 0.5, startingPointRig(7) - 0.5 };
	//double ubp[8] = { startingPointRig(0) + 0.5, startingPointRig(1) + 0.5, startingPointRig(2) + 0.5, startingPointRig(3) + 0.5, startingPointRig(4) + 0.5, startingPointRig(5) + 0.5, startingPointRig(6) + 0.5, startingPointRig(7) + 0.5 };
	estimation::convertA2C(paramCRig, startingPointRig);

	nlopt_opt optG;
	nlopt_opt optL;

	optG = nlopt_create(NLOPT_GN_CRS2_LM, 8);
	nlopt_set_population(optG, 500);
	optL = nlopt_create(NLOPT_LN_COBYLA, 8);
	nlopt_set_lower_bounds(optG, lb);
	nlopt_set_upper_bounds(optG, ub);
	nlopt_set_lower_bounds(optL, lb);
	nlopt_set_upper_bounds(optL, ub);

	nlopt_set_min_objective(optG, estimation::errorRigCyDiff, NULL);
	nlopt_set_min_objective(optL, estimation::errorRigCyDiff, NULL);

	//nlopt_set_min_objective(optG, estimation::errorRigCyDiff, NULL);
	//nlopt_set_min_objective(optL, estimation::errorRigCyDiff, NULL);

	nlopt_add_inequality_constraint(optG, estimation::constraintErrorRigC, NULL, 1e-8);
	nlopt_add_inequality_constraint(optL, estimation::constraintErrorRigC, NULL, 1e-8);

	nlopt_set_ftol_abs(optG, 1e-5);
	nlopt_set_ftol_abs(optL, 1e-8);
	nlopt_set_maxeval(optL, 1000);
	double minError;
	if (nlopt_optimize(optG, paramCRig, &minError) < 0) {
		printf("nlopt failed!\n");
	}
	else {
		printf("found minimum = %0.10g\n", minError);
		cout << " -------------- finalPointRig" << endl;
		for (int i = 0; i < 8; i++){
			cout << paramCRig[i] << "\n";
		}
	}


	if (nlopt_optimize(optL, paramCRig, &minError) < 0) {
		printf("nlopt failed!\n");
	}
	else {
		printf("found minimum = %0.10g\n", minError);
		cout << " -------------- finalPointRig" << endl;
		for (int i = 0; i < 8; i++){
			cout << paramCRig[i] << "\n";
		}
	}

	estimation::convertC2A(paramCRig, finalPointRig);

	if (axis == 0){
		(finalPointRig).save("RigTFAz.txt", raw_ascii);
	}
	if (axis == 1){
		(finalPointRig).save("RigTFEl.txt", raw_ascii);
	}

}

void estimation::optimizeRes(){
	cout << " -------------- startingPointRes" << endl;
	cout << startingPointRes;

	double lb[9] = { -3, -3, -3, -3, -3, -3, -3, -3, -3 };
	double ub[9] = { 3, 3, 3, 3, 3, 3, 3, 3, 3 };
	estimation::convertA2C(paramCRes, startingPointRes);

	nlopt_opt optG;
	nlopt_opt optL;

	optG = nlopt_create(NLOPT_GN_CRS2_LM, 9);
	nlopt_set_population(optG, 200);
	optL = nlopt_create(NLOPT_LN_BOBYQA, 9);

	nlopt_set_lower_bounds(optG, lb);
	nlopt_set_upper_bounds(optG, ub);
	nlopt_set_lower_bounds(optL, lb);
	nlopt_set_upper_bounds(optL, ub);

	nlopt_set_min_objective(optG, estimation::errorResCyDiff, NULL);
	nlopt_set_min_objective(optL, estimation::errorResCyDiff, NULL);

	//nlopt_set_min_objective(optG, estimation::errorResCyDiff, NULL);
	//nlopt_set_min_objective(optL, estimation::errorResCyDiff, NULL);

	nlopt_add_inequality_constraint(optG, estimation::constraintErrorResC, NULL, 1e-8);
	nlopt_add_inequality_constraint(optL, estimation::constraintErrorResC, NULL, 1e-8);

	nlopt_set_xtol_rel(optG, 1e-6);
	nlopt_set_xtol_rel(optL, 1e-10);

	double minError;
	/*if (nlopt_optimize(optG, paramCRes, &minError) < 0) {
		printf("nlopt failed!\n");
	}
	else {
		printf("found minimum = %0.10g\n", minError);
		cout << " -------------- finalPointRes" << endl;
		for (int i = 0; i < 9; i++){
			cout << paramCRes[i] << "\n";
		}
	}*/


	if (nlopt_optimize(optL, paramCRes, &minError) < 0) {
		printf("nlopt failed!\n");
	}
	else {
		printf("found minimum = %0.10g\n", minError);
		cout << " -------------- finalPointRes" << endl;
		for (int i = 0; i < 9; i++){
			cout << paramCRes[i] << "\n";
		}
	}

	estimation::convertC2A(paramCRes, finalPointRes);

	if (axis == 0){
		(finalPointRes).save("ResTFAz.txt", raw_ascii);
	}
	if (axis == 1){
		(finalPointRes).save("ResTFEl.txt", raw_ascii);
	}

}

void estimation::calculateYRig(){
	yRig.set_size(u.n_rows);
	yRig.zeros();

	for (int i = 2; i < u.n_rows; i++){

		//yRig(i) = finalPointRig(0)*uInt(i) + finalPointRig(1)*uInt(i - 1) - finalPointRig(6)*yRig(i - 1) - finalPointRig(7)*yRig(i - 2);
		yRig(i) = finalPointRig(0)*uInt(i-1) + finalPointRig(1)*uInt(i - 2) - finalPointRig(6)*yRig(i - 1) - finalPointRig(7)*yRig(i - 2);
	}
	(yRig).save("yRig.txt", raw_ascii);
}

void estimation::calculateYRigyDiff(){
	yRigyDiff.set_size(u.n_rows-1);
	yRigyDiff.zeros();

	for (int i = 2; i < (u.n_rows-1); i++){

		//yRig(i) = finalPointRig(0)*uInt(i) + finalPointRig(1)*uInt(i - 1) - finalPointRig(6)*yRig(i - 1) - finalPointRig(7)*yRig(i - 2);
		yRigyDiff(i) = finalPointRig(0)*uShift(i - 1) + finalPointRig(1)*uShift(i - 2) - finalPointRig(6)*yRigyDiff(i - 1) - finalPointRig(7)*yRigyDiff(i - 2);
	}
	(yRigyDiff).save("yRigyDiff.txt", raw_ascii);
}

void estimation::highpassFilter(){
	column_vector_rig filtInput;
	Mat<double> chebySos;


	chebySos <<0.994094721291412<<-1.98818944258282<<0.994094721291412<<1<<-1.99688617500281<<0.997846841164197<<endr
		     <<0.991347325673526<<-1.98269465134705<<0.991347325673526<<1<<-1.99126432987649<<0.992428378211037<<endr
			 <<0.986009962664366<<-1.97201992532873<<0.986009962664366<<1<<-1.98023450482040<<0.982010206637331<<endr
			 <<0.970653704443535<<-1.94130740888707<<0.970653704443535<<1<<-1.94839365212324<<0.952142501381536<<endr
			 <<0.908010243576930<<-1.81602048715386<<0.908010243576930<<1<<-1.81841024173674<<0.830395471990924<<endr;


	yRigFilt.set_size(u.n_rows);
	yTotFilt.set_size(u.n_rows);
	filtInput.set_size(u.n_rows);


	filtInput = yRig;
	for (int j = 0; j < 5; j++){

		yRigFilt(0) = 0;
		yRigFilt(1) = 0;

		for (int i = 2; i < filtInput.n_rows; i++){

			yRigFilt(i) = chebySos(j, 0)*filtInput(i) + chebySos(j, 1)*filtInput(i - 1) + chebySos(j, 2)*filtInput(i - 2) - chebySos(j, 4)*yRigFilt(i - 1) - chebySos(j, 5)*yRigFilt(i - 2);

		}

		filtInput = yRigFilt;
	}





	filtInput = y;
	for (int j = 0; j < 5; j++){

		yTotFilt(0) = 0;
		yTotFilt(1) = 0;

		for (int i = 2; i < filtInput.n_rows; i++){

			yTotFilt(i) = chebySos(j, 0)*filtInput(i) + chebySos(j, 1)*filtInput(i - 1) + chebySos(j, 2)*filtInput(i - 2) - chebySos(j, 4)*yTotFilt(i - 1) - chebySos(j, 5)*yTotFilt(i - 2);

		}

		filtInput = yTotFilt;
	}




	(yRigFilt).save("yRigFilt.txt", raw_ascii);
	(yTotFilt).save("yTotFilt.txt", raw_ascii);
}
void estimation::highpassFilteryDiff(){
	column_vector_rig filtInput;
	Mat<double> chebySos;


	chebySos << 0.994094721291412 << -1.98818944258282 << 0.994094721291412 << 1 << -1.99688617500281 << 0.997846841164197 << endr
		<< 0.991347325673526 << -1.98269465134705 << 0.991347325673526 << 1 << -1.99126432987649 << 0.992428378211037 << endr
		<< 0.986009962664366 << -1.97201992532873 << 0.986009962664366 << 1 << -1.98023450482040 << 0.982010206637331 << endr
		<< 0.970653704443535 << -1.94130740888707 << 0.970653704443535 << 1 << -1.94839365212324 << 0.952142501381536 << endr
		<< 0.908010243576930 << -1.81602048715386 << 0.908010243576930 << 1 << -1.81841024173674 << 0.830395471990924 << endr;


	yRigyDiffFilt.set_size(u.n_rows);
	yRigyDiffFilt.zeros();

	yDiffFilt.set_size(u.n_rows);
	yDiffFilt.zeros();

	filtInput.set_size(u.n_rows);
	filtInput.zeros();


	filtInput = yRigyDiff;
	for (int j = 0; j < 5; j++){

		yRigyDiffFilt(0) = 0;
		yRigyDiffFilt(1) = 0;

		for (int i = 2; i < filtInput.n_rows; i++){

			yRigyDiffFilt(i) = chebySos(j, 0)*filtInput(i) + chebySos(j, 1)*filtInput(i - 1) + chebySos(j, 2)*filtInput(i - 2) - chebySos(j, 4)*yRigyDiffFilt(i - 1) - chebySos(j, 5)*yRigyDiffFilt(i - 2);

		}

		filtInput = yRigyDiffFilt;
	}





	filtInput = yDiff;
	for (int j = 0; j < 5; j++){

		yDiffFilt(0) = 0;
		yDiffFilt(1) = 0;

		for (int i = 2; i < filtInput.n_rows; i++){

			yDiffFilt(i) = chebySos(j, 0)*filtInput(i) + chebySos(j, 1)*filtInput(i - 1) + chebySos(j, 2)*filtInput(i - 2) - chebySos(j, 4)*yDiffFilt(i - 1) - chebySos(j, 5)*yDiffFilt(i - 2);

		}

		filtInput = yDiffFilt;
	}




	(yRigyDiffFilt).save("yRigyDiffFilt.txt", raw_ascii);
	(yDiffFilt).save("yDiffFilt.txt", raw_ascii);
}