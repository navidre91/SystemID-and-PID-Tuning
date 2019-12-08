#pragma once

#include <iostream>
#include "stdafx.h"
#include <armadillo>
#include <complex>
#include <iomanip>
#include <cmath>
#include <dlib/optimization.h>
#include <fstream>
#include "estimation.h"


using namespace std;
using namespace dlib;
using namespace arma;

typedef dlib::matrix<double, 0, 1> column_vector;


double modelRig::operator()(
	const column_vector& x
	) const {
	return estimation::errorRig(x);
}

void modelRig::get_derivative_and_hessian(
	const column_vector& x,
	column_vector& der,
	general_matrix& hess
	) const
{
	der = estimation::derivativeRig(x);
	hess = estimation::hessianRig(x);
}