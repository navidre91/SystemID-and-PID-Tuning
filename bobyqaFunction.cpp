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

typedef dlib::matrix<double, 0, 1> column_vector_rig;

bobyqaFunction::bobyqaFunction(
	const column_vector_rig& input
	)
{
	target = input;
}

double bobyqaFunction::operator() (const column_vector_rig& arg) const
{
	column_vector_rig temp;
	temp=estimation::errorRigVect(arg);
	return mean(squared(target - temp));
}