#pragma once

#include <iostream>
#include "stdafx.h"
#include <armadillo>
#include <complex>
#include <iomanip>
#include <cmath>
#include <dlib/optimization.h>
#include <fstream>


using namespace std;
using namespace dlib;
using namespace arma;

typedef dlib::matrix<double, 0, 1> column_vector_rig;

class bobyqaFunction{
public:

	bobyqaFunction(
		const column_vector_rig& input
		);

	double operator() (const column_vector_rig& arg) const;
private:
	column_vector_rig target;
};