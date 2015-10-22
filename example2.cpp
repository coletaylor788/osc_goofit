#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include "thrust/transform_reduce.h"
#include "rootstuff/TRandom.hh"
#include "rootstuff/TMinuit.hh" 
#include <iostream>
#include <fstream>

using namespace thrust; 

//////////////////////////////////////////////////////
// Chi Square Fit
//
// Test performance of Chi square fit

// Pointers to avoid crash on exit. 
device_vector<double>* dev_yvals;
device_vector<double>* dev_xvals;

// Note that 'unary_function' is ambiguous with STL class of same name. 
struct ChisqFunctor : public thrust::unary_function<tuple<double, double>, double> {
  ChisqFunctor (double q, double l, double c, double e) 
    : quad(q)
    , line(l)
    , cons(c)
    , oneOverError(e)
  {}

  __device__ double operator () (tuple<double, double> xypair) {
    // Extract x and y from tuple
    double xval = get<0>(xypair); 
    double yval = get<1>(xypair);

    // Calculate expected y
    double expected = quad * xval * xval;
    expected       += line * xval;
    expected       += cons; 

    // Chisquare
    double chisq = (expected - yval); 
    chisq *= chisq; 
    chisq *= oneOverError; // Faster than division with hardcoded error
    return chisq; 
  }

  double quad;
  double line;
  double cons;
  double oneOverError; 
}; 

void chisq (int& npar, double* deriv, double& fVal, double param[], int flag) {
  // Extract parameters from MINUIT and put into ChisqFunctor constructor. 
  double quad = param[0];
  double line = param[1];
  double cons = param[2]; 
  ChisqFunctor functor(quad, line, cons, 1/0.1); 
  // Note hardcoded error!

  double initVal = 0; 
  fVal = transform_reduce(make_zip_iterator(make_tuple(dev_xvals->begin(), dev_yvals->begin())),
			  make_zip_iterator(make_tuple(dev_xvals->end(), dev_yvals->end())),
			  functor, 
			  initVal, 
			  thrust::plus<double>()); // 'plus' also exists in STL. 

  //std::cout << fVal << " " << quad << " " << line << " " << cons << std::endl; 
}

int main (int argc, char** argv) {
  int sizeOfVector = atoi(argv[1]);

  // Generate random data
  TRandom donram(42); 
  host_vector<double> host_yvals;
  host_vector<double> host_xvals;
  for (int i = 0; i < sizeOfVector; ++i) {
    double currX = 0.1*i;
    double currVal =  2.0*currX*currX;
    currVal       -=  0.8*currX;
    currVal       +=  3.2; 
    host_yvals.push_back(currVal + donram.Gaus(0.0, 0.1));
    host_xvals.push_back(currX);
  }

  double startTime = omp_get_wtime();

  // Move to device
  dev_yvals = new device_vector<double>(host_yvals); 
  dev_xvals = new device_vector<double>(host_xvals); 
  
  // Fit to degree-two polynomial 
  TMinuit minuit(3); 
  minuit.DefineParameter(0, "quad", 1.8, 0.01, -4.0, 4.0); 
  minuit.DefineParameter(1, "line", 1.0, 0.01, -4.0, 4.0); 
  minuit.DefineParameter(2, "cons", 3.0, 0.01, -4.0, 4.0); 
  minuit.SetFCN(chisq);
  minuit.Migrad(); 

  double endTime = omp_get_wtime();
  double runTime = endTime - startTime;
  ofstream myfile;
  myfile.open("ompTimes.txt", ios::app);
  myfile << "Input Size: " << sizeOfVector << " , Run Time: " << runTime << "\n";
  myfile.close();

  // Free the device memory. 
  delete dev_yvals;
  delete dev_xvals; 
  return 0; 
}
