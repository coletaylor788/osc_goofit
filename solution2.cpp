#include <iostream> 
#include <cmath> 
#include <cassert> 
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include "thrust/transform_reduce.h"
#include "rootstuff/TRandom.hh"
#include "rootstuff/TMinuit.hh" 
#include <fstream>

using namespace thrust; 

///////////////////////////////////////////////////
// Gauss Fit Example
//
// Test the performance of a Gauss fit

struct GaussFunctor : public thrust::unary_function<double, double> {

  GaussFunctor (double m, double s) 
    : mean(m)
    , sigma(s)
  {} 

  __device__ double operator () (double xval) {
    double ret = (xval - mean);
    ret /= sigma;
    ret *= ret;
    ret = exp(-0.5*ret);
    // ret /= sigma;
    // ret /= sqrt(2*M_PI);
	// Same order as on CPU for consistant results
	ret /= (sqrt(2*M_PI) * sigma);
    return ret; 
  }

  double mean;
  double sigma; 
}; 

struct NllFunctor : public thrust::unary_function<double, double> {

  NllFunctor (double m, double s) 
    : mean(m)
    , sigma(s)
  {}

  __device__ double operator () (double xval) {
    // Taking the log of an exp - simplify: 
    double ret = -0.5*pow((xval - mean) / sigma, 2);
    ret -= log(sigma); 
    // Ignore constant term sqrt(2pi)
    return -2*ret; 
  }

  double mean;
  double sigma; 
}; 


device_vector<double>* device_data = 0; 

void fcn_nll (int& npar, double* deriv, double& fVal, double param[], int flag) {
  double mean = param[0];
  double sigma = param[1];

  double initVal = 0; 
  fVal = transform_reduce(device_data->begin(), device_data->end(), 
			  NllFunctor(mean, sigma),
			  initVal, 
			  thrust::plus<double>()); 
}

int main (int argc, char** argv) {
  int sizeOfVector = atoi(argv[1]); 
  // NOTE: This time we will use Thrust, 
  // so we need not limit ourselves to 
  // small vectors - the library will take
  // care of reductions for us.

  //OMP TIME
  double runTime = 0;
  double startTime, endTime;

  double mean = 5;
  double sigma = 3; 

  // Generate a host-side vector and fill it with random numbers. 
  TRandom donram(42); 
  host_vector<double> host_data;
  // EXERCISE: Create a thrust::host_vector and fill it with random numbers. 
  for (int i = 0; i < sizeOfVector; ++i) {
    double currVal = donram.Gaus(mean, sigma); 
    host_data.push_back(currVal); 
  }

  //OMP TIME
  startTime = omp_get_wtime();

  // EXERCISE: Create the device_vector called 'device_data' (use the existing
  //           pointer) and copy the host data into it. 
  device_data = new device_vector<double>(host_data);

  // EXERCISE: Use Thrust to calculate the Gaussian probability of 
  //           each generated event, storing the results in another device_vector.
  device_vector<double> device_results(device_data->size());
  thrust::transform(device_data->begin(), device_data->end(), device_results.begin(), GaussFunctor(mean, sigma)); 
  // EXERCISE: Copy the result back to the host side. Compare the numbers with
  //           ones calculated by the CPU. 
  host_vector<double> host_results = device_results;

  //OMP TIME
  endTime = omp_get_wtime();
  runTime += endTime - startTime;

  double tolerance = 1e-6; 
  double cpusum = 0; 
 
  for (unsigned int i = 0; i < host_data.size(); ++i) {
    double currVal = exp(-0.5*(pow((host_data[i] - mean) / sigma, 2))) / (sigma * sqrt(2*M_PI));
	cpusum += currVal; 
    if (fabs(currVal - host_results[i]) < tolerance) continue;
    std::cout << "Problem with event " << i << ": "
	      << currVal << " != " << host_results[i] << std::endl;  
  }

  //OMP TIME
  startTime = omp_get_wtime();

  double gpusum = reduce(device_results.begin(), device_results.end());
  if (fabs(gpusum - cpusum) > tolerance)
    std::cout << "Problem with sums: " << gpusum << " != " << cpusum << std::endl; 
  TMinuit minuit(2);
  minuit.DefineParameter(0, "mean", mean,   0.01, -10.0, 10.0); 
  minuit.DefineParameter(1, "sigma", sigma, 0.01,   0.0, 10.0); 
 
  minuit.SetFCN(fcn_nll);
  minuit.Migrad(); 

  //OMP TIME
  endTime = omp_get_wtime();
  runTime += endTime - startTime;
  ofstream myfile;
  myfile.open("ompTimes.txt", ios::app);
  myfile << "Input Size: " << sizeOfVector << " , Run Time: " << runTime << "\n";
  myfile.close();


  // Cleanup. 
  if (device_data) delete device_data; 
  
  return 0;
}
