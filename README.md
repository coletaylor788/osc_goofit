# osc_goofit
Highlights of my summer internship at OSC working on GooFit

During the summer of 2013, I optimized two different graph fitting algorithms for various devices. I modified existing CUDA code for a Gaussian Fit and a Chi Square Distribution to be run on various devices. Using the thrust library, I modified the code so that it could be run on GPUs using a CUDA backend, CPUs using an OpenMP backend, and on a Xeon Phi. I spent time testing various optimizations, getting it to work on the new Xeon Phi architecture, making code optimizations, and testing various compiler optimizations to increase performance.

<b>Thrust OpenMP Backend.pdf<b>: The results of my project

<b>example2.cpp</b>: The Chi Square Distribution code using thrust

<b>solution2.cpp</b>: The Gaussian Fit Distribution code using thrust
