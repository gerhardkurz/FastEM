FastEM for Matlab
=================

FastEM is an efficient implementation of the expectation maximization (EM) algorithm in MATLAB.

Requirements
------------

  * a reasonably recent version of MATLAB
  * a C++ Compiler (tested with Visual Studio 2015)
  * [https://bitbucket.org/nonlinearestimation/toolbox/](Nonlinear Estimation Toolbox)

Features
--------

  * supports weighted samples (most public implementations do not!)
  * kmeans++ initialization
  * fast performance, can make use of OpenMP, SSE, AVX using Eigen
  * avoids overflows and underflows by use of logarithmic representation
  * avoids degeneration of covariance by use of regularization

Example usage
-------------
First, compile the mex file by adding the Nonlinear Estimation Toolbox to the MATLAB path and then calling
	>> compile



License
-------

FastEM is licensed under the GPLv3 license.

Contact
-------

Author: Gerhard Kurz

Mail: gerhard.kurz (at) kit (dot) edu

Web: [http://isas.uka.de/User:Kurz](http://isas.uka.de/User:Kurz)

