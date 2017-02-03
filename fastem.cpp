#define _USE_MATH_DEFINES 

#include <Mex/Mex.h>
#include <cmath>
#include <iostream>
#include <chrono>
#include <random>

/* FastEM by Gerhard Kurz
 * 
 * Usage:
 * [means, covariances, weights] = fastem(data, sampleWeights, n)
 * Arguments:
 *      data
 *          samples as d x m Matrix (d dimensions, m samples)
 *      sampleWeights 
 *          1 x m row vector
 *      n
 *          number of Gaussian components to fit
 * Returns:
 *      means 
 *          d x n matrix with mean vectors
 *      covariances
 *          d x d x n tensor with covariance matrices
 *      weights
 *          1 x n vector with weight of each Gaussian component
 */


Eigen::VectorXd logmvnpdf(const Mex::ConstMatrixXd &X, const Eigen::VectorXd &mu, const Eigen::MatrixXd &C){
    // Logarithm of multivariate normal distrbution probability density
    const int dimension = X.rows();
    
    const Eigen::MatrixXd L = C.llt().matrixL().transpose(); // Cholesky decomposition

    const double det = L.diagonal().prod(); //determinant of L is equal to square rooot of determinant of C
    const double lognormconst = -log(2 * M_PI)*dimension/2 - log(fabs(det));

    const Eigen::MatrixXd X0 = (X.transpose().rowwise() - mu.transpose())*L.inverse();
    const Eigen::VectorXd result = (X0.rowwise().squaredNorm()).array() * (-0.5) + lognormconst;    
    
    return result;
}

Eigen::VectorXd logsumexp(const Eigen::MatrixXd &x){
    // Computes the logarithm of the sum of the exponential of the entries of a given matrix (for each column)
    // Subtracts the maximum first to prevent over/underflows
    Eigen::VectorXd result;
    Eigen::VectorXd max = x.rowwise().maxCoeff();
    result = max.array() + (x.colwise()-max).array().exp().rowwise().sum().array().log();
    
    for(int i=0; i<result.cols(); i++){
        if (result(i) == std::numeric_limits<double>::infinity() || result(i) == -std::numeric_limits<double>::infinity()){
            result(i) = max(i);
        }
    }

    return result;
}

void kmeansplusplus(const Mex::ConstMatrixXd &samples, const Mex::ConstRowVectorXd &sampleWeights, const int nGauss, 
        Mex::OutputMatrixXd &means){
    // Performs kmeans++ initialization
    const int nSamples = samples.cols(); 
    
    //std::ostringstream sout;
 
    std::random_device rd;     // only used once to initialise (seed) engine
    std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
    
    // Choose first mean
    std::uniform_int_distribution<int> uni(0,nSamples-1); 
    const int firstMeanIndex = uni(rng);
    means.col(0) = samples.col(firstMeanIndex);
    
    for(int k=1; k<nGauss; k++){
        //compute squared distances
        Eigen::MatrixXd distances(k, nSamples); //distances(i,j) contains squared distance from sample j to component mean i
                
        for(int j=0; j<k; j++){ //iterate over component means
            distances.row(j) = (samples.colwise() - means.col(j)).colwise().squaredNorm();
        }
               
        Eigen::RowVectorXd minDistance = distances.colwise().minCoeff();
        
        //coose new center
        std::uniform_real_distribution<double> uni2(0,minDistance.sum());
        const double r = uni2(rng);
        double sum = 0;
        int m;
        for(m=0; m<nSamples; m++){
            sum = sum + minDistance(m);
            if (sum>r){
                break;
            }
        }        
        
        means.col(k) = samples.col(m);
    }
}

void emalgo(Mex::ConstMatrixXd &samples, Mex::ConstRowVectorXd &sampleWeights, int nGauss, 
        Mex::OutputMatrixXd &means, Mex::OutputMatrixXDXd &covariances, Mex::OutputVectorXd &weights){
    // Expectation Maximization algorithm
    
    const int dim = samples.rows();
    const int nSamples = samples.cols();
    
    Eigen::MatrixXd gamma{nSamples,nGauss}; //gamma(i,j) = weight for sample i, component j
    
    std::ostringstream sout;
    
    //todo error handling/covariance regularization
   
    //init
    //initialize covariances with sample covariance?
    
    //todo normalize sampleweights
    for(int iGauss=0; iGauss<nGauss; iGauss++){
        //means.col(iGauss) = samples.col(iGauss); //todo randomize
        covariances.slice(iGauss) = Eigen::MatrixXd::Identity(dim, dim);
        weights(iGauss) = 1.0/nGauss;
    }
    kmeansplusplus(samples, sampleWeights, nGauss, means);
    
    const int maxIter = 10; //max number of iterations
    double oldloglikelihood = -DBL_MAX;
    const double threshold = 1E-5; //abort when likelihood improvement is less than threshold
    const double reg = 1E-6; //regularization to prevent numerical issues: maybe choose depending on data?
    
    for(int iter=0; iter<maxIter; iter++){
        std::chrono::time_point<std::chrono::high_resolution_clock> start;
        start = std::chrono:: high_resolution_clock::now();

        //E Step
        //#pragma omp parallel for
        for(int iGauss=0; iGauss<nGauss; iGauss++){ //for each Gaussian component
            gamma.col(iGauss) = logmvnpdf(samples, (Eigen::VectorXd)(means.col(iGauss)), (Eigen::MatrixXd)(covariances.slice(iGauss)));
            gamma.col(iGauss).array() += log(weights(iGauss));
        }
        
        //sout << "gamma:\n" << gamma << std::endl;
       
        std::chrono::duration<double> elapsed_seconds = std::chrono::high_resolution_clock::now()-start;
        sout << "E step time: " << elapsed_seconds.count() << "s\n";        
        
        start = std::chrono:: high_resolution_clock::now();
        //loglikelihood
        Eigen::VectorXd lltemp = logsumexp(gamma); //gamma.array().exp().rowwise().sum().array().log();
        //sout << "lltemp:\n" << lltemp << std::endl;
        const double loglikelihood = (lltemp.transpose().array() * sampleWeights.array()).sum()*nSamples;
        sout << iter << " " << loglikelihood << std::endl;
        if(loglikelihood < oldloglikelihood + threshold && iter > 1){
            //check convergence
            break;
        }
        oldloglikelihood = loglikelihood;
        elapsed_seconds = std::chrono::high_resolution_clock::now()-start;
        sout << "ll time: " << elapsed_seconds.count() << "s\n";  
        
        start = std::chrono:: high_resolution_clock::now();
        //normalize rows
        Eigen::VectorXd gammaRowSumLog = lltemp;//gamma.rowwise().sum();
        for(int iSample=0; iSample<nSamples; iSample++){
            gamma.row(iSample).array() -= gammaRowSumLog(iSample);
        }
        gamma = gamma.array().exp();
        elapsed_seconds = std::chrono::high_resolution_clock::now()-start;
        sout << "normalization time: " << elapsed_seconds.count() << "s\n";
        
        //sout << "gamma:\n" << gamma << std::endl;

        //M Step
        Eigen::RowVectorXd weightsNew(nGauss);
        for(int iGauss=0; iGauss<nGauss; iGauss++){
            start = std::chrono:: high_resolution_clock::now();
            
            Eigen::RowVectorXd currentWeights = gamma.col(iGauss).transpose().array() * sampleWeights.array();
            //sout << "currentWeights:\n" << currentWeights << std::endl;
            const double Nk = currentWeights.sum();
            //sout << "Nk:\n" << Nk << std::endl;
            
            elapsed_seconds = std::chrono::high_resolution_clock::now()-start;
            sout << "M Step weights time: " << elapsed_seconds.count() << "s\n";
            
            start = std::chrono:: high_resolution_clock::now();
            means.col(iGauss) = 1.0/Nk * (currentWeights.replicate(dim,1).array() * samples.array()).rowwise().sum();
            //sout << "mean:\n" << means.col(iGauss) << std::endl;
            elapsed_seconds = std::chrono::high_resolution_clock::now()-start;
            sout << "M Step mean time: " << elapsed_seconds.count() << "s\n";
            
            start = std::chrono:: high_resolution_clock::now();
            Eigen::MatrixXd shiftedSamples = samples.colwise() - means.col(iGauss);
            /*covariances.slice(iGauss).setZero();
            for(int iSample=0; iSample<nSamples; iSample++){
                covariances.slice(iGauss) += currentWeights(iSample) * shiftedSamples.col(iSample) * shiftedSamples.col(iSample).transpose();
            }
            covariances.slice(iGauss) /= Nk;*/
            //sout << covariances.slice(iGauss) << std::endl;
            Eigen::MatrixXd shiftedSamplesWeighted = currentWeights.replicate(dim,1).array() * shiftedSamples.array();
            covariances.slice(iGauss) = 1.0/Nk * shiftedSamplesWeighted * shiftedSamples.transpose() + reg*Eigen::MatrixXd::Identity(dim, dim);
            //sout << covariances.slice(iGauss) << std::endl;
            elapsed_seconds = std::chrono::high_resolution_clock::now()-start;
            sout << "M Step cov time: " << elapsed_seconds.count() << "s\n";
            
            weightsNew(iGauss) = Nk;
        }
        weights = weightsNew;
        
    }
    //printf("%s", sout.str().c_str()) ;
    sout.str("");
}


void mexFunction(int numOutputs, mxArray *outputs[],
                 int numInputs, const mxArray *inputs[])
{
    try {
        /* Check for proper number of arguments */
        // todo better usage documentation
        if (numInputs != 3) {
            throw std::invalid_argument("Three inputs are required.");
        }
        
        if (numOutputs != 3) {
            throw std::invalid_argument("Three outputs is required.");
        }
        
        Mex::ConstMatrixXd samples(inputs[0]);
        
        const unsigned int dim = samples.rows();
        const unsigned int numSamples = samples.cols();
       
        Mex::ConstRowVectorXd sampleWeights(inputs[1]);

        if (sampleWeights.cols() != numSamples) {
            throw std::invalid_argument("invalid weights");
        }
               
        const unsigned int n = *mxGetPr(inputs[2]);
        
        if (numSamples < n) {
            throw std::invalid_argument("Need more samples than Gaussian components.");
        }        
                
        Mex::OutputMatrixXd means{dim,n};
        Mex::OutputMatrixXDXd covariances{dim,dim,n};
        Mex::OutputVectorXd weights{n};
        
        // run EM algorithm
        emalgo(samples, sampleWeights, n, means, covariances, weights);
                
        // assign outputs
        outputs[0] = means;
        outputs[1] = covariances;
        outputs[2] = weights;
    } catch (std::exception& ex) {
        mexErrMsgTxt(ex.what());
    }
}