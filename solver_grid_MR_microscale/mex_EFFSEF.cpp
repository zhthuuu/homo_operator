//  main.cpp
//
//  Created by Brian Staber on 18/03/(double)2016.
//  Copyright © 2016 Brian Staber. All rights reserved.

#include <iostream>
#include <math.h>
#include "mex.h"
#include <vector>
#include </opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3/Eigen/Dense>
#include </opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3/Eigen/Sparse>

#define t2D(A, i, j, N) (A[ (j)*(N) + (i) ])

void mexFunction(int          nlhs,
                 mxArray      *plhs[],
                 int          nrhs,
                 const mxArray *prhs[])
{
        
    /* Check for proper number of arguments */
    if (nrhs != 7) {
        mexErrMsgIdAndTxt("MATLAB:mexcpp:nargin",
                          "MEXCPP requires seven input arguments.");
    } else if (nlhs != 1) {
        mexErrMsgIdAndTxt("MATLAB:mexcpp:nargout",
                          "MEXCPP requires 1 output argument.");
    }
    
    double *DISPTD, *PROP;
    double A, We, J, alpha1, beta1, s1, s2, WEFF, Ae, s;
    int32_T *PHASES, *DOFe;
    int NE, phaseElem;
    
    DISPTD = (double *) mxGetPr(prhs[0]);
    NE = (int) mxGetScalar(prhs[1]);
    PROP = (double *) mxGetPr(prhs[2]);
    PHASES = (int32_T *) mxGetData(prhs[3]);
    s = (double) mxGetScalar(prhs[4]);
    Ae = (double) mxGetScalar(prhs[5]);
    DOFe = (int32_T *) mxGetData(prhs[6]);
    
    Eigen::Matrix<double,2,2> I, F, C;
    Eigen::Matrix<double,2,4> U;
    Eigen::Matrix<double,4,2> D;
    Eigen::Matrix<double,8,1> uE;
    
    I << 1.0, 0.0,
         0.0, 1.0;
    
    A = 0;
    WEFF = 0;
    
    for (unsigned int e=0;e<NE;e++)
        {
        
        phaseElem = PHASES[e];
        
        alpha1 = PROP[phaseElem-1];
        beta1 = PROP[phaseElem+1];
        s1 = PROP[phaseElem+3];
        s2 = 2*(alpha1+2*beta1);
        
        for (unsigned int j=0;j<8;j++){
            uE[j]=DISPTD[t2D(DOFe,j,e,8)-1];
        }
        
        U << uE[0], uE[2], uE[4], uE[6],
             uE[1], uE[3], uE[5], uE[7];
        
        D << -1, -1,
              1, -1,
              1,  1,
             -1,  1;
        D = D*s/2;
    
        F = U*D + I;
        J = F.determinant();
        C = F.transpose()*F;
        
        We = alpha1*(C.trace() - 2) + 
                beta1*(C(0,0)+C(1,1)+C(0,0)*C(1,1)-C(0,1)*C(1,0)-3) + 
                s1*(J-1)*(J-1)/double(2) - s2*log(J);
        
        A += Ae;
        WEFF += Ae*We;
        
        } 
    
    WEFF = WEFF/A;
    plhs[0] = mxCreateDoubleScalar(WEFF);   
    
}

/* Neo-Hookean case
 * lambda = PROP[phaseElem-1];
   mu = PROP[phaseElem+1];
 * We = (mu/(double(2)))*(C.trace() - 2) + (lambda/(double(2)))*(J-1.0)*(J-1.0) - mu*log(J); 
 */
