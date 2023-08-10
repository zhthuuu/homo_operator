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
    if (nrhs != 8) {
        mexErrMsgIdAndTxt("MATLAB:mexcpp:nargin",
                          "MEXCPP requires eight input arguments.");
    } else if (nlhs != 1) {
        mexErrMsgIdAndTxt("MATLAB:mexcpp:nargout",
                          "MEXCPP requires no output argument.");
    }
    
    double *DISPTD, *sb, *sc, *Ae, *bulk, *shear;
    double A, We, J, lambda, mu, WEFF;
    int32_T *DOFe;
    int NE;
    
    DISPTD = (double *) mxGetPr(prhs[0]);
    NE = (int) mxGetScalar(prhs[1]);
    bulk = (double *) mxGetPr(prhs[2]);
    shear = (double *) mxGetPr(prhs[3]);
    sb = (double *) mxGetPr(prhs[4]);
    sc = (double *) mxGetPr(prhs[5]);
    Ae = (double *) mxGetPr(prhs[6]);
    DOFe = (int32_T *) mxGetData(prhs[7]);
    
    Eigen::Matrix<double,2,2> I, F, C;
    Eigen::Matrix<double,2,3> U;
    Eigen::Matrix<double,3,2> D;
    Eigen::Matrix<double,6,1> uE;
    
    I << 1.0, 0.0,
         0.0, 1.0;
    
    A = 0;
    WEFF = 0;
    
    for (unsigned int e=0;e<NE;e++)
    {

        lambda = bulk[e];
        mu = shear[e];
        We = (mu/(double(2)))*(C.trace() - 2) + (lambda/(double(2)))*(J-1.0)*(J-1.0) - mu*log(J); 
        
        for (unsigned int j=0;j<6;j++){
            uE[j]=DISPTD[t2D(DOFe,j,e,6)-1];
        }
        
        U << uE[0], uE[2], uE[4],
             uE[1], uE[3], uE[5];
        
        D << t2D(sb,0,e,3), t2D(sc,0,e,3),
             t2D(sb,1,e,3), t2D(sc,1,e,3),
             t2D(sb,2,e,3), t2D(sc,2,e,3);
    
        F = U*D + I;
        J = F.determinant();
        C = F.transpose()*F;
        
        We = (mu/(double(2)))*(C.trace() - 2) + (lambda/(double(2)))*(J-1.0)*(J-1.0) - mu*log(J); 
        
        A += Ae[e];
        WEFF += Ae[e]*We;
        
        } 
    
    WEFF = WEFF/A;
    plhs[0] = mxCreateDoubleScalar(WEFF);   
    
}

/* Neo-Hookean case
 * lambda = PROP[phaseElem-1];
   mu = PROP[phaseElem+1];
 * We = (mu/(double(2)))*(C.trace() - 2) + (lambda/(double(2)))*(J-1.0)*(J-1.0) - mu*log(J); 
 */

/* Mooney-Rivlin case
    phaseElem = PHASES[e];
    alpha1 = PROP[phaseElem-1];
    beta1 = PROP[phaseElem+1];
    s1 = PROP[phaseElem+3];
    s2 = 2*(alpha1+2*beta1);
    We = alpha1*(C.trace() - 2) + 
            beta1*(C(0,0)+C(1,1)+C(0,0)*C(1,1)-C(0,1)*C(1,0)-3) + 
            s1*(J-1)*(J-1)/double(2) - s2*log(J);
*/  
