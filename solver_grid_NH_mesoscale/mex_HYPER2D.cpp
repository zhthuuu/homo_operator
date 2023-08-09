//  main.cpp
//
//  Created by Brian Staber on 18/03/(double)2016.
//  Copyright © 2016 Brian Staber. All rights reserved.

#include <iostream>
#include <math.h>
#include <ctime>
#include <vector>
#include </opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3/Eigen/Dense>
#include </opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3/Eigen/Sparse>
#include "mex.h"

#define t2D(A, i, j, N) (A[ (j)*(N) + (i) ])

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    
    double *BULK, *SHEAR, *DISPTD;
    double s, Ae, lambda, mu;
    int32_T *DOFe;
    int NE, GDOF;
    
    /* Check for proper number of arguments */
    if (nrhs != 8) {
        mexErrMsgIdAndTxt("MATLAB:mexcpp:nargin",
                          "MEXCPP requires eight input arguments.");
    } else if (nlhs != 2) {
        mexErrMsgIdAndTxt("MATLAB:mexcpp:nargout",
                          "MEXCPP requires no output argument.");
    }
    /* ------------------------------------- */
    
    BULK = (double *) mxGetPr(prhs[0]);
    SHEAR = (double *) mxGetPr(prhs[1]);
    NE = (int) mxGetScalar(prhs[2]);
//     PHASES = (int32_T *) mxGetData(prhs[3]);
//     sb = (double *) mxGetPr(prhs[3]);
//     sc = (double *) mxGetPr(prhs[4]);
    s = (double) mxGetScalar(prhs[3]);
    Ae = (double) mxGetScalar(prhs[4]);
    GDOF = (int) mxGetScalar(prhs[5]);
    DOFe = (int32_T *) mxGetData(prhs[6]);
    DISPTD = (double *) mxGetPr(prhs[7]);
    
    double *X, *Y;
    double J, C1, C2, Aire;
    Eigen::Matrix<double,2,2> I, F, C, L, S;
    Eigen::Matrix<double,2,4> U;
    Eigen::Matrix<double,4,2> D;
    Eigen::Matrix<double,8,1> uE;
    Eigen::Matrix<double,3,8> BD;
    Eigen::Matrix<double,4,8> BG;
    Eigen::Matrix<double,3,3> K; // previous 3,3
    Eigen::Matrix<double,4,4> G; // do not know
    Eigen::Matrix<double,8,8> GKFe; 
    Eigen::Matrix<double,8,1> FORCEe;
    Eigen::Matrix<double,3,1> vecS; //previous 3,1
    
    plhs[0] = mxCreateDoubleMatrix(64*NE,1,mxREAL);
    plhs[1] = mxCreateDoubleMatrix(8*NE,1,mxREAL);
    
    X = mxGetPr(plhs[0]);
    Y = mxGetPr(plhs[1]);

    I << 1.0, 0.0,
         0.0, 1.0;
    
    for (int e=0;e<NE;e++)
    {
        Aire = Ae;
        
        lambda = BULK[e];
        mu = SHEAR[e];
        
        for (unsigned int j=0;j<8;j++)
        {
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
        L = C.inverse();
        
        BD << F(0,0)*D(0,0), F(1,0)*D(0,0), F(0,0)*D(1,0), F(1,0)*D(1,0), F(0,0)*D(2,0), F(1,0)*D(2,0), F(0,0)*D(3,0), F(1,0)*D(3,0),
              F(0,1)*D(0,1), F(1,1)*D(0,1), F(0,1)*D(1,1), F(1,1)*D(1,1), F(0,1)*D(2,1), F(1,1)*D(2,1), F(0,1)*D(3,1), F(1,1)*D(3,1),
              F(0,0)*D(0,1)+F(0,1)*D(0,0), F(1,0)*D(0,1)+F(1,1)*D(0,0), F(0,0)*D(1,1)+F(0,1)*D(1,0), F(1,0)*D(1,1)+F(1,1)*D(1,0), F(0,0)*D(2,1)+F(0,1)*D(2,0), F(1,0)*D(2,1)+F(1,1)*D(2,0), F(0,0)*D(3,1)+F(0,1)*D(3,0), F(1,0)*D(3,1)+F(1,1)*D(3,0);
                                                                                                    
        BG << D(0,0), 0.0, D(1,0), 0.0, D(2,0), 0.0, D(3,0), 0.0,
              D(0,1), 0.0, D(1,1), 0.0, D(2,1), 0.0, D(3,1), 0.0,
              0.0, D(0,0), 0.0, D(1,0), 0.0, D(2,0), 0.0, D(3,0),
              0.0, D(0,1), 0.0, D(1,1), 0.0, D(2,1), 0.0, D(3,1);
        
        C1 = lambda*J*(2.0*J-1.0);
        C2 = 2.0*lambda*J*(J-1.0)-2.0*mu;

        K << L(0,0)*L(0,0)*(C1-C2), -C2*L(0,1)*L(0,1) + C1*L(0,0)*L(1,1), L(0,0)*L(0,1)*(C1-C2),
             -C2*L(1,0)*L(1,0) + C1*L(0,0)*L(1,1), L(1,1)*L(1,1)*(C1-C2), L(1,1)*(C1*L(0,1)-C2*L(1,0)),
             L(0,0)*(C1*L(0,1)-C2*L(1,0)), L(0,1)*L(1,1)*(C1-C2), C1*L(0,1)*L(0,1) - (C2*(L(0,0)*L(1,1) + L(0,1)*L(1,0)))/(double(2));
        
        S = mu*I + (lambda*J*(J-1.0)-mu)*L;
        
        G << S(0,0),S(0,1),0.0,0.0,
             S(1,0),S(1,1),0.0,0.0,
             0.0,0.0,S(0,0),S(0,1),
             0.0,0.0,S(1,0),S(1,1);

        vecS << S(0,0),S(1,1),S(0,1);
        
        GKFe = Aire*BD.transpose()*K*BD + Aire*BG.transpose()*G*BG;
        FORCEe = -Aire*BD.transpose()*vecS;
        
        for (unsigned int i=0;i<64;i++)
        {
            X[e*64+i] = GKFe(i);
        }
        
        for (unsigned int j=0;j<8;j++)
        {
            Y[e*8+j] = FORCEe(j);
        } 
    }
    
    
}
        
 /*Neo-Hookean case
  *     lambda = PROP[phaseElem-1];
        mu = PROP[phaseElem+1];
  *     C1 = lambda*J*(2.0*J-1.0);
        C2 = 2.0*lambda*J*(J-1.0)-2.0*mu;
        K << L(0,0)*L(0,0)*(C1-C2), -C2*L(0,1)*L(0,1) + C1*L(0,0)*L(1,1), L(0,0)*L(0,1)*(C1-C2),
             -C2*L(1,0)*L(1,0) + C1*L(0,0)*L(1,1), L(1,1)*L(1,1)*(C1-C2), L(1,1)*(C1*L(0,1)-C2*L(1,0)),
             L(0,0)*(C1*L(0,1)-C2*L(1,0)), L(0,1)*L(1,1)*(C1-C2), C1*L(0,1)*L(0,1) - (C2*(L(0,0)*L(1,1) + L(0,1)*L(1,0)))/(double(2));
        S = mu*I + (lambda*J*(J-1.0)-mu)*L;*/
