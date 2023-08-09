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
    
    double *bulk, *shear, *sb, *sc, *Ae, *DISPTD;
    int32_T *DOFe;
    int NE, GDOF;
    
    /* Check for proper number of arguments */
    if (nrhs != 9) {
        mexErrMsgIdAndTxt("MATLAB:mexcpp:nargin",
                          "MEXCPP requires nine input arguments.");
    } else if (nlhs != 2) {
        mexErrMsgIdAndTxt("MATLAB:mexcpp:nargout",
                          "MEXCPP requires no output argument.");
    }
    /* ------------------------------------- */
    
    NE = (int) mxGetScalar(prhs[0]);
    bulk = (double *) mxGetPr(prhs[1]);
    shear = (double *) mxGetPr(prhs[2]);
    sb = (double *) mxGetPr(prhs[3]);
    sc = (double *) mxGetPr(prhs[4]);
    Ae = (double *) mxGetPr(prhs[5]);
    GDOF = (int) mxGetScalar(prhs[6]);
    DOFe = (int32_T *) mxGetData(prhs[7]);
    DISPTD = (double *) mxGetPr(prhs[8]);
    
    double *X, *Y;
    double J, C1, C2, Aire, lambda, mu;
    Eigen::Matrix<double,2,2> I, F, C, L, S;
    Eigen::Matrix<double,2,3> U;
    Eigen::Matrix<double,3,2> D;
    Eigen::Matrix<double,6,1> uE;
    Eigen::Matrix<double,3,6> BD;
    Eigen::Matrix<double,4,6> BG;
    Eigen::Matrix<double,3,3> K;
    Eigen::Matrix<double,4,4> G;
    Eigen::Matrix<double,6,6> GKFe;
    Eigen::Matrix<double,6,1> FORCEe;
    Eigen::Matrix<double,3,1> vecS;
    
    plhs[0] = mxCreateDoubleMatrix(36*NE,1,mxREAL);
    plhs[1] = mxCreateDoubleMatrix(6*NE,1,mxREAL);
    
    X = mxGetPr(plhs[0]);
    Y = mxGetPr(plhs[1]);

    I << 1.0, 0.0,
         0.0, 1.0;
    
    for (int e=0;e<NE;e++)
    {
        Aire = Ae[e];
        lambda = bulk[e];
        mu = shear[e];
        
        for (unsigned int j=0;j<6;j++)
        {
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
        L = C.inverse();
        
        BD << F(0,0)*t2D(sb,0,e,3), F(1,0)*t2D(sb,0,e,3), F(0,0)*t2D(sb,1,e,3), F(1,0)*t2D(sb,1,e,3), F(0,0)*t2D(sb,2,e,3), F(1,0)*t2D(sb,2,e,3),
              F(0,1)*t2D(sc,0,e,3), F(1,1)*t2D(sc,0,e,3), F(0,1)*t2D(sc,1,e,3), F(1,1)*t2D(sc,1,e,3), F(0,1)*t2D(sc,2,e,3), F(1,1)*t2D(sc,2,e,3),
              F(0,0)*t2D(sc,0,e,3)+F(0,1)*t2D(sb,0,e,3), F(1,0)*t2D(sc,0,e,3)+F(1,1)*t2D(sb,0,e,3), F(0,0)*t2D(sc,1,e,3)+F(0,1)*t2D(sb,1,e,3), F(1,0)*t2D(sc,1,e,3)+F(1,1)*t2D(sb,1,e,3), F(0,0)*t2D(sc,2,e,3)+F(0,1)*t2D(sb,2,e,3), F(1,0)*t2D(sc,2,e,3)+F(1,1)*t2D(sb,2,e,3);
                                                                                                    
        BG << t2D(sb,0,e,3), 0.0, t2D(sb,1,e,3), 0.0, t2D(sb,2,e,3), 0.0,
              t2D(sc,0,e,3), 0.0, t2D(sc,1,e,3), 0.0, t2D(sc,2,e,3), 0.0,
              0.0, t2D(sb,0,e,3), 0.0, t2D(sb,1,e,3), 0.0, t2D(sb,2,e,3),
              0.0, t2D(sc,0,e,3), 0.0, t2D(sc,1,e,3), 0.0, t2D(sc,2,e,3);
        
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
        
        for (unsigned int i=0;i<36;i++)
        {
            X[e*36+i] = GKFe(i);
        }
        
        for (unsigned int j=0;j<6;j++)
        {
            Y[e*6+j] = FORCEe(j);
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

/*Mooney-Rivlin case
    phaseElem = PHASES[e];
    
    alpha1 = PROP[phaseElem-1];
    beta1 = PROP[phaseElem+1];
    s1 = PROP[phaseElem+3];
    s2 = 2*(alpha1+2*beta1);
            C1 = s1*J*(2*J - 1); 
    C2 = 2*s1*J*(J-1)-2*s2;
    
    K << L(0,0)*L(0,0)*(C1-C2), - C2*L(0,1)*L(0,1) + 4*beta1 + C1*L(0,0)*L(1,1), L(0,0)*L(0,1)*(C1-C2),
         -C2*L(1,0)*L(1,0) + 4*beta1 + C1*L(0,0)*L(1,1), L(1,1)*L(1,1)*(C1-C2), L(1,1)*(C1*L(0,1) - C2*L(1,0)),
        L(0,0)*(C1*L(0,1)-C2*L(1,0)), L(0,1)*L(1,1)*(C1-C2), C1*L(0,1)*L(0,1) - 4*beta1 - (C2*(L(0,0)*L(1,1) + L(0,1)*L(1,0)))/double(2);
    
    S = 2*(alpha1 + beta1*(C.trace()+1))*I 
            - 2*beta1*C + (s1*J*(J-1)-s2)*L;
*/          