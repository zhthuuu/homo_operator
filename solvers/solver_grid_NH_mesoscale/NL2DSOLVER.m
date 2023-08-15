function [WEFF, DISPTD_FINAL] = NL2DSOLVER(FIXEDNODES, p, BULK, SHEAR, NE, GDOF, ND, g, s, Ae, DOFe, ISPARSE, JSPARSE, KSPARSE)
                          
%% INITIALIZATION DATA 

global DISPDD DISPTD GKF FORCE

TIMS = [0, 1, 1]';

TMIN = TIMS(1, 1);
TMAX = TIMS(2, 1);
DELTA = TIMS(3, 1);

TIME = TMIN;
DELTA0 = DELTA;

%prescribed displacements [nodes, dof, value]

PDISP = [ [FIXEDNODES; FIXEDNODES], ...
           [ones(ND, 1); 2*ones(ND, 1)], ...
           [g(1, 1)*p(FIXEDNODES, 1)+g(1, 2)*p(FIXEDNODES, 2); g(2, 1)*p(FIXEDNODES, 1) + g(2, 2)*p(FIXEDNODES, 2)] ];
       
GND = size(PDISP, 1);       

%% NONLINEAR SOLVER

EPS = 1E-8;
ITRA = 40;
ITOL = 1;
ATOL = 1E5;
TOL = 1E-7;
NTOL = 20;

DISPTD = zeros(GDOF, 1);
DISPDD = zeros(GDOF, 1);

FLAG10 = 1;
while (FLAG10 == 1)
    
    FLAG10 = 0;
    FLAG11 = 1;
    FLAG20 = 1;
    
    CDISP = DISPTD;
    
    if (ITOL==1) 
        DELTA = DELTA0;
        TARY(ITOL) = TIME + DELTA;
    else
        ITOL = ITOL-1;
        DELTA = TARY(ITOL)-TARY(ITOL+1);
        TARY(ITOL+1) = 0;
    end
    
    TIME0 = TIME;
    TIME = TIME + DELTA;
    
    while (FLAG11 == 1)
        FLAG11 = 0;
        if (TIME-TMAX)>EPS
           if (TMAX + DELTA - TIME)>EPS
               DELTA = TMAX + DELTA - TIME;
               DELTA0 = DELTA;
               TIME = TMAX;
           else 
               FLAG10 = 0; 
               break; 
           end
        end
   
    SDISP = DELTA*PDISP(:, 3);
   
    ITER = 0;
    DISPDD = zeros(GDOF, 1);
    
        while (FLAG20 == 1)
            FLAG20 = 0;
            ITER = ITER + 1;
            if(ITER>ITRA), error('Iteration limit exceeds'); end  

            [X, Y] = mex_HYPER2D(BULK, SHEAR, NE, s, Ae, GDOF, DOFe, DISPTD);
            GKF = sparse(ISPARSE, JSPARSE, X, GDOF, GDOF);
            FORCE = sparse(KSPARSE, ones(8*NE, 1), Y, GDOF, 1);
            
            KUBC(PDISP, SDISP, GDOF, ITER, GND);
                        
            if (ITER>1)
                RESN = RESIDUAL(GDOF, PDISP);
%                     if ITER>2
%                       fprintf(1,'   %27d %14.5e %14.e \n',ITER,full(RESN),full(NSTEP));
%                     else
%                       fprintf(1,'\n \tTime   Time step   Iter   Residual     Norm increment \n');
%                       fprintf(1,'   %10.5f %10.3e %5d %14.5e %14.5e \n',TIME,DELTA,ITER,full(RESN), full(NSTEP));
%                     end
                if (RESN<TOL)
                    FLAG10 = 1;
                    break;
                end

                if ((RESN>ATOL)||(ITER>=ITRA))
                    ITOL = ITOL + 1;
                    if (ITOL<NTOL) 
                        DELTA = 0.5*DELTA;
                        TIME = TIME0 + DELTA;
                        TARY(ITOL) = TIME;
                        DISPTD=CDISP; 
                        %fprintf('Not converged. Bisecting load increment %3d\n', ITOL);
                    else
                        error('Max No. of bisection'); 
                    end
                FLAG11 = 1; 
                FLAG20 = 1; 
                break;
                end
                
            end
                if(FLAG11 == 0) 
                    
                    %spparms('spumoni',2)
                    SOLN = GKF\FORCE;
                    %dlmwrite('SOLN.txt', full(SOLN), 'precision', '%.20f');
                    %pause
                    DISPDD = DISPDD + SOLN;
                    DISPTD = DISPTD + SOLN;
                    
                    NSTEP = norm(SOLN);
                    FLAG20 = 1; 
                else
                    FLAG20 = 0; 
                end    
        if(FLAG10 == 1), break; end 
        end
    end 
end 

WEFF = mex_EFFSEF(DISPTD, NE, BULK, SHEAR, s, Ae, DOFe);
DISPTD_FINAL = DISPTD;

end

function RESN = RESIDUAL(GDOF, SDISPT)

  global FORCE

  FIXEDDOF = 2*(SDISPT(:, 1) - 1) + SDISPT(:, 2);
  
  ACTIVEDOF = setdiff(1:GDOF, FIXEDDOF);
  
  RESN = max(abs(FORCE(ACTIVEDOF)));
  
end
    
function KUBC(SDISPT, SDISP, GDOF, ITER, GND)

global FORCE GKF

FIXEDDOF = 2*(SDISPT(:, 1)-1) + SDISPT(:, 2);
GKF(FIXEDDOF, :) = zeros(GND, GDOF);
GKF(FIXEDDOF, FIXEDDOF) = eye(GND);

FORCE(FIXEDDOF, 1) = 0;

if ITER==1, FORCE(FIXEDDOF) = SDISP; end

end



