% Genetic optimization algorithm

savedir = '~/Desktop/GeneticTest/';
saveloc = strcat(savedir, '2021_11_05_test1');

if ~exist(saveloc, 'dir')
   mkdir(saveloc) 
end

%seed rand

[zmax, inds] = analytical_optimization(@(x,y)myfunc, 2, 3, 1);
disp(zmax)
disp(inds)

genetic_alg(@(x,y)myfunc, 2, 1)

function f = myfunc(x1, x2)
    %f =  x^3 + 5*y^2 + 8*Sin(z);
    f = 3*(1-x1).^2.*exp(-(x1.^2) - (x2+1).^2) ... 
   - 10*(x1/5 - x1.^3 - x2.^5).*exp(-x1.^2-x2.^2) ... 
   - 1/3*exp(-(x1+1).^2 - x2.^2); % + (z-1)^2 ; %"peaks" function
end

function [zmax, inds] = analytical_optimization(func, nvar, gridsize, span)
    %Numerically optimize some differentiable multivariable function 
    %through (inefficient) gridsearch
    %Enumerate with some step size dx, and calculate the function 
    %at each point
    %:func: function to optimize
    %:nvar: number of variables in function to optimize
    %:gridsize: size of grid to optimize over
    %:span: span to optimize grid over (from -span to span, for each
    %variable)
    
    %Create grid to search over
    
    X1 = linspace(-span, span, gridsize);
    X2 = linspace(-span, span, gridsize);
    
    [x1,x2] = ndgrid(X1, X2);
    
    z = myfunc(x1, x2);
    
    [zmax, linear_ind] = max(z, [], 'all', 'linear');
    [i1, i2] = ind2sub([gridsize, gridsize], linear_ind);
    inds = [i1, i2];
    
end


function [] = genetic_alg(func, nvar, span)
    %% Set up hyperparameters
    f1 = .20; %Fraction to randomly mutate
    f2 = .30; %Fraction to randomly combine
    f3 = .10; %Keep the top f3 fraction for the next iteration
    
    popsize = 50; 
    
    %% Generate random population
    pop = (rand([popsize,nvar]) - 0.5) * span;
    
    %% Fitness test on random population
    fitness = myfunc(pop(:,1), pop(:,2));
    
    
    %% Mutate random f1
    
    %% Genetically combine random f2
    
    %% Keep top f3 in fitness
    
    %% Repeat
end