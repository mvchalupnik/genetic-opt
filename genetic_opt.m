% Genetic optimization algorithm

savedir = '~/Desktop/GeneticTest/';
saveloc = strcat(savedir, '2021_11_05_test1');

if ~exist(saveloc, 'dir')
   mkdir(saveloc) 
end

%seed rand

[zmax, inds] = analytical_optimization( 2, 3, 1);
disp(zmax)
disp(inds)

genetic_alg(6, 1)

function f = myfunc(x1, x2)
    %f =  x^3 + 5*y^2 + 8*Sin(z);
    f = 3*(1-x1).^2.*exp(-(x1.^2) - (x2+1).^2) ... 
   - 10*(x1/5 - x1.^3 - x2.^5).*exp(-x1.^2-x2.^2) ... 
   - 1/3*exp(-(x1+1).^2 - x2.^2); % + (z-1)^2 ; %"peaks" function
end

function [zmax, inds] = analytical_optimization(nvar, gridsize, span)
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


function [] = genetic_alg(nvar, span)
    %% Set up hyperparameters
    f1 = .20; %Fraction to randomly mutate
    f2 = .30; %Fraction to randomly combine
    f3 = .40; %Keep the top f3 fraction for the next iteration
    epochs = 1; %Number of generations to cycle through
    
    popsize = 3; 
    
    %% Generate random population, each number bounded by span
    pop = (rand([popsize,nvar]) - 0.5) * 2*span;
    
    for j = 1:epochs
        disp(strcat('Running epoch ', num2str(j)));
       
        %% If len(pop) < popsize, randomly create new individuals
        %if length(pop) 

        %% Mutate random f1 proportion
        mutate_indices = randperm(popsize); %Use random permuation to avoid repeating indices
        mutate_indices = mutate_indices(1:ceil(popsize*f1));

        %For each element in pop, randomly mutate _mutate_els_ variables
        mutate_els = 1; %Can modify this
        for i = 1:length(mutate_indices)
            %Randomly select indices to mutate
            mutate_el_indices = randperm(nvar);
            mutate_el_indices = mutate_el_indices(1:mutate_els);
            
            %Mutate that index of individual i of the population
            pop(mutate_indices(i),mutate_el_indices) = (rand(1) - 0.5) * 2*span;
        end

        %% Genetically combine random f2; produce 2 offspring per 2 parents
        % to avoid population loss
        combine_indices = randperm(popsize); %Use random permuation to avoid repeating indices
        num_parents = ceil((popsize*f2)/2)*2; %make sure result is even
        combine_indices = combine_indices(1:num_parents);

        %For each 2 elements in pop (the parents), randomly combine
        %variables
        for i = 1:2:length(num_parents)
            disp(pop)
            %Randomly select indices to mutate
            combine_el_indices = randperm(nvar);
            firsthalf_indices = combine_el_indices(1:ceil(nvar/2));
            secondhalf_indices = combine_el_indices(ceil(nvar/2)+1:nvar);
            
            parent1 = pop(combine_indices(i), :);
            disp(parent1)
            parent2 = pop(combine_indices(i+1), :);
            disp(parent2)
            
            %Mutate that index of individual i of the population
            pop(combine_indices(i),firsthalf_indices) = parent2(firsthalf_indices);
            pop(combine_indices(i+1),firsthalf_indices) = parent1(firsthalf_indices);
            disp(pop)
        end
        
        
        %% Fitness test on random population
        fitness = myfunc(pop(:,1), pop(:,2));
        arr = [pop, fitness]; 
        disp('Initial parameters (first two columns) and fitness values (third col): ')
        disp(num2str(arr))

        %% Keep top f3 in fitness
        arr = sortrows(arr, nvar+1);
        arr = arr(ceil(size(arr, 1)*(1-f3)):size(arr,1), :); %keep top f3 in fitness
        disp(arr)
        %% Repeat
    end
end