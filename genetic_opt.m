% Genetic optimization algorithm
clear;

%Create directories for saving files
savedir = '~/Desktop/GeneticTest/';
saveloc = strcat(savedir, '2021_11_15_test2/');

if ~exist(saveloc, 'dir')
   mkdir(saveloc) 
end

%seed rng for reproducibility
rng(1);

%Plot the function we will optimize
gridsize = 300;
span = 1;
plot_2d_function(span,gridsize, strcat(saveloc, 'surf_plot'));

%Run grid optimization 
nvar = 2; 
[zmax, inds] = grid_optimization(nvar, gridsize, span);

disp(strcat('Grid search optimization maximum: ', num2str(zmax)))
disp('Grid search optimization maximum indices: ')
disp(inds)

%Run genetic optimization
nvar = 6; %add extra variables as noise here
span = 1;
% Set up hyperparameters
hp.f1 = .20; %Fraction to randomly mutate
hp.mutate_els = 1; %number of parameter elements to mutate within a parameter set
hp.f2 = .30; %Fraction to randomly combine
hp.f3 = .40; %Keep the top f3 fraction for the next iteration
hp.epochs = 20; %Number of generations to cycle through
hp.popsize = 20; %Size of the population

[fh, sv, svf] = genetic_alg(nvar, span, hp);

%Plot fitness history
scatterplot_fitness(fh, strcat(saveloc, 'fitness_scatterplot_test'), zmax);

%Save hyperparameters to a .mat file
save(strcat(saveloc, 'optimization_parameters.mat'), 'hp', 'fh', 'sv', 'svf')



function f = myfunc(x1, x2, varargin)
    f = 3*(1-x1).^2.*exp(-(x1.^2) - (x2+1).^2) ... 
   - 10*(x1/5 - x1.^3 - x2.^5).*exp(-x1.^2-x2.^2) ... 
   - 1/3*exp(-(x1+1).^2 - x2.^2); %Matlab's "peaks" function
end

function [zmax, inds] = grid_optimization(nvar, gridsize, span)
    %Numerically optimize a function with 2 variables
    %through (inefficient) gridsearch
    %Enumerate with some step size dx, and calculate the function 
    %at each point
    %:nvar: number of variables in function to optimize
    %:gridsize: size of grid to optimize over
    %:span: span to optimize grid over (from -span to span, for each
    %variable)
    %return :zmax: maximum function value found
    %return :inds: indices for maximum function value found
    
    %Create grid to search over
    % x to NDGRID specified at runtime by user as elements of a cell array
    x = repmat({linspace(-span, span, gridsize)}, 1, nvar);
    % X from NDGRID specified at runtime by user as elements of a cell array
    X = cell(1, nvar);        
    
    [X{:}] = ndgrid(x{:});
        
    z = myfunc(X{:});
    
    [zmax, linear_ind] = max(z, [], 'all', 'linear');
    
    ind_out = cell(1,nvar);
    [ind_out{:}] = ind2sub(ones(1,nvar)*gridsize, linear_ind);
    inds = arrayfun(@(i) x{i}(ind_out{i}), 1:nvar);
    
end


function [fit_hist, survivor, survivor_fitness] = genetic_alg(nvar, span, hp)
    %Optimize via a genetic algorithm, using mutation and combination
    %to create new members in the population, and keeping only the 
    %"fittest" individuals
    %:nvar: number of variables to optimize the function over
    %:span: bound the optimization space of each variable over span
    %:hp: struct containing hyperparameters
    %return :fit_hist: array containing the fitness history over each epoch
    %return :survivor: the fittest individual
    %return :survivor_fitness: the fitness of the fittest individual
    
    f1 = hp.f1; %Fraction to randomly mutate
    mutate_els = hp.mutate_els; %number of parameter elements to mutate within a parameter set
    f2 = hp.f2; %Fraction to randomly combine
    f3 = hp.f3; %Keep the top f3 fraction for the next iteration
    epochs = hp.epochs; %Number of generations to cycle through
    popsize = hp.popsize; %Size of the population

    
    %% Generate random population, each number bounded by span
    pop = (rand([popsize,nvar]) - 0.5) * 2*span;
    
    %Store fitness history here
    fit_hist = zeros(epochs, popsize);
    
    for j = 1:epochs
        
        disp(strcat('Running epoch ', num2str(j)));
       
        %% If len(pop) < popsize, randomly create new individuals
        if size(pop, 1) < popsize
           new_pop_members = (rand([popsize - size(pop, 1),nvar]) - 0.5) * 2*span;
           pop = [pop; new_pop_members];
        end

        %% Mutate random f1 proportion
        mutate_indices = randperm(popsize); %Use random permuation to avoid repeating indices
        mutate_indices = mutate_indices(1:ceil(popsize*f1));

        %For each element in pop, randomly mutate _mutate_els_ variables
        for i = 1:length(mutate_indices)
            %Randomly select indices to mutate
            mutate_el_indices = randperm(nvar);
            mutate_el_indices = mutate_el_indices(1:mutate_els);
            
            %Mutate that index of individual mutate_indices(i) of the population
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
            %Randomly select indices to mutate
            combine_el_indices = randperm(nvar);
            firsthalf_indices = combine_el_indices(1:ceil(nvar/2));
            secondhalf_indices = combine_el_indices(ceil(nvar/2)+1:nvar);
            
            parent1 = pop(combine_indices(i), :);
            parent2 = pop(combine_indices(i+1), :);
            
            %Combine parent1 and parent2 of the population to produce 2
            %offspring
            pop(combine_indices(i),firsthalf_indices) = parent2(firsthalf_indices);
            pop(combine_indices(i+1),firsthalf_indices) = parent1(firsthalf_indices);
        end
        
        
        %% Fitness test on random population
        
        %fitness = myfunc(pop(:,1), pop(:,2));
        pop_in = num2cell(pop, 1);
        fitness = myfunc(pop_in{:});
        
        arr = [pop, fitness]; 
        disp('Initial parameters (first _nvar_ columns) and fitness values (last col): ')
        disp(num2str(arr))
        
        %Save fitness history
        fit_hist(j, :) = squeeze(arr(:, nvar+1));

        %% Keep top f3 in fitness
        arr = sortrows(arr, nvar+1);
        arr = arr(ceil(size(arr, 1)*(1-f3)):size(arr,1), :); %keep top f3 in fitness
        
        %% Strip off fitness values and repeat
        pop = arr(:, 1:nvar);
        
    end
    survivor = pop(size(arr,1), :);
    survivor_fitness = arr(size(arr,1), nvar+1);
end

function [] = scatterplot_fitness(hist, savepath, fmax)
    %Scatter plot the fitness of each individual at each epoch
    %:hist: a 2D array of fitnesses with each epoch
    %:savepath: the path to save the figure
    %:fmax: function maximum determined by gridsearch
    
    fontsize_1 = 20;
    fontsize_2 = 16;
    fontsize_3 = 14;
    
    hdl = figure;
    hold on;
    plot(1:size(hist, 1), mean(hist, 2), '--k')
    plot(1:size(hist, 1), fmax*ones(1,size(hist, 1)), '--r')
    legend('Population Mean', 'Grid Search Max', 'AutoUpdate', 'off')
    
    scatter(1:size(hist, 1), hist, 40, 'MarkerEdgeColor',[0 .5 .5],...
              'MarkerFaceColor',[0 .7 .7],...
              'LineWidth',1.5)
             
    ax = gca;

    ax.XLim = [0,size(hist,1)+1];

    title('Fitnesses during genetic optimization', 'FontSize', fontsize_1);
    ylabel('Fitness', 'FontSize', fontsize_2)
    xlabel('Epoch', 'FontSize', fontsize_2)
    ax.FontSize = fontsize_3;
    
    hold off; 
    
    saveas(hdl, [savepath, '.png']);
    savefig(hdl, [savepath, '.fig']);
    close(hdl);
    
end

function [] = plot_2d_function(span, gridsize, savepath)
    %Plot the function we plan to optimize
    %:span: span to plot over
    %:gridsize: gridsize to use
    %:savepath: save path
    
    fontsize_1 = 20;
    fontsize_2 = 16;
    
    hdl = figure;
    hold on;
    
    x = linspace(-span, span, gridsize);
    y = linspace(-span, span, gridsize);
    
    [X, Y] = meshgrid(x, y);
    contour(X, Y, myfunc(X, Y), 20);
    
             
    ax = gca;


     title('Contour plot of 2D function to be optimized', 'FontSize', fontsize_1);
     ylabel('Y', 'FontSize', fontsize_2)
     xlabel('X', 'FontSize', fontsize_2)

    hold off; 
    
    saveas(hdl, [savepath, '.png']);
    savefig(hdl, [savepath, '.fig']);
    close(hdl);
end