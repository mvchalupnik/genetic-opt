#############
#Genetic optimization algorithm 
##############
import pdb

import matplotlib.pyplot as plt
import numpy as np


class GridOptimization():
    """
    Parameters
    ----------
    func: function
        function to optimize
    params: dict
        Dictionary containing parameters for genetic optimization
    params['nvar']: int
        Number of variables to optimize over
    params['span']: float
        Span to optimize variables over
    params['gridsize']: int
        Size of grid to search over

    """

    def __init__(self, params, func):
        self.params = params
        self.f = func



    def grid_optimization(self):
        """
        Numerically optimize a function with 2 variables through (inefficient) gridsearch
        Enumerate with some step size dx, and calculate the function at each point
        # return :zmax: float
            maximum function value found
        return :inds: array[float]
            indices for maximum function value found
        """
        span = self.params['span']
        gridsize = self.params['gridsize']
        nvar = self.params['nvar']

        
        ls = np.linspace(-span, span, gridsize)
        X = np.meshgrid(*(2*[ls]))
    
        z = self.f(*X)

        zmax = np.amax(z)

        linear_ind = np.argmax(z)
        ind_out = np.unravel_index(linear_ind, tuple([gridsize for i in range(0,nvar)]))
        inds = [ls[i] for i in ind_out]
        inds = np.flip(inds) #Meshgrid switches dimensions; must flip

        return zmax, inds


    def plot_2d_function(self, savepath):
        """
        Plot the function we plan to optimize
        savepath: string
            save path
        """

        fontsize_1 = 16
        fontsize_2 = 14
    
        fig, ax = plt.subplots()
            
        # ax.plot(self.df.index, self.df[feature], '-o',color='blue')

        # ax.set(xlabel='Time', ylabel = feature, title=feature+ ' vs ' + self.resample_by)
        # plt.xticks(rotation=60)
        # ax.grid(b=True)
        # ax.legend()

        span = self.params['span']
        gridsize = self.params['gridsize']

        
        x = np.linspace(-span, span, gridsize)
        y = np.linspace(-span, span, gridsize)
        
        [X, Y] = np.meshgrid(x, y)
        plt.contour(X, Y, self.f(X, Y), levels=20)

        plt.title('Contour plot of 2D function to be optimized', fontsize=fontsize_1)
        plt.ylabel('Y', fontsize=fontsize_2)
        plt.xlabel('X', fontsize=fontsize_2)
        plt.tight_layout() 
        fig.savefig(savepath + ".png")




class GeneticOptimization():
    """ 
    Parameters
    ----------
    func: function
        function to optimize
    params: dict
        Dictionary containing parameters for genetic optimization
    params['nvar']: int
        Number of variables to optimize over
    params['span']: float
        Span to optimize variables over
    params['f1']: float
        Fraction of population to randomly mutate
    params['mutate_els']: int
        number of parameter elements to mutate within a parameter set
    params['f2']: float
        Fraction of population to randomly combine
    params['f3']: float
        Keep the top f3 fraction for the next iteration
    params['epochs']: int
        Number of generations to cycle through
    params['popsize']: int
        Size of the population
    """

    def __init__(self, func, params):
        self.params = params
        self.func = func









# function [fit_hist, survivor, survivor_fitness] = genetic_alg(nvar, span, hp)
#     %Optimize via a genetic algorithm, using mutation and combination
#     %to create new members in the population, and keeping only the 
#     %"fittest" individuals
#     %:nvar: number of variables to optimize the function over
#     %:span: bound the optimization space of each variable over span
#     %:hp: struct containing hyperparameters
#     %return :fit_hist: array containing the fitness history over each epoch
#     %return :survivor: the fittest individual
#     %return :survivor_fitness: the fitness of the fittest individual
    
#     f1 = hp.f1; %Fraction to randomly mutate
#     mutate_els = hp.mutate_els; %number of parameter elements to mutate within a parameter set
#     f2 = hp.f2; %Fraction to randomly combine
#     f3 = hp.f3; %Keep the top f3 fraction for the next iteration
#     epochs = hp.epochs; %Number of generations to cycle through
#     popsize = hp.popsize; %Size of the population

    
#     %% Generate random population, each number bounded by span
#     pop = (rand([popsize,nvar]) - 0.5) * 2*span;
    
#     %Store fitness history here
#     fit_hist = zeros(epochs, popsize);
    
#     for j = 1:epochs
        
#         disp(strcat('Running epoch ', num2str(j)));
       
#         %% If len(pop) < popsize, randomly create new individuals
#         if size(pop, 1) < popsize
#            new_pop_members = (rand([popsize - size(pop, 1),nvar]) - 0.5) * 2*span;
#            pop = [pop; new_pop_members];
#         end

#         %% Mutate random f1 proportion
#         mutate_indices = randperm(popsize); %Use random permuation to avoid repeating indices
#         mutate_indices = mutate_indices(1:ceil(popsize*f1));

#         %For each element in pop, randomly mutate _mutate_els_ variables
#         for i = 1:length(mutate_indices)
#             %Randomly select indices to mutate
#             mutate_el_indices = randperm(nvar);
#             mutate_el_indices = mutate_el_indices(1:mutate_els);
            
#             %Mutate that index of individual mutate_indices(i) of the population
#             pop(mutate_indices(i),mutate_el_indices) = (rand(1) - 0.5) * 2*span;
#         end

#         %% Genetically combine random f2; produce 2 offspring per 2 parents
#         % to avoid population loss
#         combine_indices = randperm(popsize); %Use random permuation to avoid repeating indices
#         num_parents = ceil((popsize*f2)/2)*2; %make sure result is even
#         combine_indices = combine_indices(1:num_parents);

#         %For each 2 elements in pop (the parents), randomly combine
#         %variables
#         for i = 1:2:length(num_parents)
#             %Randomly select indices to mutate
#             combine_el_indices = randperm(nvar);
#             firsthalf_indices = combine_el_indices(1:ceil(nvar/2));
#             secondhalf_indices = combine_el_indices(ceil(nvar/2)+1:nvar);
            
#             parent1 = pop(combine_indices(i), :);
#             parent2 = pop(combine_indices(i+1), :);
            
#             %Combine parent1 and parent2 of the population to produce 2
#             %offspring
#             pop(combine_indices(i),firsthalf_indices) = parent2(firsthalf_indices);
#             pop(combine_indices(i+1),firsthalf_indices) = parent1(firsthalf_indices);
#         end
        
        
#         %% Fitness test on random population
        
#         %fitness = myfunc(pop(:,1), pop(:,2));
#         pop_in = num2cell(pop, 1);
#         fitness = myfunc(pop_in{:});
        
#         arr = [pop, fitness]; 
#         disp('Initial parameters (first _nvar_ columns) and fitness values (last col): ')
#         disp(num2str(arr))
        
#         %Save fitness history
#         fit_hist(j, :) = squeeze(arr(:, nvar+1));

#         %% Keep top f3 in fitness
#         arr = sortrows(arr, nvar+1);
#         arr = arr(ceil(size(arr, 1)*(1-f3)):size(arr,1), :); %keep top f3 in fitness
        
#         %% Strip off fitness values and repeat
#         pop = arr(:, 1:nvar);
        
#     end
#     survivor = pop(size(arr,1), :);
#     survivor_fitness = arr(size(arr,1), nvar+1);
# end

# function [] = scatterplot_fitness(hist, savepath, fmax)
#     %Scatter plot the fitness of each individual at each epoch
#     %:hist: a 2D array of fitnesses with each epoch
#     %:savepath: the path to save the figure
#     %:fmax: function maximum determined by gridsearch
    
#     fontsize_1 = 20;
#     fontsize_2 = 16;
#     fontsize_3 = 14;
    
#     hdl = figure;
#     hold on;
#     plot(1:size(hist, 1), mean(hist, 2), '--k')
#     plot(1:size(hist, 1), fmax*ones(1,size(hist, 1)), '--r')
#     legend('Population Mean', 'Grid Search Max', 'AutoUpdate', 'off')
    
#     scatter(1:size(hist, 1), hist, 40, 'MarkerEdgeColor',[0 .5 .5],...
#               'MarkerFaceColor',[0 .7 .7],...
#               'LineWidth',1.5)
             
#     ax = gca;

#     ax.XLim = [0,size(hist,1)+1];

#     title('Fitnesses during genetic optimization', 'FontSize', fontsize_1);
#     ylabel('Fitness', 'FontSize', fontsize_2)
#     xlabel('Epoch', 'FontSize', fontsize_2)
#     ax.FontSize = fontsize_3;
    
#     hold off; 
    
#     saveas(hdl, [savepath, '.png']);
#     savefig(hdl, [savepath, '.fig']);
#     close(hdl);
    
# end

