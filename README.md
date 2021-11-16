# genetic-opt
## Genetic optimization implemented in Matlab and Python
### Introduction
Optimization of functions with many variables can be time and resource consuming, particularly if the function 
landscape contains many local minima, or when analytical optimization is not possible. Genetic 
optimizations introduce randomness that can help avoid local minima, while optimizing
function through a "survival of the fittest" strategy. This
repo implements a simple genetic optimization, with the same code duplicated for both Matlab and python. 

In a genetic optimization, 
a population of function parameters are randomly generated. Some function parameters "mutate" 
(through random replacement of some of their parameters) and some combine with other function
parameters to produce "offspring" parameter sets. At the end of an epoch, the function 
parameters are evaluated on a fitness function. Only a top-scoring fraction of function parameters "survive", 
the rest are discarded.  Unlike stochastic gradient descent, 
the evaluation of fitness functions in genetic optimization can be run in parallel 
(though not implemented in this repo), which can allow for an increase in optimization speed.


The following flowchart, taken from JOURNAL OF LIGHTWAVE TECHNOLOGY, VOL. 16, NO. 10, OCTOBER 1998
"A Genetic Algorithm for the Inverse Problem in Synthesis of Fiber Gratings"
by Johannes Skaar and Knut Magne Risvik, shows the general process of a genetic algorithm. 


<img src="imgs/flowchart.png" width = "600">

The same paper gives illuminating schematics for the crossover and mutation operations: 

<img src="imgs/crossover.png" width = "600">

<img src="imgs/mutation.png" width = "600">

### Code description
This code gives a simple implementation of a genetic optimization. Solely for 
ease of visualization and ease of comparison to another optimization technique (grid search),
 the example implemented is only a 2-D function; however, 
the genetic optimization will work best for functions with many more than 
two input variables. 

The contour plot below shows the function we want to optimize. 

<img src="imgs/surf_plot.png" width = "600">

Since there are only two variables, X and Y, we can pretty easily optimize this function 
using a grid search. The optimal values within the span of [-1, 1] return as 
x = -0.4582, y = -0.6254. 

Next, we run the genetic algorithm. We introduce 4 additional dummy variables
to make full use of the mutation and combination. We run with the following
hyperparameters: 

    f1 = .20; %Fraction to randomly mutate
    mutate_els = 1; %number of parameter elements to mutate within a parameter set
    f2 = .30; %Fraction to randomly combine
    f3 = .40; %Keep the top f3 fraction for the next iteration
    epochs = 20; %Number of generations to cycle through
    popsize = 20; %Size of the population

One optimization result is plotted below:

<img src="imgs/fitness_scatterplot_test1.png" width = "600">

We can see the average population fitness mean does increase over epochs, 
though there is randomness and the increase is not monotonic. 

Here is another optimization result example: 

<img src="imgs/fitness_scatterplot_test2.png" width = "600">

Here we can see the population fitness mean increases significantly until some significant number of
unadvantagous mutations or offspring occur around epoch 9. The stochastic nature
of genetic optimization means sometimes the fittest survivor will die off or
produce offspring which are less fit, lowering the population fitness mean.

Note that in this example we are only optimizing over
two variables, and have four additional dummy variables which do not affect 
the fitness function; to use genetic optimization at its full potential, it will 
make most sense to use it for functions of number of variables > 2. 

The same code is provided both in Matlab (genetic_opt.m) and python 
(genetic_opt.py, example.py). 
