# BDD_VariableOrdering_NodeSelection-MISP-


The associated code is an implementation of the paper: 
"Strengthening Relaxed Decision Diagrams for Maximum Independent Set Problem: Novel Variable Ordering and Merge Heuristics" published at CP 2024.

This code is designed to compile BDDs for Maximum Independent Set Problems, i.e. relaxed BDD, restricted BDD, and DD-based Branch-and-Bound.

File "BDD_VO_NS(CDS_MIN-BT-SO.jl" is a Pluto Notebook containing 3 cells; 

first cell, which is a very highly generic written code, is the related code 
that implements all the needed functions and structures,

second cell is the running code for solving the problem via DD-based Branch-and-Bound,

third cell is the running code for building relaxed BDDs for MISP,

To install Julia please visit the following link:
https://julialang.org/ .

For an introduction to Pluto Notebook and start using it visit the following link:
https://plutojl.org/ .

Notes about the instances:

There is a folder containing the instances for problems, i.e. instance_100;

it contains 10 folders for different graph densities, each containing 20 instances and text file including the optimum value. 
