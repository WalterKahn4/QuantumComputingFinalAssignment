import numpy as np
import sys
import time

from Utils.utils import *
from Utils.ClassicalApproach import *
from Utils.QuantumM1Approach import *
from Utils.QuantumM2Approach import *

Project = "189W63-ZOU"
# Project = "111"
# Project = "134"
# Project = "380=61"
# Project = "380=35"

num_constraints = 5 # Hyperparameter

np.set_printoptions(threshold=4,suppress=True)
project_splitser(Project)

with open(f"Projects\{Project}.csv", "r") as file:
    
    # read project.csv into Measurement class
    measurements = read_measurements(file)
    
    ## initialize models
    ClassicModel       = ClassicalMethod(measurements)
    QuantumModel1 = QuantumMeasurementsModel1(measurements)
    QuantumModel2 = QuantumMeasurementsModel2(measurements)
    
    ## calculation of classical model 
    
    x_i                   = ClassicModel.create_x_i()
    norm_var, var         = ClassicModel.create_var()
    u_i, last_constraint  = ClassicModel.create_u_i(num_constraints)
    u_o                   = ClassicModel.create_u_o(num_constraints,last_constraint)
    y_e                   = ClassicModel.create_y_e(x_i,u_i,u_o)
    g_et                  = ClassicModel.create_g_et(u_i,norm_var)
    g_et_inv, succes      = ClassicModel.create_inverse_g_et(g_et)
    if succes:
        e_i                   = ClassicModel.create_e_i(norm_var,u_i,g_et_inv,y_e)
        updated_x_i           = ClassicModel.updated_x_i(x_i,e_i)

        ClassicModel.calculate_f_test(x_i, updated_x_i)
    else:
        print("Unable to compute the inverse due to g_et being singular")
    
    ## calculation of QUBO model 1
    
    x_i                   = QuantumModel1.create_x_i()
    norm_var, var         = QuantumModel1.create_var()
    u_i, last_constraint  = QuantumModel1.create_u_i(num_constraints)
    u_o                   = QuantumModel1.create_u_o(num_constraints,last_constraint)
    y_e                   = QuantumModel1.create_y_e(x_i,u_i,u_o)
    g_et                  = QuantumModel1.create_g_et(u_i,norm_var)
    quantum_g_et_inv      = QuantumModel1.create_inverse_g_et(g_et=g_et,dimension=num_constraints,num_constraints=num_constraints)
    e_i                   = QuantumModel1.create_e_i(norm_var,u_i,quantum_g_et_inv,y_e)
    updated_x_i           = QuantumModel1.updated_x_i(x_i,e_i)

    QuantumModel1.calculate_f_test(x_i, updated_x_i)
    
    ## calculation of QUBO model 2
    
    x_i                   = QuantumModel2.create_x_i()
    norm_var, var         = QuantumModel2.create_var()
    u_i, last_constraint  = QuantumModel2.create_u_i(num_constraints)
    u_o                   = QuantumModel2.create_u_o(num_constraints,last_constraint)
    y_e                   = QuantumModel2.create_y_e(x_i,u_i,u_o)
    g_et                  = QuantumModel2.create_g_et(u_i,norm_var)
    quantum_g_et_inv      = QuantumModel2.create_inverse_g_et(g_et=g_et,dimension=num_constraints,num_constraints=num_constraints)
    e_i                   = QuantumModel2.create_e_i(norm_var,u_i,quantum_g_et_inv,y_e)
    updated_x_i           = QuantumModel2.updated_x_i(x_i,e_i)

    QuantumModel2.calculate_f_test(x_i, updated_x_i)