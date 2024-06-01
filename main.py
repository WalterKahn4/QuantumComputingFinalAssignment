import numpy as np
import sys
import time

from ClassicalApproach import *
from QuantumM1Approach import *
from QuantumM2Approach import *
# from MessyGraph        import *

np.set_printoptions(threshold=4,suppress=True)

# # steps
# 1 load data 
# TODO create messy graph in MessyGraph.py
# TODO add QGIS exporter for graph (maybe)
# 2 create x_i
# 3 create var 
# TODO normalize the var (completed)
# 4 create y_e
# TODO select better constraints (completed)
# TODO add QGIS exporter for constraints (maybe)
# 5 create g_et
# 6 invert g_et 
# TODO implement QUBO model 2 (completed)
# TODO allow non dwave implementation (completed)
# TODO calculate number of qubits (completed)
# TODO translate final result of each Ax = b into a vector i (completed)
# TODO concatenate i vectors into the inverse matrix and return it (completed)
# 7 create e_i
# 8 update x_i
# 9 calculate f-test go back to # 2 if test is rejected

with open("189W63-ZOU.csv", "r") as file:
    measurements = []
    file.readline()
    
    for index, line in enumerate(file.readlines()):
        parts = line.split(",")
        start          = parts[0]    # = start
        eind           = parts[1]    # = eind
        afstand        = parts[4]    # = afstand
        hoogteverschil = parts[5]    # = hoogteverschil
        std_a          = parts[7]    # = std_a
        std_b          = parts[8]    # = std_b
        std_c          = parts[9]    # = std_c
        measurement = Measurement(index=index,start=start,eind=eind,afstand=afstand,hoogteverschil=hoogteverschil,std_a=std_a,std_b=std_b,std_c=std_c)
        measurements.append(measurement)
        
    Classic = Measurements(measurements)
    QuantumModel1 = QuantumMeasurementsModel1(measurements)
    QuantumModel2 = QuantumMeasurementsModel2(measurements)
    
    num_constraints = 5
    
    x_i  = Classic.create_x_i()
    norm_var, var  = Classic.create_var()
    u_i, last_constraint  = Classic.create_u_i(num_constraints)
    u_o  = Classic.create_u_o(num_constraints,last_constraint)
    y_e  = Classic.create_y_e(x_i,u_i,u_o)
    g_et = Classic.create_g_et(u_i,norm_var)
    g_et_inv = Classic.create_inverse_g_et(g_et)
    e_i      = Classic.create_e_i(norm_var,u_i,g_et_inv,y_e)
    updated_x_i      = Classic.updated_x_i(x_i,e_i)
    
    # for original, updated, adjusted in zip(x_i,updated_x_i,e_i):
    #     print(f"original value: {original} || updated value {updated} || changed by {adjusted}")

    Classic.calculate_f_test(x_i, updated_x_i)
    
    x_i                   = QuantumModel1.create_x_i()
    norm_var, var         = QuantumModel1.create_var()
    u_i, last_constraint  = QuantumModel1.create_u_i(num_constraints)
    u_o                   = QuantumModel1.create_u_o(num_constraints,last_constraint)
    y_e                   = QuantumModel1.create_y_e(x_i,u_i,u_o)
    g_et                  = QuantumModel1.create_g_et(u_i,norm_var)
    quantum_g_et_inv      = QuantumModel1.create_inverse_g_et(g_et=g_et,dimension=num_constraints,num_constraints=num_constraints)
    e_i                   = QuantumModel1.create_e_i(norm_var,u_i,quantum_g_et_inv,y_e)
    updated_x_i           = QuantumModel1.updated_x_i(x_i,e_i)
    
    # for original, updated, adjusted in zip(x_i,updated_x_i,e_i):
    #     print(f"original value: {original} || updated value {updated} || changed by {adjusted}")

    QuantumModel1.calculate_f_test(x_i, updated_x_i)
    
    x_i                   = QuantumModel2.create_x_i()
    norm_var, var         = QuantumModel2.create_var()
    u_i, last_constraint  = QuantumModel2.create_u_i(num_constraints)
    u_o                   = QuantumModel2.create_u_o(num_constraints,last_constraint)
    y_e                   = QuantumModel2.create_y_e(x_i,u_i,u_o)
    g_et                  = QuantumModel2.create_g_et(u_i,norm_var)
    quantum_g_et_inv      = QuantumModel2.create_inverse_g_et(g_et=g_et,dimension=num_constraints,num_constraints=num_constraints)
    e_i                   = QuantumModel2.create_e_i(norm_var,u_i,quantum_g_et_inv,y_e)
    updated_x_i           = QuantumModel2.updated_x_i(x_i,e_i)
    
    # for original, updated, adjusted in zip(x_i,updated_x_i,e_i):
    #     print(f"original value: {original} || updated value {updated} || changed by {adjusted}")

    QuantumModel2.calculate_f_test(x_i, updated_x_i)