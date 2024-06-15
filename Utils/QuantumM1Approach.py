import numpy as np
import random, math
import copy
from dwave.system import DWaveSampler, EmbeddingComposite
import neal
import re

from Utils.ClassicalApproach import *
from Utils.Measurement import *

class QuantumMeasurementsModel1(ClassicalMethod):
    def __init__(self, 
                 measurements: Measurement):
        
        super().__init__(measurements)
    
    def calculate_num_qubits(self,A: np.array,b: np.array) -> tuple[int,list[str]]:
        
        # check if matrix is square
        assert A.shape[0] == A.shape[1], "Matrix A is not square"
        matrix_size = A.shape[0]
        # num rows A needs to equal num rows b
        assert A.shape[0] == b.shape[0], "Matrix A and vector b are non communicative"
        max_val_b = np.max(b)
        length_max_val_b = len(str(max_val_b))
        start = 0 # look at this
        solutions = []
        val = [x for x in range(5)] # look at this
        for i in val:
            if i == 0:
                pattern = self.generate_solution(start,length_max_val_b)
                solutions.append(pattern)
            elif i == 1:
                pattern = self.generate_solution(length_max_val_b*2,length_max_val_b)
                solutions.append(pattern)
            else:
                pattern = self.generate_solution(length_max_val_b*2*i,length_max_val_b)
                solutions.append(pattern)
        max_val = self.largest_num_in_lst_of_str(solutions)
        
        return max_val, solutions
    
    def generate_solution(self, 
                          start: int, 
                          length_max_val_b: int) -> str:
        
        pattern = ""
        qubit = start
        val = [x for x in range(length_max_val_b)]
        for index in val:
            if index == 0:
                positive_term = f"{2**(index)}q{(qubit)}"
                qubit += 1
                negative_term = f"{2**(index)}q{(qubit)}"
                qubit += 1
            else:
                positive_term = f"{2**(index)}q{(qubit)}"
                qubit += 1
                negative_term = f"{2**(index)}q{(qubit)}"
                qubit += 1
            if index > 0:
                pattern += " + " + positive_term + " - " + negative_term
            else:
                pattern += positive_term + " - " + negative_term
                
        return pattern
    
    def largest_num_in_lst_of_str(self,
                                  solutions: list[str]) -> int:
        
        max_val = 0
        for i in solutions:
            numbers = re.findall(r"\d+", i)
            int_numbers = [int(num) for num in numbers]
            if max(int_numbers) > max_val:
                max_val = max(int_numbers)
                
        return max_val
    
    def extract_q_terms(self,
                        strings: str) -> dict[list[tuple[int,str,str]]]:
        
        q_dict = {}
        for index, string in enumerate(strings):
            key = f"x_{index+1}"
            terms = re.findall(r'([+-]?\d+)q(\d+)', string)
            tuples = []
            for term in terms:
                coefficient, q_index = term
                coefficient = int(coefficient)
                if int(q_index) % 2 == 1:
                    tuples.append((coefficient,"-", f"q{q_index}"))
                else:
                    tuples.append((coefficient,"+", f"q{q_index}"))
            q_dict[key] = tuples

        return q_dict

    def create_inverse_g_et(self, 
                            g_et: np.array, 
                            dimension: int, 
                            num_constraints: int, 
                            model: str = "NEAL_QUBO") -> np.array:
        
        solutions = []
        for i in range(num_constraints):
            i_vector = np.zeros([num_constraints,1])
            i_vector[i] = 1
            qubo_size, solve = self.calculate_num_qubits(g_et,i_vector)
            QM = np.zeros((qubo_size,qubo_size))
            qubits = int(qubo_size / (dimension * 2)) 
            assert QM.shape[0] == qubo_size, "not correct"
            # linear terms
            for k in range(dimension):
                for i in range(dimension):
                    for l in range(qubits):
                        cef1 = pow(2,2*l)*pow(g_et[k][i],2)
                        cef2 = pow(2,l+1)* g_et[k][i]* i_vector[k]
                        po1 = 2 * qubits * i + l 
                        po2 = 2 * qubits * i + l + qubits
                        QM[po1][po2] = QM[po1][po2] + cef1 - cef2
                        QM[po1][po2] = QM[po1][po2] + cef1 + cef2
            # first quadratic terms
            for k in range(dimension):
                for i in range(dimension):
                    for l1 in range(qubits-1):
                        for l2 in range(l1+1, qubits):
                            qcef = pow(2, l1+l2+1)*pow(g_et[k][i],2)
                            po1  = 2*qubits*i + l1
                            po2  = 2*qubits*i + l2
                            QM[po1][po2] = QM[po1][po2] + qcef
                            po3  = 2*qubits*i + l1 + qubits
                            po4  = 2*qubits*i + l2 + qubits
                            QM[po3][po4] = QM[po3][po4] + qcef
            # second quadratic terms
            for k in range(dimension):
                for i in range(dimension-1):
                    for j in range(i+1, dimension):
                        for l1 in range(qubits):
                            for l2 in range(qubits):
                                qcef = pow(2,l1+l2+1)*g_et[k][i]*g_et[k][j]
                                po1  = 2*qubits*i + l1
                                po2  = 2*qubits*j + l2
                                QM[po1][po2] = QM[po1][po2] + qcef
                                po3  = 2*qubits*i + l1 + qubits
                                po4  = 2*qubits*j + l2 + qubits
                                QM[po3][po4] = QM[po3][po4] + qcef
                                po5  = 2*qubits*i + l1
                                po6  = 2*qubits*j + l2 + qubits
                                QM[po5][po6] = QM[po5][po6] - qcef
                                po7  = 2*qubits*i + l1 + qubits
                                po8  = 2*qubits*j + l2
                                QM[po7][po8] = QM[po7][po8] - qcef
            linear = {}
            for i in range(len(QM[0])):
                linear[(f"q{i}",f"q{i}")] = QM[i][i]
            quadratic = {}
            for i in range(len(QM[0])):
                for j in range(len(QM[0])):
                    if (i != j) and (i < j):
                        quadratic[(f"q{i}",f"q{j}")] = QM[i][j]
            Q = dict(linear)
            Q.update(quadratic)
            if model == "NEAL_QUBO":
                sampler = neal.SimulatedAnnealingSampler()
                response = sampler.sample_qubo(Q,
                                            num_reads=100,
                                            label='Optimal Qubits for Inverse vector')
                vec = self.construct_answer_vector(response,solve,num_constraints)
                solutions.append(vec)
            elif model == "DWAVE":
                sample_auto = EmbeddingComposite(DWaveSampler(solver={"qpu":True}))
                sampleset = sample_auto.sample_qubo(Q, num_reads=1000)
                print(sampleset)
            else:
                print("No valid model selected")
        concatenated_array = np.concatenate(solutions, axis=1)
        
        return concatenated_array
            
    def construct_answer_vector(self, 
                                dwave_solution, 
                                solve, num_constraints) -> np.array:
        
        answer_vector = np.zeros([num_constraints,1])
        solution = self.extract_q_terms(solve)
        highest_energy_sample = None
        highest_energy = float('-inf')
        for sample, energy in dwave_solution.data(['sample', 'energy']):
            if energy > highest_energy:
                highest_energy = energy
                highest_energy_sample = sample
        l = 1
        for i in range(answer_vector.shape[0]):
            val = 0
            for j in highest_energy_sample.keys():
                for k in solution[f"x_{l}"]:
                    if j == k[2]:
                        if k[1] == "+":
                            val += k[0] * highest_energy_sample[f"{k[2]}"]
                        else:
                            val -= k[0] * highest_energy_sample[f"{k[2]}"]
            l += 1
            answer_vector[i] = val
            
        return answer_vector