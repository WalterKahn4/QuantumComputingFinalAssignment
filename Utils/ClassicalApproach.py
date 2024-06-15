import numpy as np
import random
from scipy.stats import f_oneway

from Utils.Measurement import *

class ClassicalMethod:
    def __init__(self,
                 measurement: list) -> None:
        
        self.measurements = measurement
        
    def length(self) -> int:
        
        return len(self.measurements)
        
    def create_x_i(self) -> np.array:
        
        x_i = np.zeros([self.length(),1])
        for x_index,measurement in enumerate(self.measurements):
            x_i[x_index] = measurement.hoogteverschil
            
        return x_i
    
    def create_var_a(self) -> np.array:
        
        var_a = np.zeros([self.length(),self.length()])
        for index, measurement in enumerate(self.measurements):
            var_a[index][index] = measurement.std_a
            
        return var_a
    
    def create_var_b(self) -> np.array:
        
        var_b = np.zeros([self.length(),self.length()])
        for index, measurement in enumerate(self.measurements):
            var_b[index][index] = measurement.std_b * np.sqrt(measurement.afstand/1000)
            
        return var_b
    
    def create_var_c(self) -> np.array:
        
        var_c = np.zeros([self.length(),self.length()])
        for index, measurement in enumerate(self.measurements):
            var_c[index][index] = measurement.std_c
            
        return var_c
    
    def normalize(self, x: float,x_min: float,x_max: float) -> float:
        
        x_new = (x - x_min) / np.abs(x_min - x_max)
        
        return x_new
    
    def create_var(self) -> np.array:
        
        var_a = self.create_var_a()
        var_b = self.create_var_b()
        var_c = self.create_var_c()
        var = var_a + var_b + var_c
        # normalize matrix
        num_measurements = var.shape[0]
        # get min value
        min_value = float("inf")
        for i in range(num_measurements):
            if var[i][i] < min_value:
                 min_value = var[i][i]
        # get max value
        max_value = 0
        for i in range(num_measurements):
            if var[i][i] > max_value:
                 max_value = var[i][i]
        new_var = np.zeros([num_measurements,num_measurements])
        # normalize mat
        for i in range(num_measurements):
            new_var[i][i] = self.normalize(var[i][i],min_value,max_value)
            
        return new_var, var
    
    def create_adjacency_matrix(self) -> np.array:
        
        matrix_size = self.length()
        adjacency_matrix = np.zeros([matrix_size, matrix_size], dtype=int)
        nodes = set()
        for measurement in self.measurements:
            nodes.add(measurement.start)
            nodes.add(measurement.eind)
        nodes = sorted(list(nodes))
        node_indices = {node: idx for idx, node in enumerate(nodes)}
        for measurement in self.measurements:
            start_idx = node_indices[measurement.start]
            end_idx = node_indices[measurement.eind]
            adjacency_matrix[start_idx, end_idx] = 1
            
        return adjacency_matrix
        
    def adjacency_lists_from_matrix(self, 
                                    A: np.array) -> dict:
        
        adj_lists = {}
        n = len(A)
        for i in range(n):
            adj_list = []
            for j in range(n):
                if A[i][j] == 1:
                    adj_list.append(j)
            adj_lists[i] = adj_list
            
        return adj_lists
    
    def find_circuits(self, 
                      A: np.array) -> list:
        
        adj_list = self.adjacency_lists_from_matrix(A)
        circuits = []
        def dfs(node, start, path, visited_edges):
            path.append(node)
            if len(path) > 1 and node == start:
                circuits.append(path[:])
                path.pop()
                return
            for neighbor in adj_list[node]:
                edge = (node, neighbor)
                if edge not in visited_edges:
                    visited_edges.add(edge)
                    visited_edges.add((neighbor, node))  # For undirected graphs
                    dfs(neighbor, start, path, visited_edges)
                    visited_edges.remove(edge)
                    visited_edges.remove((neighbor, node))
            path.pop()
        for start in adj_list.keys():
            dfs(start, start, [], set())
            
        return circuits
    
    def longest(self, 
                list_of_lists: list) -> int:
        
        longest_list = max(len(elem) for elem in list_of_lists)
        
        return longest_list

    def return_longest_x(self, 
                         list_of_lists: list, x: int) -> list:
        
        longest_val = self.longest(list_of_lists)
        conditions = [longest_val-y for y in range(x)]
        longest_constraints = []
        for i in list_of_lists:
            for condition in conditions:
                if len(i) == condition:
                    longest_constraints.append(i)
                    
        return longest_constraints
    
    def random_longest(self, 
                       longest_constraints: list, 
                       num_constraints: int) -> list:
        
        amount_longest = len(longest_constraints)
        n_longest_constraints = []
        for i in range(num_constraints):
            v = random.randint(0, amount_longest)
            n_longest_constraints.append(longest_constraints[v])
            
        return n_longest_constraints
    
    def create_u_i(self, 
                   num_constraints: int) -> np.array:
        
        u_i = np.zeros([num_constraints,self.length()])
        # improved way of making constraints
        A = self.create_adjacency_matrix()
        circuits = self.find_circuits(A)
        selected_longest = self.return_longest_x(circuits,num_constraints)
        n_chosen = self.random_longest(selected_longest, num_constraints)
        for i in range(num_constraints - 1):
            for j in n_chosen[i]:
                u_i[i][j] = 1
        last_constraint = random.randint(0,self.length())
        u_i[-1][last_constraint] = 1
        
        return u_i, last_constraint
    
    def create_u_o(self, 
                   num_constraints: int, 
                   last_contraint: int) -> np.array:
        
        u_o = np.zeros([num_constraints,1])
        last_value = self.measurements[last_contraint].hoogteverschil
        u_o[-1] = last_value    
        
        return u_o
    
    def create_y_e(self, 
                   x_i: np.array, 
                   u_i: np.array, 
                   u_o: np.array) -> np.array:
        
        return u_i.dot(x_i) - u_o
    
    def create_g_et(self, 
                    u_i: np.array, 
                    var: np.array) -> np.array:
        
        g_et = (u_i @ var @ np.transpose(u_i))
        
        return g_et
    
    def create_inverse_g_et(self, 
                            g_et: np.array) -> np.array:
        
        try:
            return np.linalg.inv(g_et), True
        except:
            
            return g_et, False
    
    def create_e_i(self, 
                   var: np.array, 
                   u_i: np.array, 
                   inverse_g_et: np.array, 
                   y_e: np.array) -> np.array:
        
        return var @ np.transpose(u_i) @ inverse_g_et @ (-1 * y_e)
    
    def updated_x_i(self, 
                    x_i: np.array, 
                    e_i: np.array) -> np.array:
        
        return x_i + e_i
    
    def calculate_f_test(self,
                         x_original: np.array,
                         x_updated: np.array,
                         alpha: float = 0.05) -> None:
        
        f_statistic, p_value = f_oneway(x_original,x_updated)        
        print("F-statistic:", f_statistic)
        if p_value < alpha:
            print("Reject the null hypothesis. There are significant differences between the groups.")
        else:
            print("Fail to reject the null hypothesis. There are no significant differences between the groups.")