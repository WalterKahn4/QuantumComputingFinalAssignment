from Utils.Measurement import *

# Chose a project to split from "Metingen_NNL.csv" into a seperate csv that can be read into main

def project_splitser(Project: str):
    lst = []
    with open("Data/Metingen_NNL.csv", "r") as file:
        for i in file.readlines():
            parts = i.split(",")
            if parts[3] == Project:
                lst.append(i)
    with open(f"Projects\{Project}.csv", "w") as output_file:
        output_file.write(f"start_punt,eind_punt,gebruiken,project_id,afstand,hoogteverschil,meetdatum,std_afw_dh_a,std_afw_dh_b,std_afw_dh_c,,,,,,")
        for i in lst:
            output_file.write(i)
            
def read_measurements(file):
    measurements = []
    file.readline()
    
    ## read selected project
    
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
        
    return measurements