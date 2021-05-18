from numpy.core.records import record
import pandas as pd
import pickle
import dill
import pdb
import os

if __name__ == "__main__":
    ehr='ehr_adj_final.pkl'
    ehr_adj = pickle.load(open(ehr,"rb"))
    data_path = '../data/tree/records_final.pkl'
    
    records = pickle.load(open(data_path,"rb"))
    total_med = 0
    med_map = {}
    for patient in records:
        for visit in patient:
            total_med += len(visit[2])
            for med in visit[2]:
                if med in med_map.keys():
                    med_map[med] +=1
                else:
                    med_map[med] =1
    
    rare_med = {}
    for med in med_map.keys():
        if(med_map[med]<=10000):
            rare_med[med] = med_map[med]
    
    print(sorted(med_map.items(), key=lambda item:item[1]))
    print(len(rare_med)/ len(med_map))
    with open('rare_med.pkl', 'wb') as f:
        pickle.dump(list(rare_med.keys()), f)
    # pdb.set_trace()