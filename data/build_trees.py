# -*- coding: utf-8 -*-
# @Time       : 2021/03/24 10:25:19
# @Author     : Zhan Genze <947783684@qq.com>
# @Project    : projects
# @Description:  this program accepts the ontology file,
#                                     the records file,(diagnose, procedure, medicine) in interger form
#                                     the voc file, which maps the disease to interger
#                                     the output_path which saves the tree path
#                             returns the new voc file (include the abstract concept)
#                                     the tree file, according to the length 
#                                     the patient visit record（only the diagnose part）
import sys, copy
import dill
import os
import argparse
import pdb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree",default = "ccs_multi_dx_tool_2015.csv",help = "the tree file for diagnose")
    parser.add_argument("--seqs",default = "records_final.pkl",help = "the seqs file")
    parser.add_argument("--voc",default = "voc_final.pkl",help = "the voc file")
    parser.add_argument("--output_path",default = "./tree/")
    args = parser.parse_args()
    
    infile = args.tree
    seqFile = args.seqs
    typeFile = args.voc
    outFile = args.output_path

    infd = open(infile, 'r')
    _ = infd.readline()

    seqs = dill.load(open(seqFile, 'rb'))
    voc = dill.load(open(typeFile, 'rb'))  # {'diag_voc': , 'med_voc': ,'pro_voc': }
    types = voc['diag_voc'].word2idx # {'A_Malaise and fatigue [252.]': 2679}

    startSet = set(types.keys())
    hitList = []
    missList = []
    cat1count = 0
    cat2count = 0
    cat3count = 0
    cat4count = 0
    for line in infd:
    # 遍历列表中每一行，记录
        tokens = line.strip().split(',')
        icd9 = tokens[0][1:-1].strip()
        cat1 = tokens[1][1:-1].strip()
        desc1 = 'A_' + tokens[2][1:-1].strip()
        cat2 = tokens[3][1:-1].strip()
        desc2 = 'A_' + tokens[4][1:-1].strip()
        cat3 = tokens[5][1:-1].strip()
        desc3 = 'A_' + tokens[6][1:-1].strip()
        cat4 = tokens[7][1:-1].strip()
        desc4 = 'A_' + tokens[8][1:-1].strip()
        # pdb.set_trace()
        # if icd9.startswith('E'):
        #     if len(icd9) > 4: icd9 = icd9[:4] + '.' + icd9[4:]
        # else:
        #     if len(icd9) > 3: icd9 = icd9[:3] + '.' + icd9[3:]
        # icd9 = 'D_' + icd9

        # 若当前icd9不存在于diagnose的类型中,即病人所患病以外的疾病
        if icd9 not in types: 
            missList.append(icd9)
        else: 
            hitList.append(icd9)

        if desc1 not in types: 
            cat1count += 1
            # 描述作为key
            types[desc1] = len(types)

        if len(cat2) > 0:
            if desc2 not in types: 
                cat2count += 1
                types[desc2] = len(types)
        if len(cat3) > 0:
            if desc3 not in types: 
                cat3count += 1
                types[desc3] = len(types)
        if len(cat4) > 0:
            if desc4 not in types: 
                cat4count += 1
                types[desc4] = len(types)
    infd.close()

    rootCode = len(types)
    types['A_ROOT'] = rootCode
    print(rootCode)                                                                                 # 2688

    print('cat1count: %d' % cat1count)                                                              # 19
    print('cat2count: %d' % cat2count)                                                              # 135
    print('cat3count: %d' % cat3count)                                                              # 366
    print('cat4count: %d' % cat4count)                                                              # 208
    print('Number of total ancestors: %d' % (cat1count + cat2count + cat3count + cat4count + 1))    # 729
    print('hit count: %d' % len(set(hitList)))                                                      # 1958
    print('miss count: %d' % len(startSet - set(hitList)))                                          # 2
    missSet = startSet - set(hitList)
    
    dill.dump(types,open("type.pkl","wb"))
    #pickle.dump(types, open(outFile + '.types', 'wb'), -1)
    #pickle.dump(missSet, open(outFile + '.miss', 'wb'), -1)


    fiveMap = {}
    fourMap = {}
    threeMap = {}
    twoMap = {}     # {113: [113, 2688, 2687]}
    oneMap = dict([(types[icd], [types[icd], rootCode]) for icd in missSet])

    infd = open(infile, 'r')
    infd.readline()

    for line in infd:
        tokens = line.strip().split(',')
        icd9 = tokens[0][1:-1].strip()
        cat1 = tokens[1][1:-1].strip()
        desc1 = 'A_' + tokens[2][1:-1].strip()
        cat2 = tokens[3][1:-1].strip()
        desc2 = 'A_' + tokens[4][1:-1].strip()
        cat3 = tokens[5][1:-1].strip()
        desc3 = 'A_' + tokens[6][1:-1].strip()
        cat4 = tokens[7][1:-1].strip()
        desc4 = 'A_' + tokens[8][1:-1].strip()

        # if icd9.startswith('E'):
        #     if len(icd9) > 4: icd9 = icd9[:4] + '.' + icd9[4:]
        # else:
        #     if len(icd9) > 3: icd9 = icd9[:3] + '.' + icd9[3:]
        # icd9 = 'D_' + icd9

        if icd9 not in types: continue
        icdCode = types[icd9]

        codeVec = []

        # 记录诊断码叶子结点到根节点的路径
        # 从最长的路径开始遍历，以免出现同时记录长度为4和3和2的路径
        if len(cat4) > 0:
            code4 = types[desc4]
            code3 = types[desc3]
            code2 = types[desc2]
            code1 = types[desc1]
            fiveMap[icdCode] = [icdCode, rootCode, code1, code2, code3, code4]
        elif len(cat3) > 0:
            code3 = types[desc3]
            code2 = types[desc2]
            code1 = types[desc1]
            fourMap[icdCode] = [icdCode, rootCode, code1, code2, code3]
        elif len(cat2) > 0:
            code2 = types[desc2]
            code1 = types[desc1]
            threeMap[icdCode] = [icdCode, rootCode, code1, code2]
        else:
            code1 = types[desc1]
            twoMap[icdCode] = [icdCode, rootCode, code1]
    
    # Now we re-map the integers to all medical codes(the concrete leaves, rather than the abstract concepts)
    newFiveMap = {}
    newFourMap = {}
    newThreeMap = {}
    newTwoMap = {}
    newOneMap = {}
    newTypes = {}
    rtypes = dict([(v, k) for k, v in types.items()])   # {2682: 'A_Rehabilitation care; fitting of prostheses; and adjustment of devices}
    # pdb.set_trace()

    codeCount = 0
    for icdCode, ancestors in fiveMap.items():
        newTypes[rtypes[icdCode]] = codeCount
        newFiveMap[codeCount] = [codeCount] + ancestors[1:]
        codeCount += 1
    for icdCode, ancestors in fourMap.items():
        newTypes[rtypes[icdCode]] = codeCount
        newFourMap[codeCount] = [codeCount] + ancestors[1:]
        codeCount += 1
    for icdCode, ancestors in threeMap.items():
        newTypes[rtypes[icdCode]] = codeCount
        newThreeMap[codeCount] = [codeCount] + ancestors[1:]
        codeCount += 1
    for icdCode, ancestors in twoMap.items():
        newTypes[rtypes[icdCode]] = codeCount
        newTwoMap[codeCount] = [codeCount] + ancestors[1:]
        codeCount += 1
    for icdCode, ancestors in oneMap.items():
        newTypes[rtypes[icdCode]] = codeCount
        newOneMap[codeCount] = [codeCount] + ancestors[1:]
        codeCount += 1

    voc['diag_voc'].word2idx = newTypes
    voc['diag_voc'].idx2word = dict([(v, k) for k, v in newTypes.items()])

    newSeqs = []
    for patient in seqs:
        newPatient = []
        for visit in patient:
            newVisit = []
            # visit: diagnose, procedure, medicine
            # pdb.set_trace()
            for code in visit[0]:
                # pdb.set_trace()
                newVisit.append(newTypes[rtypes[code]])
            visit[0] = newVisit
            # pdb.set_trace()

            newPatient.append(newVisit)
        newSeqs.append(newPatient)
    # pdb.set_trace()
    if not os.path.isdir(outFile):
        os.mkdir(outFile)
    dill.dump(newFiveMap, open(outFile + '.level5.pk', 'wb'))
    dill.dump(newFourMap, open(outFile + '.level4.pk', 'wb'))
    dill.dump(newThreeMap, open(outFile + '.level3.pk', 'wb'))
    dill.dump(newTwoMap, open(outFile + '.level2.pk', 'wb'))
    dill.dump(newOneMap, open(outFile + '.level1.pk', 'wb'))
    dill.dump(voc, open(outFile + 'voc_final.pkl', 'wb'))
    # dill.dump(newSeqs, open(outFile + 'records_final.pkl', 'wb'))
    dill.dump(seqs, open(outFile + 'records_final.pkl', 'wb'))

