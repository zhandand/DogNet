# -*- coding: utf-8 -*-
# @Time       : 2021/03/24 10:57:53
# @Author     : Zhan Genze <947783684@qq.com>
# @Project    : projects
# @Description:  this program accepts the tree_path which saves the treee information
#									  the  patient diagnose records
# 							  returns the cooccurence map for diagnose in map format (this is for glove.py)
import sys
import dill
import numpy as np
import argparse
from tqdm import tqdm

def augmentVisit(visit, code, treeList):
	'''
		{code, rootCode, code1, code2,...,coden}
		根据code找到Leaf后，加入rootCode以及后面的结点
	'''
	for tree in treeList:
		if code in tree:
			visit.extend(tree[code][1:])
			break
	return

def countCooccurrenceProduct(visit, coMap):
	""" 根据augmented visit 构建共现矩阵 

	Args:
		visit (list): the augmented visit for a single visit
		coMap (dict({(disease,disease):cooccurrence})): 两种medical code的共现信息
	"""	
	codeSet = set()
	for code in visit:
		codeSet.add(code)

	for code1 in codeSet:
		for code2 in codeSet:
			if code1 == code2: continue

			product = visit.count(code1) * visit.count(code2)
			key1 = (code1, code2)
			key2 = (code2, code1)

			if key1 in coMap: coMap[key1] += product
			else: coMap[key1] = product

			if key2 in coMap: coMap[key2] += product
			else: coMap[key2] = product


if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--tree_path",default = "./tree/",help = "the path that stores tree information")
	parser.add_argument("--seqs",default = "records_final.pkl",help = "the seqs file")
	# parser.add_argument("--voc",default = "voc.pkl",help = "the voc file")
	args = parser.parse_args()	

	seqFile = args.seqs
	treeFile = args.tree_path
	outFile = 'cooccurrenceMap.pkl'

	maxLevel = 5
	seqs = dill.load(open(seqFile, 'rb'))
	treeList = [dill.load(open(treeFile+'.level'+str(i)+'.pk', 'rb')) for i in range(1,maxLevel+1)]

	coMap = {}
	count = 0
	for patient in tqdm(seqs):
		# if count % 1000 == 0: print (count)
		count += 1
		for visit in patient:
			# 扩充diagnose的数据，加入根节点到叶子节点路径上的结点
			for code in visit[0]: 
				augmentVisit(visit[0], code, treeList)
			# 对于一次visit，计算augmented visit后，构建co-occurrence maxtrix
			countCooccurrenceProduct(visit[0], coMap)
	
	dill.dump(coMap, open(outFile, 'wb'))
