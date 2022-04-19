import pandas as pd

f = open('evaluation_policy.txt', 'r')

contenuto = f.readlines()

array_temporaneo_secondo = []
array_temporaneo_terzo = []

for i in range(len(contenuto)):
	array_temporaneo = ''
	if 'Episode number' in contenuto[i]:
		for j in range(len(contenuto[i])):
			if contenuto[i][j].isdigit():
				array_temporaneo += contenuto[i][j]
		array_temporaneo_secondo.append(array_temporaneo)
	if 'Metric' in contenuto[i]:
		for j in range(len(contenuto[i])):
			if contenuto[i][j].isdigit() or contenuto[i][j] == '.':
				array_temporaneo += contenuto[i][j]
		array_temporaneo_terzo.append(array_temporaneo)

array = []

for i in range(len(array_temporaneo_secondo)):
    array.append([array_temporaneo_secondo[i], array_temporaneo_terzo[i]])
    

import numpy as np

np.savetxt("GFG.csv", 
           array,
           delimiter =", ", 
           fmt ='% s')

import pandas as pd

read_file = pd.read_csv ('GFG.csv')
read_file.to_excel ('evaluation.xlsx', index = None, header=True)
