import scipy.io
import numpy as np
import csv

lab2_1 = scipy.io.loadmat('lab2_1.mat')
lab2_2 = scipy.io.loadmat('lab2_2.mat')
lab2_3 = scipy.io.loadmat('lab2_3.mat')

# Lab 2_1
np.savetxt('lab2_1_a.csv', lab2_1.get('a'), fmt='%s', delimiter=',')
np.savetxt('lab2_1_b.csv', lab2_1.get('b'), fmt='%s', delimiter=',')

# Lab 2_2
np.savetxt('lab2_2_al.csv', lab2_2.get('al'), fmt='%s', delimiter=',')
np.savetxt('lab2_2_bl.csv', lab2_2.get('bl'), fmt='%s', delimiter=',')
np.savetxt('lab2_2_cl.csv', lab2_2.get('cl'), fmt='%s', delimiter=',')
np.savetxt('lab2_2_at.csv', lab2_2.get('at'), fmt='%s', delimiter=',')
np.savetxt('lab2_2_bt.csv', lab2_2.get('bt'), fmt='%s', delimiter=',')
np.savetxt('lab2_2_ct.csv', lab2_2.get('ct'), fmt='%s', delimiter=',')

# Lab 2_3
np.savetxt('lab2_3_a.csv', lab2_3.get('a'), fmt='%s', delimiter=',')
np.savetxt('lab2_3_b.csv', lab2_3.get('b'), fmt='%s', delimiter=',')
