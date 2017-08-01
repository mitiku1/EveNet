import numpy as np


dat = np.genfromtxt("/home/ralf/HR/EveNet/EveNet/test/reader_test_data/test.dat", delimiter=",", dtype=float)
emo = np.genfromtxt("/home/ralf/HR/EveNet/EveNet/test/reader_test_data/test.emo", delimiter=",", dtype=str)
pho = np.genfromtxt("/home/ralf/HR/EveNet/EveNet/test/reader_test_data/test.pho", delimiter=",", dtype=str)

mock_dat = []
mock_emo = []
mock_pho = []

for i in range(1000):
    random_index = int(np.round(np.random.random() * len(dat)))-1
    mock_dat.append(dat[random_index])
    #mock_dat.append(np.maximum(dat[random_index] + ((np.random.random()-0.5)*0.05),0.0))
    mock_emo.append(emo[random_index])
    mock_pho.append(pho[random_index])

np.savetxt("/home/ralf/HR/EveNet/EveNet/test/generation_test_data/test.dat", mock_dat, delimiter=",")
np.savetxt("/home/ralf/HR/EveNet/EveNet/test/generation_test_data/test.emo", mock_emo, delimiter=",", fmt="%s")
np.savetxt("/home/ralf/HR/EveNet/EveNet/test/generation_test_data/test.pho", mock_pho, delimiter=",", fmt="%s")
