import json
import numpy as np

SEGMENT_NR = 80
SEGMENT_MIN = 50
SEGMENT_MAX = 300

with open("./config.json") as json_file:
    reader_config = json.load(json_file)

c_emo = reader_config['emotion_categories']
c_pho = reader_config['phoneme_categories']

mock_dat = []
mock_emo = []
mock_pho = []

for _ in range(SEGMENT_NR):
    emo_index = np.random.randint(len(c_emo))
    pho_index = np.random.randint(len(c_pho))

    amplitude = 0.1 + (float(emo_index) / float(len(c_emo)+1))
    frequency = 0.5 + (float(pho_index) / float(len(c_pho))) * 5.0

    x = np.arange(SEGMENT_MIN + int(np.random.random() * (SEGMENT_MAX-SEGMENT_MIN)))
    sindata = np.abs(np.sin(x * frequency) * amplitude)
    cosdata = np.abs(np.cos(x * frequency) * amplitude)

    for i in range(len(x)):
        mock_dat.append([sindata[i], cosdata[i]])
        mock_emo.append(c_emo[emo_index])
        mock_pho.append(c_pho[pho_index])

np.savetxt("./test.dat", np.array(mock_dat), delimiter=",")
np.savetxt("./test.emo", mock_emo, delimiter=",", fmt="%s")
np.savetxt("./test.pho", mock_pho, delimiter=",", fmt="%s")
