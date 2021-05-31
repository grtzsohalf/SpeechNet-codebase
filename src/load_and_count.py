import glob
import torchaudio
import librosa
from collections import Counter
from tqdm import tqdm

#names = [x for x in glob.glob('/VoxCeleb1/*/*/*.wav')]
#names = [x for x in glob.glob('/CMU18/*/*/*/*.wav')]
names = [x for x in glob.glob('/VCTK/*/*/*.wav')]

lengths = Counter()
lengths_d = Counter()
for n in tqdm(names):
    wav_len = librosa.get_duration(filename=n)
    #wav, sr = torchaudio.load(n)
    #print(len(wav[0]), sr)
    lengths[wav_len] += 1
    lengths_d[int(wav_len)] += 1
print(lengths)
print(lengths_d)
#with open('VoxCeleb1_lengths.txt', 'w') as f:
#with open('CMU18_lengths.txt', 'w') as f:
with open('VCTK_lengths.txt', 'w') as f:
    for k, v in sorted(lengths.items(), key = lambda i: i[0]):
        f.write(str(k)+' '+str(v)+'\n')
#with open('VoxCeleb1_lengths_d.txt', 'w') as f:
#with open('CMU18_lengths_d.txt', 'w') as f:
with open('VCTK_lengths_d.txt', 'w') as f:
    for k, v in sorted(lengths_d.items(), key = lambda i: i[0]):
        f.write(str(k)+' '+str(v)+'\n')
