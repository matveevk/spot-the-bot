import glob
import re

fnames = []
with open('full_list_of_filenames.txt', 'r') as f:
    for fname in f:
        fname = re.findall(r'[^/]+.txt', fname)[0]
        fnames.append(f'full_preprocessed/{fname}')

sticked_text = ''
for fname in fnames:
    with open(fname, 'r') as f:
        sticked_text += f.read() + '\n\n'

with open('sticked_file.txt', 'w') as f:
    f.write(sticked_text)
