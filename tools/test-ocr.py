import os
import subprocess
import argparse
import re
from collections import defaultdict


parser = argparse.ArgumentParser(description='aaa')
parser.add_argument('target_dir', type=str)
parser.add_argument('--exe', type=str, default='../src/kocr')
parser.add_argument('--weights', type=str, default='../databases/cnn-num.bin')

args = parser.parse_args()
cmd = args.exe + ' ' + args.weights + ' '

results = defaultdict(lambda:defaultdict(int))

total, correct = 0, 0
for name in os.listdir(args.target_dir):
    if not name.endswith('.png'):
        continue
    path = os.path.join(args.target_dir, name)
    res = subprocess.check_output(cmd + path, shell=True)

    ans = name[0]    
    pred = re.search(r'Result: (.*)\n', res).group(1)
    results[pred][ans] += 1

    total += 1
    correct += (ans == pred)


# Print the cross table.
rows = sorted(set(results.keys()))
cols = sorted(set(sum([v.keys() for v in results.values()], [])))

crosstab = ['' for _ in range(len(rows) + 1)]

max_len = max([len(r) for r in rows]) + 1
crosstab[0] += ' ' * max_len
for i, r in enumerate(rows):
    crosstab[i + 1] += ' ' * (max_len - len(r)) + r

for c in cols:
    lengths = [len(str(results[r][c])) for r in rows]
    max_len = max(lengths) + 1
    crosstab[0] += ' ' * (max_len - len(c)) + c
    for i, (l, r) in enumerate(zip(lengths, rows)):
        crosstab[i + 1] += ' ' * (max_len - l) + str(results[r][c])

for line in crosstab:
    print(line)

print('Accuracy: {:.5f}'.format(float(correct) / total))