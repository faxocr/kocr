import sys
import subprocess
import os
import numpy as np
from PIL import Image


preprocess = "../src/preprocess"
remove_conv_images = True


if len(sys.argv) != 2:
    sys.exit(1)

target = sys.argv[1]
if target[-1] != "/":
    target += "/"

if not os.path.exists(target):
    print "target-dir not exists"
    sys.exit(1)

conv_files = []
for file_name in os.listdir(target):
    if file_name[-4:] not in [".png", ".pbm", ".jpg"]:
        continue

    file_path = target + file_name
    try:
        subprocess.check_output(preprocess + " " + file_path, shell=True)
        conv_files.append(file_name[:-4] + "-conv.png")
    except KeyboardInterrupt:
        break
    except:
        print "An error occurred:", file_name

X, y = [], []
for file_name in conv_files:
    y.append(file_name.split("-")[0])
    X.append(np.array(Image.open(target + file_name)).reshape([-1]))

X, y = np.array(X), np.array(y)
print X.shape, y.shape
np.save("images.npy", X)
np.save("labels.npy", y)

if remove_conv_images:
    for file_name in conv_files:
        os.remove(target + file_name)
