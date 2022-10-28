import numpy as np
import os 
import imageio

dir1 = "new_result/"
dir2 = "old_result/"

new_fnames = os.listdir(dir1)
old_fnames = os.listdir(dir2)


for fname in new_fnames:
    if fname not in old_fnames:
        continue

    file1 = (np.array(imageio.imread(os.path.join(dir1,fname))) /255. ).astype(np.float32)
    file2 = (np.array(imageio.imread(os.path.join(dir2,fname))) /255. ).astype(np.float32)

    psnr = -10. * np.log10(np.mean(np.square(file1 - file2)))

    print(fname, psnr)