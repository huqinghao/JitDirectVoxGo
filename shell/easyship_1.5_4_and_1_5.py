# train all tests for near 1.5, far 4 and near 1, far 5

import os

CUDA_IDs = [0,1,2,3,4,5,6,7]
folders = ["configs/easyship_near1_far5", "configs/easyship_near1.5_far4", "configs/easyship_near2_far6", "configs/easyship_near2_far4"]

if __name__ == "__main__":
    per_filenames = os.listdir(folders[0])
    total_filenames = []
    for folder in folders:
        for fname in per_filenames:
            total_filenames.append(os.path.join(folder,fname))
            

    print(total_filenames)
    
    
    cmd = ""

    for idx in range(len(CUDA_IDs)):
        CUDA_ID = CUDA_IDs[idx]
        curr_filenames = total_filenames[idx::len(CUDA_IDs)]
        for fname in curr_filenames:
            cmd += f"""CUDA_VISIBLE_DEVICES={CUDA_ID} python run.py --config {fname} --render_val --render_test && """
        
        cmd = cmd[:-3] + " & "

    print(cmd)
    os.system(cmd)