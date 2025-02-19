import os
import shutil

data_dir = "datasets/PS_UM2020_128/valid/"
new_data_dir = "datasets/PS_UM2020_128_div8/valid/"
new_inputs = "datasets/PS_UM2020_128_div8/valid/inputs/"
new_targets = "datasets/PS_UM2020_128_div8/valid/targets/"
os.makedirs(new_data_dir, exist_ok=True)
os.makedirs(new_inputs, exist_ok=True)
os.makedirs(new_targets, exist_ok=True)
input_dir = data_dir + "inputs/"

n = 8

inputs = sorted(
    [os.path.join(input_dir, fname)
     for fname in os.listdir(input_dir)
     if fname.endswith(".tif") and not fname.startswith(".")]
)

for i in range(0, len(inputs)):
    if i % n == n - 1:
        target = inputs[i].replace("/inputs/", "/targets/")
        input_bname = os.path.basename(inputs[i])
        target_bname = os.path.basename(target)
        shutil.copyfile(inputs[i], new_inputs + input_bname)
        shutil.copyfile(target, new_targets + target_bname)