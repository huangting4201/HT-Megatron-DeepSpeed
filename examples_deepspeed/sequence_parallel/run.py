import os
import subprocess

# seqlen_list = [128 * 1024, 256 * 1024, 512 * 1024, 1024 * 1024]
seqlen_list = [64 * 1024]

# node_num_list = [4, 8, 16, 32, 64]
node_num_list = [4]

# for i, seqlen in enumerate(seqlen_list):
for i, node_num in enumerate(node_num_list):

    # 运行命令
    # command = f"srun -p Intern5 -N 8 -n 64 --ntasks-per-node=8 --gpus-per-task=1 python ../train.py --config ./{output_path} --profiling 2>&1 | tee '{log_path}'"
    command = f"srun -p llm_s --preempt --exclusive -N{node_num} --ntasks-per-node=1 --gpus-per-task=8 sh sp_7B.sh {seqlen_list[0]} 2>&1 | tee '{8*node_num}_zero3_{seqlen_list[0]}_mla_70b.log'"
    process = subprocess.run(command, shell=True)

    if process.returncode != 0:
        print(f"运行命令时出错：{command}")
    else:
        print(f"命令成功运行：{command}")
