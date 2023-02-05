#!/bin/bash
#SBATCH --job-name=train6
#SBATCH --output=/homes/pmartinello/output_tr_6.txt
#SBATCH --error=/homes/pmartinello/error_tr_6.txt
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --partition=students-prod

export PYTHONNOUSERSITE=1
cd /homes/pmartinello/People-Analyzer/ReID

IFS=',' read -r -a nodelist <<<$SLURM_NODELIST

echo "${nodelist[*]}"

export MASTER_ADDR="${nodelist[0]}"
export MASTER_PORT=43437
export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=1


base="
--train_path /work/cvcs_2022_group02/datasets/singleclip/train
--query_path /work/cvcs_2022_group02/datasets/singleclip/queries
--gallery_path /work/cvcs_2022_group02/datasets/singleclip/gallery

--checkpoints_folder /work/cvcs_2022_group02/training_6/checkpoints
--results_folder /work/cvcs_2022_group02/training_6/results

--num_classes 1000
--max_epoch 100

--triplet_loss_multiplier 10.0
--center_loss_multiplier 0.025

--batch_size 16
--queries_batch 30
--checkpoint_every 1

--workers 4
--test_interval 2
"

# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate /homes/pmartinello/.conda/envs/cvcs22_06

for i in {1..1}
do
  srun -N1 -n1 -w "${nodelist[$i-1]}" --gres=gpu:1 python -u -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=$((i-1)) --use_env --master_addr="$MASTER_ADDR" --master_port=$MASTER_PORT deep_trainer.py ${base} &
done

wait

