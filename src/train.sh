git pull
source ~/miniconda/etc/profile.d/conda.sh
conda activate ml_env
python3 ./vsa_encoder/train.py --batch_size 512 --path_to_dataset "../one_exchange/" --seed 0 --mode "dsprites" --devices 0 --max_epochs 400 --lr 0.00025 --kld_coef 0.001

