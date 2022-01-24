nohup python3.7  -m paddle.distributed.launch \
          --selected_gpus="4,5,6,7" \
          retrain_admm.py \
          --batch_size 64 \
          --data mnist \
          --pruning_mode ratio \
          --ratio 0.90 \
          --lr 0.01 \
          --model MobileNet \
          --pretrained_model "save_models" \
          --num_epochs 30 \
          --step_epochs  71 88 \
          --initial_ratio 0.15 \
          --pruning_steps 100 \
          --stable_epochs 0 \
          --pruning_epochs 54 \
          --tunning_epochs 54 \
          --last_epoch -1 \
          --pruning_strategy gmp \
          --local_sparsity True \
          --prune_params_type 'conv1x1_only' \
          --rho 1 \
          1>log.txt \
          2>error.log &
