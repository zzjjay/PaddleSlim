nohup python3.7  -m paddle.distributed.launch \
          --selected_gpus="0,1,2,3" \
          train_admm.py \
          --batch_size 64 \
          --data imagenet \
          --pruning_mode ratio \
          --ratio 0.75 \
          --lr 0.01 \
          --model MobileNet \
          --num_epochs 50 \
          --pretrained_model "MobileNetV1_pretrained" \
          --step_epochs  10 30 \
          --initial_ratio 0.15 \
          --pruning_steps 100 \
          --stable_epochs 0 \
          --pruning_epochs 54 \
          --tunning_epochs 54 \
          --last_epoch -1 \
          --pruning_strategy gmp \
          --local_sparsity True \
          --prune_params_type 'conv1x1_only' \
          --rho 0.1 \
          1>log.txt \
          2>error.log &
