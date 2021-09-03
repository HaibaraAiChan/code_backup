#!/bin/bash
export RANK=0
export WORLD_SIZE=1
# batch_size=(24577 12289 6245 3000 1500)
batch_size=(196571 98308 49154 24577 12289 6245 3000 1500)
# batch_size=(1000 )
# python reddit_base_test.py --aggre lstm --batch-size $i --num-epochs 6 --eval-every 5 > lstm_baseline_log/bs_${i}_6_epoch.log
# for i in ${batch_size[@]};do
#   # python products_pseudo.py --batch-size $i > pseudo_log_1/bs_${i}_5_epoch.log
#   # python reddit_base_test.py --aggre lstm --batch-size $i --num-epochs 6 --eval-every 5 > lstm_baseline_log/bs_${i}_6_epoch.log
#   python products_pseudo.py --aggre lstm --batch-size $i --num-epochs 6 --eval-every 5 > lstm_pseudo_log/bs_${i}_6_epoch.log
#   # python products_pseudo.py --batch-size $i --num-epochs 26 --eval-every 25 > pseudo_log_1/bs_${i}_26_epoch.log
# done
# for i in ${batch_size[@]};do

#   # CUDA_VISIBLE_DEVICES=0 deepspeed --num_gpus=1  train_serial.py --batch-size $i --deepspeed_config ds_config.json > serial_log/${i}_101.log
#   CUDA_VISIBLE_DEVICES=0 deepspeed --num_gpus=1  deepspeed_pseudo_train.py --aggre lstm --batch-size $i --log-every 120 --deepspeed_config ds_config.json > pseudo_log_1/${i}_101.log
# done


# reddit dataset 
batch_size=(153431 76716 38357 19178 9590 4795 2400 1200)

for i in ${batch_size[@]};do
  
  python products_pseudo_pure_.py --dataset reddit --aggre mean --batch-size $i --num-epochs 6 --eval-every 5 > reddit_mean_pseudo_log/bs_${i}_6_epoch.log
  # python products_pseudo.py --batch-size $i --num-epochs 26 --eval-every 25 > pseudo_log_1/bs_${i}_26_epoch.log
done