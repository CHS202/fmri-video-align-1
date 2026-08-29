#!/bin/bash

# python main.py --task design --loss_func ce
# python main.py --data_use sub-01 --task design --loss_func ce
# python main.py --data_use sub-02 --task design --loss_func ce
# python main.py --data_use sub-03 --task design --loss_func ce
# python main.py --data_use sub-09 --task design --loss_func ce
# python main.py --data_use sub-06 --task design --loss_func ce
# python main.py --data_use sub-08 --task design --loss_func ce
# python main.py --data_use sub-12 --task design --loss_func ce

# TODO (remember to modify the path in utils.py)
# --co_train True --train_from_checkpoint True --freezeall True
# python main.py --task annot-reg --loss_func mse
python main.py --data_use sub-01 --task annot-reg --loss_func mse
python main.py --data_use sub-02 --task annot-reg --loss_func mse
python main.py --data_use sub-03 --task annot-reg --loss_func mse
python main.py --data_use sub-09 --task annot-reg --loss_func mse
python main.py --data_use sub-06 --task annot-reg --loss_func mse
python main.py --data_use sub-08 --task annot-reg --loss_func mse
python main.py --data_use sub-12 --task annot-reg --loss_func mse

# python main.py --data_use sub-01 --task annot --loss_func bce
# python main.py --data_use sub-02 --task annot --loss_func bce
# python main.py --data_use sub-03 --task annot --loss_func bce
# python main.py --data_use sub-09 --task annot --loss_func bce
# python main.py --data_use sub-06 --task annot --loss_func bce
# python main.py --data_use sub-08 --task annot --loss_func bce
# python main.py --data_use sub-12 --task annot --loss_func bce
# python main.py --task annot --loss_func bce

# for i in 'obj' 'subj'; do
# python main.py --data_use sub-06 --task annot-reg --loss_func mse --target_class 'obj'
# python main.py --data_use sub-08 --task annot-reg --loss_func mse --target_class 'obj'
# python main.py --data_use sub-12 --task annot-reg --loss_func mse --target_class 'obj'
    # python main.py --task annot-reg --loss_func mse --target_class $i
# python main.py --data_use sub-01 --task annot-reg --loss_func mse --target_class 'subj'
# python main.py --data_use sub-02 --task annot-reg --loss_func mse --target_class 'subj'
# python main.py --data_use sub-03 --task annot-reg --loss_func mse --target_class 'subj'
# python main.py --data_use sub-09 --task annot-reg --loss_func mse --target_class 'subj'
# python main.py --data_use sub-06 --task annot-reg --loss_func mse --target_class 'subj'
# python main.py --data_use sub-08 --task annot-reg --loss_func mse --target_class 'subj'
# python main.py --data_use sub-12 --task annot-reg --loss_func mse --target_class 'subj'
# done