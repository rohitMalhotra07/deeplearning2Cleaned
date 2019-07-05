 ## Installation
pip install -r requirenments.txt




## Train and Evaluate DQN
python drl.py --scenario deathmatch --wad deathmatch_rockets --n_bots 8 --action_combinations "move_fb;move_lr;turn_lr;attack"  --game_features "enemy" --network_type dqn_ff --gpu_id 0 --height 42 --width 42


## Train and Evaluate DRQN with augmented features
python drl.py --scenario deathmatch --wad deathmatch_rockets --n_bots 8 --action_combinations "move_fb;move_lr;turn_lr;attack" --game_features "enemy" --network_type dqn_rnn --recurrence lstm --n_rec_updates 5 --gpu_id 0 --height 42 --width 42


## Train and Evaluate A3C

python drl.py --scenario deathmatch --wad deathmatch_rockets --n_bots 8  --visualize 1 --gpu_id -1 --num_agents_a3c 4 --height 42 --width 42 --is_a3c 1




## Acknowledgements
https://github.com/glample/Arnold