#!/bin/bash

function usage
{
    cat << HEREDOC
    Usage: bash run_attack.sh [attack_type]
    Argument:
        attack_type: the algorithm to perform adversarial attack.
        Current supported:
            FGSM
HEREDOC
}

ATT_TYPE=$1

if [[ $ATT_TYPE == "fgsm" ]] || [[ $ATT_TYPE == "FGSM" ]]; then
cd ../src
for epsilon in 0.1 0.2 0.5 1 2 5 10; do
    python attacker.py --ckpt_path checkpoints/best.std --use_bert n --epsilon $epsilon
done
cd ../scripts
else
    echo "Unrecognized attack type, valid types are {fgsm,}"
    exit
fi