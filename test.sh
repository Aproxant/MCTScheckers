#!/bin/bash

run_checker() {
    output=$(python3 checkers.py --skip_menu | tail -n 1)

    if [[ $output == "RED WINS!" ]]
    then
        echo "red"
    elif [[ $output == "BLUE WINS!" ]]
    then
        echo "blue"
    elif [[ $output == "DRAW!" ]]
    then
        echo "draw"
    else
        echo "ERROR: $output"
    fi
}

export -f run_checker

results=$(parallel -j 10 run_checker ::: {1..100})

red_wins=$(echo "$results" | grep -c "red")
blue_wins=$(echo "$results" | grep -c "blue")
draws=$(echo "$results" | grep -c "draw")

echo "Total Red Wins: $red_wins"
echo "Total Blue Wins: $blue_wins"
echo "Total Draws: $draws"
