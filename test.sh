#!/bin/bash

# Define a function that runs the Python script and increments the counters
run_checker() {
    # Launch the checker.py program
    output=$(python3 checkers.py --skip_menu | tail -n 1)

    # Check the output and increment the appropriate counter
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

# Use GNU parallel to run the function 100 times
results=$(parallel -j 10 run_checker ::: {1..100})

# Count the red and blue wins
red_wins=$(echo "$results" | grep -c "red")
blue_wins=$(echo "$results" | grep -c "blue")
draws=$(echo "$results" | grep -c "draw")

# Print the statistics
echo "Total Red Wins: $red_wins"
echo "Total Blue Wins: $blue_wins"
echo "Total Draws: $draws"
