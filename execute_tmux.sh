#!/bin/bash

# List of commands to execute
commands=(
    "echo 'Command 1'; sleep 2"
    "echo 'Command 2'; sleep 2"
    "echo 'Command 3'; sleep 2"
)

# Create a new tmux session
tmux new-session -d -s "command_session"

# Loop through the commands and create a new window for each
for ((i = 0; i < ${#commands[@]}; i++)); do
    tmux new-window -t "command_session:$i" "${commands[$i]}"
done

# Attach to the tmux session to view the output
tmux attach-session -t "command_session"
