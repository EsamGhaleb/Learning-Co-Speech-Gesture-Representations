#!/bin/bash

# Check if data path is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <data_path>"
    exit 1
fi

data_path="$1"

# List of probing properties
probing_properties=("handedness" "position" "shape" "movement" "rotation")

# Loop through each probing property
for property in "${probing_properties[@]}"
do
    echo "Running probing script for property: $property"
    python probing_script.py --data_path "$data_path" --probing_property "$property"
    echo "Completed probing script for property: $property"
done