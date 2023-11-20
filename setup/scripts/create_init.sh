#!/bin/bash

# Path to the root of your Python project
PROJECT_ROOT="src"

# Function to recursively create __init__.py files
create_init_files() {
    for dir in "$1"/*/; do
        if [ -d "$dir" ]; then
            INIT_FILE="${dir}__init__.py"
            if [ ! -f "$INIT_FILE" ]; then
                touch "$INIT_FILE"
                echo "Created __init__.py in $dir"
            fi
            create_init_files "$dir"
        fi
    done
}

# Start the process from the project root
create_init_files "$PROJECT_ROOT"
