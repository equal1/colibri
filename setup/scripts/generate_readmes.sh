#!/bin/bash

# Small script to generate dummy README.md files for all directories in the repo.
# This is useful to allow empty directories to be tracked by git.
# At the moment the script dosn't generate READMEs for subdirectories.


# Function to extract directory description from README.md
extract_description() {
    folder=$1
    awk -v folder="$folder" '
    BEGIN {found=0}
    {
        if ($0 ~ folder ".*") {
            found = 1
        }
        if (found && $0 ~ "<-") {
            gsub("^.*<- ", "")
            print
            exit
        }
    }
    ' $2
}

# Recursive function to generate READMEs
generate_readme_recursive() {
    local parent_dir=$1
    local readme_path=$2

    # Extract directories from README.md(only first level)
    directories=$(awk -F' ' '/^├── /{gsub("^├── ", ""); gsub(" .*", ""); print}' $readme_path)

    echo "Directories found: $directories"

    # Generate README.md for each directory
    for dir in $directories; do
        full_dir_path="$parent_dir/$dir"

        # Skip if it's not a directory
        if [[ ! -d $full_dir_path ]]; then
            continue
        fi

        # Skip if README.md already exists in directory
        if [[ -f "$full_dir_path/README.md" ]]; then
            echo "Skipping $full_dir_path, README.md already exists."
            continue
        fi

        # Extract directory description from parent README.md
        description=$(extract_description $dir $readme_path)

        # Create README.md in directory
        echo "# $dir" > "$full_dir_path/README.md"
        echo "## Description" >> "$full_dir_path/README.md"
        echo "$description" >> "$full_dir_path/README.md"
        echo -e "\n---\n" >> "$full_dir_path/README.md"
        echo "Note: This README is a dummy file for now with the scope of allowing empty folders in the repo." >> "$full_dir_path/README.md"

        # Recursively generate READMEs for subdirectories
        generate_readme_recursive $full_dir_path "$full_dir_path/README.md"

    done
}

# Check if README.md exists in the root directory
if [[ ! -f README.md ]]; then
    echo "README.md not found in the current directory. Exiting."
    exit 1
fi

# Start generating READMEs from the root directory
generate_readme_recursive "." "./README.md"

echo "README files generated successfully."
