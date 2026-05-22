#!/bin/bash
OBJ_FOLDER="./input_models"
OUTPUT_FOLDER="./output_renders"

# Recursively find all .glb and .obj files
find "$OBJ_FOLDER" -type f \( -name "*.glb" -o -name "*.obj" \) | while read -r obj_file; do
    

    echo "Processing $obj_file..."

    obj_dir="$(dirname "$obj_file")"
    subject_width="$obj_dir/subject.json"
    relative_path="${obj_file#$OBJ_FOLDER/}"
    base_name="$(dirname "$relative_path")"
    out_folder="$OUTPUT_FOLDER/$base_name"
    #base_name=$(basename "$obj_file" .glb)
    #out_folder="$OUTPUT_FOLDER/$base_name"
    # Skip if already fully processed
    if [ -d "$out_folder" ]; then
        if [ -f "$out_folder/meta.npy" ]; then
            if [ -d "$out_folder/camera" ] && [ -d "$out_folder/image" ] && [ -d "$out_folder/normal" ]; then
                camera_count=$(find "$out_folder/camera" -type f | wc -l)
                image_count=$(find "$out_folder/image" -type f | wc -l)
                normal_count=$(find "$out_folder/normal" -type f | wc -l)

                if [ "$camera_count" -eq 16 ] && [ "$image_count" -eq 16 ] && [ "$normal_count" -eq 16 ]; then
                    echo "Already processed: $base_name, skipping."
                    continue
                fi
            fi
        fi
    fi


    python blender_scripts_obj.py -- --object_path "$obj_file" --output_dir "$out_folder" --subject_width "$subject_width"
done
