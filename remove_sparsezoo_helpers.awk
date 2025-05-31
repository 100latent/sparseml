# awk script to remove sparsezoo logic from src/sparseml/optim/helpers.py
BEGIN { printing = 1 }

# Delete the sparsezoo import
/from sparsezoo import File, Model/ { next }

# Variable to track if we are inside the target function
in_load_yaml_str_from_file_func = 0

# Detect start of the _load_yaml_str_from_file function
/_load_yaml_str_from_file\(file_path: Union\[str, File\]\) -> str:/ {
  print
  in_load_yaml_str_from_file_func = 1
  next
}

# Logic when inside the _load_yaml_str_from_file function
in_load_yaml_str_from_file_func {
  # Skip the block for `isinstance(file_path, File)`
  if (/if isinstance\(file_path, File\):/) {
    # Skip this line and the next line (file_path = file_path.path)
    getline
    next
  }

  # Skip the block for `file_path.startswith("custom_zoo:")`
  if (/if file_path.startswith\("custom_zoo:"\):/) {
    # Skip lines until the `elif file_path.startswith("zoo:")` or the `elif "zoo:" in file_path` (older version) or the main `elif "\\n" in file_path`
    while (getline > 0) {
      if ($0 ~ /elif file_path.startswith\("zoo:"\):/ || $0 ~ /elif "zoo:" in file_path/ || $0 ~ /elif "\\n" in file_path || $0 ~ /if "\\n" in file_path/) {
        # We've reached the next significant block. Print this line and break from inner loop.
        # However, this line itself might be a sparsezoo related line we want to skip.
        # So, we re-evaluate it in the next iteration of the outer loop.
        recheck_line = $0
        $0 = "" # Clear current line to avoid printing if it's a sparsezoo line
        # Prepend recheck_line to next input (awk specific trick or use a flag)
        # A simpler way: just let the next top-level if catch it.
        print recheck_line # Print the line that ended the skip
        break
      }
      # If not breaking, this line is part of custom_zoo block, so skip.
    }
    next
  }

  # Skip the block for `file_path.startswith("zoo:")` or `"zoo:" in file_path`
  if (/elif file_path.startswith\("zoo:"\):/ || /if file_path.startswith\("zoo:"\):/ || /elif "zoo:" in file_path/) {
    # Skip lines until the main `elif "\\n" in file_path` (or `if "\\n" ...` if other blocks removed)
    while (getline > 0) {
      if ($0 ~ /elif "\\n" in file_path/ || $0 ~ /if "\\n" in file_path/ || $0 ~ /# load the yaml string/) {
         print $0 # Print the line that ended the skip
         break
      }
      # If not breaking, this line is part of zoo block, so skip.
    }
    next
  }

  # If the line contains an explicit call to sparsezoo Model, skip it
  # This is a safeguard for any remaining direct calls if the block removal was incomplete.
  if (/\s*model = Model\(file_path\)/) {
      next
  }
  if (/file_path = model.recipes.default.path/) {
      next
  }


  # If we are past the blocks to remove, print the line
  # and if it's the end of the function's typical logic before file extension checks,
  # reset the flag (though for this script, simply printing and proceeding is fine).
  print
  next
}

# Default action: print the line if not handled by specific logic above
{ print }
