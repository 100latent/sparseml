# awk script to remove sparsezoo logic from Recipe.create_instance
BEGIN { printing = 1 }

# Delete the sparsezoo import
/from sparsezoo import Model/ { next }

# Detect start of the section to modify
/if not os.path.isfile\(path_or_modifiers\):/ {
  print
  in_conditional_block = 1
  # Skip the "not a local file" comment if it's there
  getline
  if ($0 ~ /# not a local file/) print
  else print $0 # print current line if it's not the comment
  next
}

# Inside the main conditional block for non-files
in_conditional_block {
  if (/if path_or_modifiers.startswith\("custom_zoo:"\):/) {
    # Start of custom_zoo block, skip lines until its end
    # This assumes custom_zoo block is followed by elif or else
    while (getline > 0) {
      if (/elif path_or_modifiers.startswith\("zoo:"\):/ || /else:  # assume it's a string/) {
        # Found start of next relevant block or the target else
        print $0
        # If it's the target else, reset flag and continue normal printing
        if (/else:  # assume it's a string/) {
            in_conditional_block = 0
        }
        break # Exit this inner while loop
      }
      # else, continue skipping lines of custom_zoo
    }
    next
  }
  if (/elif path_or_modifiers.startswith\("zoo:"\):/) {
    # Start of zoo block, skip lines until its end
    while (getline > 0) {
      if (/else:  # assume it's a string/) {
        # Found the target else
        print $0
        in_conditional_block = 0 # Reset flag
        break # Exit this inner while loop
      }
      # else, continue skipping lines of zoo
    }
    next
  }
  # If we reach here inside in_conditional_block, it means we are in the
  # "else: # assume it's a string" part or something unexpected.
  # If it's the "else: # assume it's a string" line, print it and reset.
  if (/else:  # assume it's a string/) {
    print
    in_conditional_block = 0
    next
  }
  # If still in conditional block but not matching known patterns, print (should not happen with correct logic)
  if(in_conditional_block) print
  next
}

# Default action: print the line
printing { print }
