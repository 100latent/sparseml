# awk script to remove sparsezoo logic from src/sparseml/pytorch/models/registry.py

# Delete the sparsezoo import
/from sparsezoo import Model, model_args_to_stub/ { next }

# Variables to track state
in_create_method = 0
in_create_zoo_model_method = 0
in_registered_wrapper_wrapper_func = 0

# Detect start/end of ModelRegistry.create method
/def create\(/ {
  print
  in_create_method = 1
  next
}
# Assuming the create method ends before another method definition or class level item
/^    @staticmethod/ || /^class / {
  if (in_create_method) {
    in_create_method = 0
  }
}

# Inside ModelRegistry.create method
in_create_method {
  # Skip custom_zoo block
  if (/if pretrained_path.startswith\("custom_zoo:"\):/) {
    while (getline > 0) {
      # Skip until the elif or the main if _checkpoint logic
      if ($0 ~ /elif pretrained_path.startswith\("zoo:"\):/ || $0 ~ /_checkpoint = torch.load\(pretrained_path\)/) {
        print $0 # print the line that terminates the custom_zoo block
        break
      }
    }
    next
  }
  # Skip zoo block
  if (/elif pretrained_path.startswith\("zoo:"\):/) {
     while (getline > 0) {
      # Skip until the main if _checkpoint logic or an else branch
      if ($0 ~ /_checkpoint = torch.load\(pretrained_path\)/ || $0 ~ /else:/ ) {
        print $0 # print the line that terminates the zoo block
        break
      }
    }
    next
  }
  # Print other lines within the create method
  print
  next
}


# Detect start of ModelRegistry.create_zoo_model method to delete it
/def create_zoo_model\(/ {
  in_create_zoo_model_method = 1
  # Skip the method definition line itself
  next
}
# Skip lines while inside create_zoo_model method
in_create_zoo_model_method {
  # Detect end of method (heuristic: starts with another def or class level item, or less indentation)
  if (/^    @staticmethod/ || /^    def / || /^class / || !match($0, /^        /) && !match($0, /^\s*$/) ) {
    in_create_zoo_model_method = 0
    # The current line is NOT part of create_zoo_model, so it should be processed by later rules.
    # Fall through to print it if it's not caught by other rules.
  } else {
    # Still inside the method, so skip this line
    next
  }
}

# Detect start of _registered_wrapper's inner wrapper function
# This is heuristic, might need adjustment based on exact spacing
/def wrapper\(/ {
  # Print the wrapper definition and then start modifying its content
  print
  # Update docstring within wrapper
  while(getline > 0 && ($0 ~ /^            "/ || $0 ~ /^                "/ || $0 ~ /^\s*$/)) {
      if ($0 ~ /A path to the pretrained weights to load/) {
          print "            :param pretrained_path: A path to the pretrained weights to load,"
          print "                if provided will override the pretrained param."
          print "                NOTE: SparseZoo stub paths (e.g. 'zoo:...') are no longer supported." # Updated
      } else if ($0 ~ /a SparseZoo stub path preceded by 'zoo:'/) {
          # Skip this line as zoo stubs are removed
          continue
      } else if ($0 ~ /If given a recipe type, the base/) {
          # Skip this line
          continue
      } else if ($0 ~ /True to load the default pretrained weights/) {
          print "            :param pretrained: True to load the default pretrained weights (if available locally via pretrained_path)," # Updated
          print "                a string to load a specific pretrained weight (if available locally via pretrained_path)," # Updated
          print "                or False to not load any pretrained weights."
          print "                NOTE: Automatic download from SparseZoo is no longer supported." # Updated
      } else {
          print $0
      }
  }
  # After docstring, the line that broke the loop is in $0, process it
  in_registered_wrapper_wrapper_func = 1
  # Fall through to process this current line $0 with the wrapper logic
}

# Inside the _registered_wrapper's inner wrapper function
in_registered_wrapper_wrapper_func {
  # Heuristic for ending the wrapper function (e.g., another def for the outer function, or class end)
  if (/^        return wrapper/ || /^\s*return wrapper/) { # End of _registered_wrapper
      in_registered_wrapper_wrapper_func = 0
      print # print the return statement
      next
  }

  # Remove the call to ModelRegistry.create_zoo_model and related logic
  if (/\s*zoo_model = ModelRegistry.create_zoo_model\(/) {
    # Skip this line and the try-except block that uses zoo_model
    while (getline > 0) {
      if ($0 ~ /except Exception:/) { # look for the except part
         getline # skip the except line
         # skip the content of except (one more line for the second attempt)
         getline
         # The block is now skipped, the next line will be processed by the outer loop
         break # exit this inner while loop
      }
      if ($0 ~ /return model/) { # If the block ends before an except (e.g. if pretrained_path is true)
          print $0 # print the return model
          break
      }
    }
    next
  }

  # Remove direct calls to download_framework_model_by_recipe_type if they exist independently
  # (though they were likely part of the block above)
  if (/\s*path = download_framework_model_by_recipe_type\(zoo_model\)/) {
    next
  }

  # if pretrained_path: load_model(...) should remain
  # if pretrained: ... logic is what needs to be removed if it relies on zoo_model

  # Print other lines within the wrapper function
  print
  next
}

# Default action: print the line if not handled by specific logic above
{ print }
