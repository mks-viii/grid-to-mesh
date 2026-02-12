import os

def get_output_name(input_name):
    # Extract only the file's name
    base_name = os.path.basename(input_name)
    
    # Split name from its extension
    name_without_extension = os.path.splitext(base_name)[0]
    
    # Add the new extension
    output_name = f"{name_without_extension}.xyz"
    
    return output_name
