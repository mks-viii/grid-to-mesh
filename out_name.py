import os

def get_output_name(input_name):
    # Ottieni il nome del file senza il percorso
    base_name = os.path.basename(input_name)
    
    # Separa il nome del file dalla sua estensione
    name_without_extension = os.path.splitext(base_name)[0]
    
    # Aggiungi la nuova estensione
    output_name = f"{name_without_extension}.xyz"
    
    return output_name
