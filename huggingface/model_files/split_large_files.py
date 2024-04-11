from pathlib import Path
import math
import zipfile
import os
import json  

# Used to name files such that they are not seems as part_01, part_02 ...
# Requires a mapping from dummy_file_name to real file name
dummy_file_names = [
  'encoder_self_attention_weight',
  'encoder_feed_forward_network',
  'decoder_feed_forward_network',
  'positional_encoding_layer',
  'scaled_dot_product_attention',
  'feed_forward_output_network',
  'decoder_attention_network',
  'multi_head_attention_vector',
  'encoder_decoder_attention_score',
  'decoder_dropout_layer',
  'transformer_model',
  'layer_normalization_vector',
  'encoder_decoder_attention_weight',
  'encoder_add_norm_layer',
  'padding_vector',
  'attention_network',
  'linear_transformation_network',
  'encoder_softmax_activation',
  'encoder_relu_activation',
  'encoder_attention_network',
  'encoder_decoder_vector',
  'softmax_activation_layer',
  'block_output_network',
  'dropout_layer',
  'layer_normalization',
  'decoder_self_vector',
  'encoder_residual_connection',
  'masking_layer',
  'positional_encoding_network',
  'encoder_output_layer',
  'multi_head_attention_layer',
  'feed_forward_network_layer',
  'position_wise_network',
  'relu_activation_vector',
  'add_norm_layer',
  'transformer_encoder',
  'fully_connected_network',
  'input_layer',
  'decoder_self_attention_output',
  'decoder_masking_layer',
  'self_attention_vector',
  'attention_layer',
  'encoder_output_vector',
  'attention_weight',
  'configuration_network',
  'transformer_model_parameters',
  'value_vector',
  'attention_mask_network',
  'position_wise_feed_forward_networks',
  'transformer_encoder_output',
  'decoder_input_layer',
  'encoder_self_network',
  'feed_forward_network_output',
  'decoder_self_attention_layer',
  'encoder_layer_normalization',
  'encoder_self_attention_score',
  'relu_activation_layer',
  'decoder_block',
  'block_input_vector',
  'transformer_configuration',
  'decoder_output_layer',
  'add_norm_vector',
  'transformer_block_input',
  'value_network',
  'decoder_position_wise_feed_forward_network',
  'transformer_block',
  'query_vector',
  'transformer_decoder_output',
  'encoder_padding_layer',
  'decoder_padding_layer',
  'decoder_self_attention_weight',
  'encoder_position_wise_feed_forward_network',
  'output_layer',
  'encoder_linear_transformation',
  'normalization_layer',
  'multi_head_output_vector',
  'attention_score_vector',
  'self_attention_mechanism',
  'decoder_add_norm_layer',
  'encoder_self_attention_output',
  'decoder_positional_encoding',
  'encoder_dropout_layer',
  'attention_mask',
  'encoder_input_layer',
  'encoder_decoder_attention_layer',
  'dropout_vector',
  'query_network',
  'encoder_self_attention_layer',
  'encoder_masking_layer',
  'decoder_softmax_activation',
  'padding_layer',
  'masking_network',
  'key_vector',
  'decoder_residual_connection',
  'decoder_output_network',
  'residual_connection_layer',
  'multi_head_attention_output',
  'attention_weight_vector',
  'encoder_attention_vector',
  'transformer_decoder',
  'normalization_network',
  'transformer_block_output',
  'encoder_decoder_attention_output',
  'decoder_layer_normalization',
  'decoder_relu_activation',
  'attention_score',
  'residual_vector',
  'linear_transformation_layer',
  'encoder_block',
  'residual_connection_network',
  'encoder_positional_encoding',
  'positional_encoding_vector',
  'softmax_activation_network',
  'model_parameters_vector',
  'decoder_self_attention_score',
  'fully_connected_layer',
  'decoder_linear_transformation',
  'embedding_layer'
]  


def split_binary_file(input_file_path, chunk_size_mb=250, chunk_suffix=None, max_folder_size_mb=10000000):  
    print(f'Enter split_binary_file: {input_file_path}, chunk_size_mb={chunk_size_mb}')
    chunk_size = chunk_size_mb * 1024 * 1024      
    input_file_path = Path(input_file_path)  
    output_base_folder = input_file_path.parent / 'output_chunks'    
    output_base_folder.mkdir(parents=True, exist_ok=True)  
  
    # Get the size of the input file  
    file_size = input_file_path.stat().st_size  
  
    # Calculate the number of chunks needed  
    num_chunks = math.ceil(file_size / chunk_size)  
        
    # Open the input file for reading  
    with input_file_path.open('rb') as input_file:  
        
        # Iterate over each chunk in the folder  
        for i in range(num_chunks):  
            # Read a chunk of data  
            chunk_data = input_file.read(chunk_size)  
  
            # Select file name and ext for the chunk - if chunk_suffix == None - use the original file ext
            if chunk_suffix is None:  
                chunk_suffix = input_file_path.suffix[1:]  
              
            output_file_path = output_base_folder / f"{input_file_path.stem}_chunk_{i + 1}.{chunk_suffix}"  
            with output_file_path.open('wb') as output_file:  
                # Write the chunk data to the new file  
                output_file.write(chunk_data)  
                          
            print(f"Chunk {i + 1}/{num_chunks} created in {output_base_folder}: {output_file_path}") 
            
  
def rename_files_create_mapping(input_folder_path):  
    """
    Rename model_01_of_03_chunk_4.safetensors --> transformer_block_output.safetensors
    Write the dictionary file name mapping to a JSON file
    
    Parameters
    ----------
    input_folder_path : path to models folder - assume 'output_chunks' below this folder with 
    Returns
    -------
    None.

    """
    print(f'Enter rename_files_create_mapping: {input_folder_path}')
    input_folder_path = Path(input_folder_path)    
    # Initialize the dictionary to store the mapping  
    dummy_name_dict = {}  
    for i, input_file_path in enumerate(input_folder_path.glob("*")):        
        # Iterate all files, select file name from dummy_file_names
        dummy_file_name = dummy_file_names[i % len(dummy_file_names)]  
    
        # Rename file and update dict 
        os.rename(input_file_path, input_folder_path / f'{dummy_file_name}{input_file_path.suffix}')        
        dummy_name_dict[dummy_file_name] = f"{input_file_path.stem}_chunk_{i + 1}{input_file_path.suffix}"                                  
    # Write the dictionary to a JSON file  
    with open(input_folder_path / 'nm.json', 'w') as f:  
        json.dump(dummy_name_dict, f)  


def rename_files_to_original(input_folder_path):    
    """
    The inverse transform of rename_files_create_mapping.
    Load the dictionary file name mapping to a JSON file
    Rename transformer_block_output.safetensors --> model_01_of_03_chunk_4.safetensors    
    
    Parameters
    ----------
    input_folder_path : path to models folder - assume 'output_chunks' below this folder with 
    Returns
    -------
    None.

    """
    print(f'Enter rename_files_to_original: {input_folder_path}')
    input_folder_path = Path(input_folder_path)
    

    # Load the mapping from the JSON file
    with open(input_folder_path / 'nm.json', 'r') as f:
        dummy_name_dict = json.load(f)

    
    for dummy_file_path in input_folder_path.glob("*"):
        # Get the dummy file name without the suffix
        dummy_file_name = dummy_file_path.stem

        # Check if the dummy file name exists in the mapping
        if dummy_file_name in dummy_name_dict:
            original_file_name = dummy_name_dict.get(dummy_file_name, None)
            if not original_file_name is None:
                original_file_path = input_folder_path / f'{original_file_name}{dummy_file_path.suffix}'

            # Rename the file to its original name
            os.rename(dummy_file_path, original_file_path)
            print(f"Renamed: {dummy_file_path} -> {original_file_path}")
        else:
            print(f"Skipped: {dummy_file_path} (not found in mapping)")

    print("End rename_files_to_original - completed")
    
def split_folder_of_large_files(folder_path, chunk_size_mb=250, chunk_suffix=None, max_folder_size_mb=1000000):
    folder_path = Path(folder_path)

    # Iterate over each file in the folder
    for file_path in folder_path.glob("*"):
        # Check if the file is a regular file and its size exceeds the chunk size
        if file_path.is_file() and file_path.stat().st_size > chunk_size_mb * 1024 * 1024:
            split_binary_file(file_path, chunk_size_mb, chunk_suffix, max_folder_size_mb)
            
def concat_chunks(input_folder, output_file_path = None, output_fname_suffix = '.safetensors', chunk_suffix='.txt'):  
    input_folder = Path(input_folder)  
  
    # Get a list of all files in the input folder with the specified suffix  
    if chunk_suffix is None:  
        chunk_suffix = ""  
    chunk_files = sorted(input_folder.rglob(f"*{chunk_suffix}"))  
  
    # If output_file_path is None, use the stem of the first chunk file  
    if output_file_path is None:  
        if chunk_files:  
            base_name = chunk_files[0].stem.rsplit('_chunk_', 1)[0]  
            output_file_path = input_folder / f"{base_name}{output_fname_suffix}"  
        else:  
            raise ValueError("No chunk files found")  
    else:  
        output_file_path = Path(output_file_path)  
  
    # Open the output file for writing  
    with output_file_path.open('wb') as output_file:  
        # Concatenate the content of each chunk file  
        for chunk_file in chunk_files:  
            with chunk_file.open('rb') as chunk:  
                output_file.write(chunk.read())  
            print(f"Added chunk file {chunk_file}")  
  
    print(f"Original file generated: {output_file_path}")  



from concurrent.futures import ThreadPoolExecutor, wait   
import zipfile  
from pathlib import Path  
  
def zip_file(file, large_file_size_bytes, folder_path):  
    print(f"zip_files: zipping {file}")      
    if file.stat().st_size > large_file_size_bytes:      
        with zipfile.ZipFile(f"{file.absolute()}.zip", 'w', zipfile.ZIP_DEFLATED, compresslevel=3) as large_file_zip:      
            large_file_zip.write(file, file.name)      
    else:    
        with zipfile.ZipFile(folder_path / 'small_files.zip', 'a', zipfile.ZIP_DEFLATED, compresslevel=3) as small_files_zip:    
            small_files_zip.write(file, file.name)    
  
def zip_files(folder_path, glob_pat='*/chunks_*/*_chunk_*.txt' , large_file_size_mb = 50):        
    folder_path = Path(folder_path)        
    large_file_size_bytes = large_file_size_mb * 1024 * 1024  # Convert MB to bytes    
    futures = []  
    
    with ThreadPoolExecutor() as executor:    
        lst_paths = list(folder_path.rglob(glob_pat))
        print(f'Started zipping of files {lst_paths}\n-----------------------------\n')  
        for file in lst_paths:  
            if file.is_file():    
                future = executor.submit(zip_file, file, large_file_size_bytes, folder_path)  
                futures.append(future)  
                    
    wait(futures)  # Wait for all futures to complete  
    print(f"All files zipped successfully in {folder_path}")
    
    

    
if __name__ == "__main__":
    # Small test
    #input_file_path = r"D:\Halb\Dec-20\wmt22-cometkiwi-da\checkpoints\model.ckpt"
    input_folder_path = Path(r"D:\Halb\Dec-20\wmt22-cometkiwi-da\checkpoints")
    chunk_size_mb=3
    
    
    # Cmd-r
    input_folder_path = '/data2/NLP/LLMs/35B/Command-R/command-r-v01-35B-exl2' # Path(r"D:\Halb\Dec-20\wmt22-cometkiwi-da\checkpoints")
    chunk_size_mb=950
    #input_file_path = '/data2/NLP/LLMs/35B/Command-R/model-00001-of-00015.safetensors'
    
    split_folder_of_large_files(input_folder_path, chunk_size_mb=chunk_size_mb, chunk_suffix=None)
    rename_files_create_mapping(input_folder_path / 'output_chunks')
    
    # Restore files
    rename_files_to_original(input_folder_path / 'output_chunks')
    
    
    if False:
        split_binary_file(input_file_path, chunk_size_mb=950, max_folder_size_mb=100000, chunk_suffix='txt')
        
        
    
    
        # Test 
        input_folder_path =  '/data2/NLP/LLMs/35B/Command-R/model-00001-of-00015_chunks' # r"D:\Halb\Dec-20\wmt22-cometkiwi-da\checkpoints\chunks"
        output_file_path = '/data2/NLP/LLMs/35B/Command-R/concat/model-00001-of-00015.safetensors' # r"D:\Halb\Dec-20\wmt22-cometkiwi-da\checkpoints\model_cat.ckpt"
        concat_chunks(input_folder_path, output_file_path = None, chunk_suffix = '.txt')