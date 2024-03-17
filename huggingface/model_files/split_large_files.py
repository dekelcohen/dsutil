from pathlib import Path
import math
import zipfile
import os


def split_binary_file(input_file_path, chunk_size_mb=250, chunk_suffix=None, max_folder_size_mb=10000000):
    chunk_size = chunk_size_mb * 1024 * 1024
    max_folder_size = max_folder_size_mb * 1024 * 1024
    input_file_path = Path(input_file_path)
    output_base_folder = input_file_path.parent / f'{input_file_path.stem}_chunks'  
    output_base_folder.mkdir(parents=True, exist_ok=True)

    # Get the size of the input file
    file_size = input_file_path.stat().st_size

    # Calculate the number of chunks needed
    num_chunks = math.ceil(file_size / chunk_size)

    # Calculate the number of folders needed
    num_folders = math.ceil(file_size / max_folder_size)

    # Iterate over each folder
    for folder_num in range(num_folders):
        current_folder_size = 0
        current_folder = output_base_folder / f'chunks_{folder_num + 1}'
        current_folder.mkdir(parents=True, exist_ok=True)
    
        # Open the input file for reading
        with input_file_path.open('rb') as input_file:
            # Seek to the start position for the current folder
            input_file.seek(folder_num * num_chunks // num_folders * chunk_size)
    
            # Iterate over each chunk in the folder
            for i in range(folder_num * num_chunks // num_folders, (folder_num + 1) * num_chunks // num_folders):
                # Read a chunk of data
                chunk_data = input_file.read(chunk_size)
    
                # Create a new file for the chunk
                if chunk_suffix is None:
                    chunk_suffix = input_file_path.suffix[1:]
                output_file_path = current_folder / f"{input_file_path.stem}_chunk_{i + 1}.{chunk_suffix}"
                with output_file_path.open('wb') as output_file:
                    # Write the chunk data to the new file
                    output_file.write(chunk_data)
    
                current_folder_size += len(chunk_data)
                print(f"Chunk {i + 1}/{num_chunks} created in {current_folder}: {output_file_path}")
    
                # Check if the folder size exceeds the maximum allowed size
                if current_folder_size >= max_folder_size:
                    break


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



from concurrent.futures import ThreadPoolExecutor  
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
  
def zip_files(folder_path, glob_pat='*/chunks_*/model-*_chunk_*.txt' , large_file_size_mb = 50):      
    folder_path = Path(folder_path)      
    large_file_size_bytes = large_file_size_mb * 1024 * 1024  # Convert MB to bytes  
  
    with ThreadPoolExecutor() as executor:  
        for file in folder_path.rglob(glob_pat):      
            if file.is_file():  
                executor.submit(zip_file, file, large_file_size_bytes, folder_path)  
                  
    print(f"All files zipped successfully in {folder_path}")    

# OLD: Delete     
def zip_files(folder_path, glob_pat='*/chunks_*/model-*_chunk_*.txt' , large_file_size_mb = 50):    
    folder_path = Path(folder_path)    
    large_file_size_bytes = large_file_size_mb * 1024 * 1024  # Convert MB to bytes    
      
    for file in folder_path.rglob(glob_pat):    
        if file.is_file():    
            print(f"zip_files: zipping {file}")    
            if file.stat().st_size > large_file_size_bytes:    
                with zipfile.ZipFile(f"{file.absolute()}.zip", 'w', zipfile.ZIP_DEFLATED, compresslevel=3) as large_file_zip:    
                    large_file_zip.write(file, file.name)    
            else:  
                with zipfile.ZipFile(folder_path / 'small_files.zip', 'a', zipfile.ZIP_DEFLATED, compresslevel=3) as small_files_zip:  
                    small_files_zip.write(file, file.name)    
    print(f"All files zipped successfully in {folder_path}")  
  

    

    
if __name__ == "__main__":
    # input_file_path = r"D:\Halb\Dec-20\wmt22-cometkiwi-da\checkpoints\model.ckpt"
    input_file_path = '/data2/NLP/LLMs/35B/Command-R/model-00001-of-00015.safetensors'
    
    input_folder_path = '/data2/NLP/LLMs/35B/Command-R/' # Path(r"D:\Halb\Dec-20\wmt22-cometkiwi-da\checkpoints")
    
    split_folder_of_large_files(input_folder_path, chunk_size_mb=950, chunk_suffix='txt')
    
    large_file_size_mb = 50
    zip_files('/data2/NLP/LLMs/35B/Command-R', glob_pat = '*/chunks_*/model-*_chunk_*.txt', large_file_size_mb=large_file_size_mb)
    
    if False:
        split_binary_file(input_file_path, chunk_size_mb=950, max_folder_size_mb=100000, chunk_suffix='txt')
        
        
    
    
        # Test 
        input_folder_path =  '/data2/NLP/LLMs/35B/Command-R/model-00001-of-00015_chunks' # r"D:\Halb\Dec-20\wmt22-cometkiwi-da\checkpoints\chunks"
        output_file_path = '/data2/NLP/LLMs/35B/Command-R/concat/model-00001-of-00015.safetensors' # r"D:\Halb\Dec-20\wmt22-cometkiwi-da\checkpoints\model_cat.ckpt"
        concat_chunks(input_folder_path, output_file_path = None, chunk_suffix = '.txt')