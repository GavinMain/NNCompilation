import subprocess
import os
import shutil
from PIL import Image

source = "default-textures/textures"
dest = "image_set"
text_file = "image_text.txt"

repo_url = "https://github.com/KygekDev/default-textures.git"

#Get from GitHub
result = subprocess.run(
    ["git", "clone", repo_url],
    capture_output=True,
    text=True
)

os.makedirs(dest, exist_ok=True)

#Place all png files into 1 folder
for root, dirs, files in os.walk(source):
    for file in files:
        if file.lower().endswith(".png"):
            file_path = os.path.join(root, file)
            
            try:
                with Image.open(file_path) as img, \
                    open(text_file, "a") as f:
                        if img.size == (16, 16):
                            dest_path = os.path.join(dest, file)
                            shutil.copy2(file_path, dest_path)
                            
                            #Text file for tokenizer
                            f.write(f"{file.split('.png')[0].replace('_', ' ')}\n")
                
            except Exception as e:
                pass
        

shutil.rmtree("default-textures")



