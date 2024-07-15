# download the orignal model from hugging face using this code

import os
from huggingface_hub import snapshot_download

def download_model(model_name, base_model_dir):
    if not os.path.exists(base_model_dir):
        os.makedirs(base_model_dir)
    snapshot_download(repo_id=model_name, local_dir=base_model_dir, local_dir_use_symlinks=False)
    print(f"Model {model_name} has been downloaded to {base_model_dir}")

if __name__ == "__main__":
    model_name = "BAAI/bge-large-en-v1.5"
    base_model_dir = "./models/original"
    download_model(model_name, base_model_dir)
