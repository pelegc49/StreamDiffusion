from huggingface_hub import snapshot_download

# Download the model
model_path = snapshot_download(
    "KBlueLeaf/kohaku-v2.1",
    local_dir="models/Model/kohaku-v2.1",
    local_dir_use_symlinks=False
)
# model_path = snapshot_download(
#     "mirroring/pastel-mix",
#     local_dir="models/Model/pastel-mix",
#     local_dir_use_symlinks=False
# )
# model_path = snapshot_download(
#     "black-forest-labs/FLUX.1-dev",
#     local_dir="models/Model/FLUX.1-dev",
#     local_dir_use_symlinks=False
# )
print(f"Model downloaded to: {model_path}")
