# Installation Steps:
## All platforms
`uv venv`
This will fail on macos/windows or if you don't have the linux wheels downloaded
Then need to run the corresponding `uv sync` commands as below

## Installing on Linux with ROCM:
To download:
```
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torch-2.9.1%2Brocm7.2.0.lw.git7e1940d4-cp312-cp312-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torchvision-0.24.0%2Brocm7.2.0.gitb919bd0c-cp312-cp312-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/triton-3.5.1%2Brocm7.2.0.gita272dfa8-cp312-cp312-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torchaudio-2.9.0%2Brocm7.2.0.gite3c6ee2b-cp312-cp312-linux_x86_64.whl
```
Download the distribution from `uv sync`


## Installing on macos/windows (don't need ROCM)
`uv sync --no-sources`

## Repo Setup
- Download OpenAI Clip weights from Google Drive: TODO
- Point Effort model to downloaded weights folder
- Download face features extraction model from: TODO