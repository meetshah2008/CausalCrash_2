conda create -n causalcrash -y python==3.12.* nvidia::cuda-toolkit==13.0.*
conda activate causalcrash

pip install -r requirements.txt
MAX_JOBS=4 pip install flash-attn --no-build-isolation
