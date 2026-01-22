import os
import sys
from huggingface_hub import hf_hub_download
# Download the GPT-2 tokens of Fineweb10B from huggingface. This
# saves about an hour of startup time compared to regenerating them.
#
# Set DATA_PATH env var to store data in a different location (e.g., persistent volume)
# DATA_PATH should point to the repo root or a persistent volume
# Data will be stored in DATA_PATH/data/fineweb10B/ to match train_gpt.py expectations
data_path = os.environ.get("DATA_PATH")
if data_path:
    local_dir = os.path.join(data_path, 'data', 'fineweb10B')
else:
    local_dir = os.path.join(os.path.dirname(__file__), 'fineweb10B')

def get(fname):
    global local_dir
    if not os.path.exists(os.path.join(local_dir, fname)):
        hf_hub_download(repo_id="kjj0/fineweb10B-gpt2", filename=fname,
                        repo_type="dataset", local_dir=local_dir)
get("fineweb_val_%06d.bin" % 0)
num_chunks = 103 # full fineweb10B. Each chunk is 100M tokens
if len(sys.argv) >= 2: # we can pass an argument to download less
    num_chunks = int(sys.argv[1])
for i in range(1, num_chunks+1):
    get("fineweb_train_%06d.bin" % i)
