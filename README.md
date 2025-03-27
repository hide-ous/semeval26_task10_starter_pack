# SemEval26 (proposal) starter pack
Scripts to facilitate participation in the 2026 Semeval Task on Conspiracy Detection and Marker Extraction

--------
## Download plain text data
1. `git clone https://github.com/hide-ous/semeval26_starter_pack.git`
2. `cd semeval26_starter_pack`
3. `pip install -r requirements.txt`
4. place `train_redacted.jsonl` in the folder
5. run `python rehydrate_data.py` to generate `train_rehydrated.jsonl` containing plain texts