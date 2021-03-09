txt_file= '../dataset/lines.txt'
gt = ' '
print("Preparing language model...")
# Open raw lines.txt
for line in open(txt_file):
    # Ignore comments
    if not line.startswith("#"):
        # Split each string by whitespaces
        info = line.strip().split()
        # If string was recognized correctly
        if info[1] == 'ok':
            # First column is filename, second column is target sentence
            gt += ' '.join(info[8:]).replace('|', ' ').lower()
text_file = open("../text.txt", "w", encoding='utf-8')
text_file.write(gt)
text_file.close()

# Build KenLM
# mkdir -p build
# cd build
# cmake ..
# make -j 4

# Use this script to generate lm.binary and vocab-500000.txt
# https://github.com/mozilla/DeepSpeech/blob/master/data/lm/generate_lm.py
# python3 generate_lm.py --input_txt text.txt --output_dir . \
#   --top_k 500000 --kenlm_bins path/to/kenlm/build/bin/ \
#   --arpa_order 5 --max_arpa_memory "85%" --arpa_prune "0|0|1" \
#   --binary_a_bits 255 --binary_q_bits 8 --binary_type trie

# Build .scorer
# curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/native_client.amd64.cpu.osx.tar.xz
# tar xvf native_client.*.tar.xz
# ./generate_scorer_package --alphabet ../chars.txt --lm lm.binary --vocab vocab-500000.txt \
#   --package kenlm.scorer --default_alpha 0.931289039105002 --default_beta 1.1834137581510284