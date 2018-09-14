#! /bin/bash
set -u
set -e

# compile
mkdir -p bin
gcc src/CLSP-SE.c -g -o bin/CLSP-SE -lm -pthread -Ofast -Wall -funroll-loops

# train
source config
train_data=corpus
word_vec=word-vec
time ./bin/CLSP-SE -mono-train1 data/$lang_pair/$train_data.$lang1 -mono-train2 data/$lang_pair/$train_data.$lang2 -lexicon1 data/$lang_pair/seed-lexicon.$lang1 -lexicon2 data/$lang_pair/seed-lexicon.$lang2 -sememe data/sememes.txt -hownet data/hownet.txt -save-sememe output/sememe_vec.txt -output1 output/$lang_pair/$word_vec.$lang1 -output2 output/$lang_pair/$word_vec.$lang2 -save-vocab1 output/$lang_pair/vocab.$lang1 -save-vocab2 output/$lang_pair/vocab.$lang2 -min-count 50 -size 200 -window 5 -sample 1e-5 -negative 10 -threshold 0.5 -epochs 10 -threads 20 -adagrad 0 -sememe-lambda 1 -lexicon-lambda 0.01 -matching-lambda 1000 -alpha 0.1 -cbow 0 

# test
python src/EvalSememePre-SPWE.py ../data/eval_data/ ../output/$lang_pair/ 2000 0
python src/EvalBilingualWordVec.py ../output/$lang_pair/ 5000