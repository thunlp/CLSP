//  Copyright 2016 Meng Zhang
//  Copyright 2015 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SEN_LEN 1000

#define NUM_LANG 2
#define CLIP_UPDATES 0.1               // biggest update per parameter per step

#define MAX_LEXICON_SIZE 10000
#define MAX_VOCAB_SIZE 21000000

#define MAX_SEMEME_SIZE 2500
#define MAX_HOWNET_SIZE 150000

#define HOWNET_HASH_SIZE 180000 // HowNet有12万词，12/0.7≈18万

const int vocab_hash_size = 30000000; // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

struct vocab_word {
	long long cn;
	int *point;
	char *word;
};

// -----------------   new begin
struct sememe_word {
	char * word;
};
struct sememe_word sememes[MAX_SEMEME_SIZE];

struct hownet_word {
	char *word;
	int sememe_num; // 义原数量
	int *sememe_idx; // 义原编号
};
struct hownet_word hownet[MAX_HOWNET_SIZE];


char *sememe_file, *hownet_file, *save_sememe_file;

int sememe_size = 0, hownet_size = 0;
int hownet_hash[HOWNET_HASH_SIZE];

real *sememe_vec1,
     *sememe_vec2,
     *word_bias,
     *sememe_bias,
     *sememe_vec_ada1,
     *sememe_vec_ada2,
     *word_bias_ada,
     *sememe_bias_ada; // AdaGrad是否需要全局变量？

real SEMEME_LAMBDA = 1;

// -----------------   new end

char *mono_train_files[NUM_LANG], *lexicon_files[NUM_LANG],
     *output_files[NUM_LANG], *save_vocab_files[NUM_LANG],
     *read_vocab_files[NUM_LANG];

struct vocab_word *vocabs[NUM_LANG];

int binary = 0, cbow = 0, debug_mode = 2, window = 5, min_count = 5,
    num_threads = 1, min_reduce = 1;

int *vocab_hashes[NUM_LANG];
long long vocab_max_size = 1000, vocab_sizes[NUM_LANG], layer1_size = 40;
long long lexicons[NUM_LANG][MAX_LEXICON_SIZE], lexicon_size;
//Assume lang_id1 corresponds to input/source language
char srcVocabInLexicon[MAX_VOCAB_SIZE];
char tgtVocabInLexicon[MAX_VOCAB_SIZE];
long long train_words[NUM_LANG], word_count_actual = 0, file_sizes[NUM_LANG];
long long lang_updates[NUM_LANG], dump_every = 0, dump_iters[NUM_LANG],
                                  epoch[NUM_LANG];
unsigned long long next_random = 0;
int learn_vocab_and_quit = 0, adagrad = 1;
real alpha = 0.025, starting_alpha, sample = 0, bilbowa_grad = 0;
real *syn0s[NUM_LANG], 	//Zm: input vectors
     *syn1s[NUM_LANG], 	//Zm: not used
     *syn1negs[NUM_LANG],	//Zm: output vectors
     *syn0grads[NUM_LANG], *syn1negGrads[NUM_LANG],	//Zm: only used in AdaGrad
     *expTable,
     *sigmoidTable;	//Zm: a look-up table for the logistic sigmoid function
clock_t start;

const int table_size = 1e8;     // const across languages
int *tables[NUM_LANG];
int negative = 5, MONO_DONE_TRAINING = 0;
//Zm: using MONO_DONE_TRAINING as termination criterion can be unreliable
char ALL_MONO_DONE = 0;
pthread_rwlock_t lock;
int MSTEP_ITER = 1;
long long NUM_EPOCHS = 1, EARLY_STOP = 0, max_train_words;
real *delta_pos, MATCHING_LAMBDA = 1, LEXICON_LAMBDA = 1, threshold = 0;

void InitUnigramTable(int lang_id) {
	int a, i;
	long long train_words_pow = 0, vocab_size = vocab_sizes[lang_id];
	int *table;
	struct vocab_word *vocab = vocabs[lang_id];
	real d1, power = 0.75;

	table = tables[lang_id] = malloc(table_size * sizeof(int));
	for (a = 0; a < vocab_size; a++)
		train_words_pow += pow(vocab[a].cn, power);
	i = 0;
	d1 = pow(vocab[i].cn, power) / (real) train_words_pow;
	for (a = 0; a < table_size; a++) {
		table[a] = i;
		if (a / (real) table_size > d1) {
			i++;
			d1 += pow(vocab[i].cn, power) / (real) train_words_pow;
		}
		if (i >= vocab_size)
			i = vocab_size - 1;
	}
}

/* Reads a single word from a file, assuming space + tab + EOL to be word
 boundaries */
void ReadWord(char *word, FILE *fin) {
	int a = 0, ch;
	while (!feof(fin)) {
		ch = fgetc(fin);
		if (ch == 13) // 回车符CR
			continue;
		if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
			if (a > 0) {
				if (ch == '\n')
					ungetc(ch, fin); // 将换行符退回，下一次再读
				break; // 已经读了若干字符，遇到空格或者tab则停止
			}
			if (ch == '\n') { // 第一个字符就是换行符
				strcpy(word, (char *) "</s>");
				return;
			} else
				continue;
		}
		word[a] = ch;
		a++;
		if (a >= MAX_STRING - 1)
			a--;   // Truncate too long words
	}
	word[a] = 0; // 在字符串结尾加上终止符v
}

/* Returns hash value of a word */
int GetWordHash(char *word) {
	unsigned long long a, hash = 0;
	for (a = 0; a < strlen(word); a++)
		hash = hash * 257 + word[a];
	hash = hash % vocab_hash_size;
	return hash;
}

int GetHowNetHash(char *word) {
	unsigned long long a, hash = 0;
	for (a = 0; a < strlen(word); a++)
		hash = hash * 257 + word[a];
	hash = hash % HOWNET_HASH_SIZE;
	return hash;
}

/* Returns position of a word in the vocabulary; if the word is not found,
 * returns -1 */
int SearchVocab(int lang_id, char *word) {
	unsigned int hash = GetWordHash(word);
	//if (lang_id >= NUM_LANG) { printf("lang_id >= NUM_LANG\n"); exit(1); }
	int *vocab_hash = vocab_hashes[lang_id];
	struct vocab_word *vocab = vocabs[lang_id];
	while (1) {
		if (vocab_hash[hash] == -1)
			return -1;
		if (!strcmp(word, vocab[vocab_hash[hash]].word))
			return vocab_hash[hash];
		hash = (hash + 1) % vocab_hash_size;
	}
	return -1;
}

/* Reads a word and returns its index in the vocabulary */
int ReadWordIndex(FILE *fin, int lang_id) {
	char word[MAX_STRING];
	ReadWord(word, fin);
	if (feof(fin))
		return -1;
	return SearchVocab(lang_id, word);         // MOD
}

/* Reads a single word from a file, assuming space + tab + EOL to be word
 boundaries. Unlike @ReadWord, this does not treat EOL to be special. */
void ReadWordNoEOL(char *word, FILE *fin) {
	int a = 0, ch;
	while (!feof(fin)) {
		ch = fgetc(fin);
		if (ch == 13)
			continue;
		if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
			if (a > 0)
				break;
			else
				continue;
		}
		word[a] = ch;
		a++;
		if (a >= MAX_STRING - 1)
			a--;   // Truncate too long words
	}
	word[a] = 0;
}

/* Calls @ReadWordNoEOL and then @SearchVocab to return its index in the vocabulary */
int ReadWordIndexNoEOL(FILE *fin, int lang_id) {
	char word[MAX_STRING];
	ReadWordNoEOL(word, fin);
	if (feof(fin))
		return -1;
	int wordID = SearchVocab(lang_id, word);
	if (wordID == -1)
		printf("Unknown word: %s\n", word);
	return wordID;
}

/* Adds a word to the vocabulary */
int AddWordToVocab(int lang_id, char *word) {
	unsigned int hash, length = strlen(word) + 1;
	struct vocab_word *vocab = vocabs[lang_id];
	int *vocab_hash = vocab_hashes[lang_id];      // array of *ints

	if (length > MAX_STRING)
		length = MAX_STRING;
	vocab[vocab_sizes[lang_id]].word = calloc(length, sizeof(char));
	strcpy(vocab[vocab_sizes[lang_id]].word, word);
	vocab[vocab_sizes[lang_id]].cn = 0;
	vocab_sizes[lang_id]++;
	// Reallocate memory if needed
	if (vocab_sizes[lang_id] + 2 >= vocab_max_size) {
		vocab_max_size += 1000;
		vocabs[lang_id] = (struct vocab_word *) realloc(vocabs[lang_id],
		                  vocab_max_size * sizeof(struct vocab_word));
	}
	hash = GetWordHash(word);
	while (vocab_hash[hash] != -1)
		hash = (hash + 1) % vocab_hash_size;
	vocab_hash[hash] = vocab_sizes[lang_id] - 1;
	return vocab_sizes[lang_id] - 1;
}

/* Used later for sorting by word counts */
int VocabCompare(const void *word1, const void *word2) {
	return ((struct vocab_word *) word2)->cn - ((struct vocab_word *) word1)->cn;
}

/* Sorts the vocabulary by frequency using word counts */
void SortVocab(int lang_id) {
	int a, size;
	unsigned int hash;
	struct vocab_word *vocab = vocabs[lang_id];
	int *vocab_hash = vocab_hashes[lang_id];

	// Sort the vocabulary and keep </s> at the first position
	qsort(&vocab[1], vocab_sizes[lang_id] - 1, sizeof(struct vocab_word),
	      VocabCompare);

	printf("After sorting, the first word in %lld's dict is %s\n", lang_id, vocab[0].word);

	for (a = 0; a < vocab_hash_size; a++)
		vocab_hash[a] = -1;
	size = vocab_sizes[lang_id];
	train_words[lang_id] = 0;
	for (a = 0; a < size; a++) {
		// Words occuring less than min_count times will be discarded from the vocab
		if (vocab[a].cn < min_count) {
			vocab_sizes[lang_id]--;
			free(vocab[vocab_sizes[lang_id]].word);
		} else {
			// Hash will be re-computed, as after the sorting it is not correct
			hash = GetWordHash(vocab[a].word);
			while (vocab_hash[hash] != -1)
				hash = (hash + 1) % vocab_hash_size;
			vocab_hash[hash] = a;
			train_words[lang_id] += vocab[a].cn;
		}
	}
	vocabs[lang_id] = (struct vocab_word *) realloc(vocabs[lang_id],
	                  (vocab_sizes[lang_id] + 1) * sizeof(struct vocab_word)); // 按照词表实际大小重新分配内存
}

/* Reduces the vocabulary by removing infrequent tokens */
void ReduceVocab(int lang_id) {
	int a, b = 1; //确保</s>不会被删掉
	unsigned int hash;
	long long vocab_size = vocab_sizes[lang_id];
	struct vocab_word *vocab = vocabs[lang_id];
	int *vocab_hash = vocab_hashes[lang_id];

	for (a = 1; a < vocab_size; a++) // 压缩词表，把词频小于等于min_reduce的词都删掉
		if (vocab[a].cn > min_reduce ) {
			vocab[b].cn = vocab[a].cn;
			vocab[b].word = vocab[a].word;
			b++;
		} else
			free(vocab[a].word);
	vocab_sizes[lang_id] = b;
	// 重新计算哈希值
	for (a = 0; a < vocab_hash_size; a++)
		vocab_hash[a] = -1;
	for (a = 0; a < vocab_size; a++) {
		// Hash will be re-computed, as it is not correct
		hash = GetWordHash(vocab[a].word);
		while (vocab_hash[hash] != -1)
			hash = (hash + 1) % vocab_hash_size;
		vocab_hash[hash] = a;
	}
	fflush(stdout);
	min_reduce++;
}

void LearnVocabFromTrainFile(int lang_id) {
	char word[MAX_STRING], *train_file = mono_train_files[lang_id];
	FILE *fin;
	long long a, i;
	int *vocab_hash = vocab_hashes[lang_id];
	struct vocab_word *vocab = vocabs[lang_id];
	for (a = 0; a < vocab_hash_size; a++)
		vocab_hash[a] = -1;
	fin = fopen(train_file, "rb");
	if (fin == NULL) {
		printf("ERROR: training data (%s) file not found (lang_id==%d)!\n",
		       train_file, lang_id);
		exit(1);
	}
	vocab_sizes[lang_id] = 0;
	AddWordToVocab(lang_id, (char *) "</s>");
	printf("Adding </s> succeed! for language %lld\n", lang_id);

	vocab = vocabs[lang_id];

	while (1) {
		ReadWord(word, fin);
		if (feof(fin))
			break;
		train_words[lang_id]++;
		// learn only on the first EARLY_STOP words if the flag is set
		if (EARLY_STOP > 0 && train_words[lang_id] > EARLY_STOP)
			break;
		if ((debug_mode > 1) && (train_words[lang_id] % 100000 == 0)) {
			fprintf(stderr, "%lldK%c", train_words[lang_id] / 1000, 13);
			fflush(stdout);
		}
		i = SearchVocab(lang_id, word);
		if (i == -1) {
			a = AddWordToVocab(lang_id, word);
			vocab = vocabs[lang_id];      // might have changed
			vocab[a].cn = 1;
		} else
			vocab[i].cn++;
		if (vocab_sizes[lang_id] > vocab_hash_size * 0.7) {
			ReduceVocab(lang_id); // 每次添加后都会检查！如果一直超会导致min_reduce一直增加
		}
	}
	printf("The first word in language %lld is %s\n", lang_id, vocab[0].word);

	fprintf(stderr, "pre SortVocab\n");
	SortVocab(lang_id);

	printf("The first word in language %lld is %s\n", lang_id, vocab[0].word);

	if (debug_mode > 0) {
		fprintf(stderr, "Vocab size: %lld\n", vocab_sizes[lang_id]);
		fprintf(stderr, "Words in train file: %lld\n", train_words[lang_id]);
	}
	file_sizes[lang_id] = ftell(fin);
	fclose(fin);
}

void SaveVocab(int lang_id) {
	long long i;
	char *save_vocab_file = save_vocab_files[lang_id];
	struct vocab_word *vocab = vocabs[lang_id];

	FILE *fo = fopen(save_vocab_file, "wb"); // 注意是二进制！
	fprintf(stderr, "Saving vocabulary with %lld entries to %s\n", vocab_sizes[lang_id],
	        save_vocab_file);
	for (i = 0; i < vocab_sizes[lang_id]; i++)
		fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
	fclose(fo);
}

void ReadVocab(int lang_id) {
	long long a, i = 0;
	char c;
	char word[MAX_STRING];
	int *vocab_hash = vocab_hashes[lang_id];
	char *train_file = mono_train_files[lang_id];
	FILE *fin = fopen(read_vocab_files[lang_id], "rb");

	if (fin == NULL) {
		printf("Vocabulary file not found\n");
		exit(1);
	}

	for (a = 0; a < vocab_hash_size; a++)
		vocab_hash[a] = -1;
	vocab_sizes[lang_id] = 0;
	while (1) {
		ReadWord(word, fin);
		if (feof(fin))
			break;
		a = AddWordToVocab(lang_id, word);      // can change vocabs
		fscanf(fin, "%lld%c", &vocabs[lang_id][a].cn, &c);
		i++;
	}
	SortVocab(lang_id);
	if (debug_mode > 0) {
		fprintf(stderr, "Vocab size: %lld\n", vocab_sizes[lang_id]);
		fprintf(stderr, "Words in train file: %lld\n", train_words[lang_id]);
	}
	fin = fopen(train_file, "rb");
	if (fin == NULL) {
		printf("ERROR: training data file (%s) not found!\n", train_file);
		exit(1);
	}
	fseek(fin, 0, SEEK_END);
	file_sizes[lang_id] = ftell(fin);
	fclose(fin);
}

void LoadLexicon() {
	FILE *fin0, *fin1;
	long long i0, i1;
	fin0 = fopen(lexicon_files[0], "rb");
	fin1 = fopen(lexicon_files[1], "rb");
	if (fin0 == NULL || fin1 == NULL) {
		printf("ERROR: lexicon file not found!\n");
		exit(1);
	}
	int j;
	for (j = 0; j < MAX_VOCAB_SIZE; j++) {
		srcVocabInLexicon[j] = 0;
		tgtVocabInLexicon[j] = 0;
	}
	lexicon_size = 0;
	i0 = SearchVocab(0, (char *) "</s>");
	i1 = SearchVocab(1, (char *) "</s>"); // i1为什么是-1？
	i0 = 0; i1 = 0; //？
	if (i0 != 0 || i1 != 0) {
		printf("ERROR: </s> at vocabulary position %lld %lld!\n", i0, i1);
		exit(1);
	}
	if (i0 != -1 && i1 != -1) { // 在两种语言的词表中都能找到</s>
		lexicons[0][lexicon_size] = i0;
		lexicons[1][lexicon_size] = i1;
		lexicon_size++;
		srcVocabInLexicon[i0] = 1;
		tgtVocabInLexicon[i1] = 1;
	}
	while (1) {
		i0 = ReadWordIndexNoEOL(fin0, 0);
		i1 = ReadWordIndexNoEOL(fin1, 1);
		if (feof(fin0) || feof(fin1))
			break;
		if (i0 != -1 && i1 != -1) {
			lexicons[0][lexicon_size] = i0;
			lexicons[1][lexicon_size] = i1;
			lexicon_size++;
			srcVocabInLexicon[i0] = 1;
			tgtVocabInLexicon[i1] = 1;
		}
	}
	fprintf(stderr, "Lexicon size (including </s>): %lld\n", lexicon_size);
	fclose(fin0);
	fclose(fin1);
}

void ReadSememes() {
	FILE * fin;
	char word[MAX_STRING];
	unsigned int length;
	fin = fopen(sememe_file, "rb");
	if (fin == NULL) {
		printf("ERROR: sememe file not found!\n");
		exit(1);
	}
	while (1) {
		ReadWordNoEOL(word, fin);
		if (feof(fin))
			break;
		length = strlen(word) + 1;
		sememes[sememe_size].word = calloc(length, sizeof(char));
		strcpy(sememes[sememe_size].word, word);
		sememe_size++;
	}
	return;
}
void ReadHowNet() {
	FILE * fin;
	int sememe_num, sememe_tmp[30], a;
	char word[MAX_STRING];
	unsigned int hash, length;

	fin = fopen(hownet_file, "rb");
	if (fin == NULL) {
		printf("ERROR: hownet file not found!\n");
		exit(1);
	}

	for (a = 0; a < HOWNET_HASH_SIZE; a++)
		hownet_hash[a] = -1;
	hownet_size = 0;


	while (1) {
		ReadWordNoEOL(word, fin); // 读词
		if (feof(fin))
			break;
		printf("Add word of HowNet: %s\n", word);

		length = strlen(word) + 1;
		hownet[hownet_size].word = calloc(length, sizeof(char));
		strcpy(hownet[hownet_size].word, word);

		// 记录哈希值
		hash = GetHowNetHash(word);
		while (hownet_hash[hash] != -1)
			hash = (hash + 1) % HOWNET_HASH_SIZE;
		hownet_hash[hash] = hownet_size;

		// 读取义原
		sememe_num = 0;
		while (1) {
			ReadWord(word, fin); // 注意这里word每次都会改变
			if (feof(fin)) // 读取到eof
				break;

			if (!strcmp(word, (char *)"</s>")) // 读取到行尾
				break;
			printf("\tAdd sememe: %s\n", word);

			for (a = 0; a < sememe_size; a++) // 遍历各个义原
				if (!strcmp(word, sememes[a].word)) {
					sememe_tmp[sememe_num++] = a;
					break;
				}
		}
		hownet[hownet_size].sememe_num = sememe_num;
		hownet[hownet_size].sememe_idx = calloc(sememe_num, sizeof(int));
		for (a = 0; a < sememe_num; a++)
			hownet[hownet_size].sememe_idx[a] = sememe_tmp[a];

		hownet_size++;
	}
	return;
}
void InitNetSememe() {
	long long a, b;
	// 为义原向量开空间
	a = posix_memalign((void **) &sememe_vec1, 128,
	                   (long long) sememe_size * layer1_size * sizeof(real));
	if (sememe_vec1 == NULL) {
		printf("Memory allocation failed\n");
		exit(1);
	}

	a = posix_memalign((void **) &sememe_vec2, 128,
	                   (long long) sememe_size * layer1_size * sizeof(real));
	if (sememe_vec2 == NULL) {
		printf("Memory allocation failed\n");
		exit(1);
	}
	// 为bias向量开空间
	a = posix_memalign((void **) &word_bias, 128,
	                   (long long) hownet_size * sizeof(real));
	if (word_bias == NULL) {
		printf("Memory allocation failed\n");
		exit(1);
	}

	a = posix_memalign((void **) &sememe_bias, 128,
	                   (long long) sememe_size * sizeof(real));
	if (sememe_bias == NULL) {
		printf("Memory allocation failed\n");
		exit(1);
	}

	// 为AdaGrad向量开空间
	a = posix_memalign((void **) &sememe_vec_ada1, 128,
	                   (long long) sememe_size * layer1_size * sizeof(real));
	if (sememe_vec_ada1 == NULL) {
		printf("Memory allocation failed\n");
		exit(1);
	}
	a = posix_memalign((void **) &sememe_vec_ada2, 128,
	                   (long long) sememe_size * layer1_size * sizeof(real));
	if (sememe_vec_ada2 == NULL) {
		printf("Memory allocation failed\n");
		exit(1);
	}
	a = posix_memalign((void **) &word_bias_ada, 128,
	                   (long long) hownet_size * sizeof(real));
	if (word_bias_ada == NULL) {
		printf("Memory allocation failed\n");
		exit(1);
	}

	a = posix_memalign((void **) &sememe_bias_ada, 128,
	                   (long long) sememe_size * sizeof(real));
	if (sememe_bias_ada == NULL) {
		printf("Memory allocation failed\n");
		exit(1);
	}

	// 初始化
	for (a = 0; a < hownet_size; a++) {
		word_bias[a] =  (rand() / (real) RAND_MAX - 0.5) / layer1_size;
		word_bias_ada[a] = 1;
	}
	for (a = 0; a < sememe_size; a++) {
		sememe_bias[a] =  (rand() / (real) RAND_MAX - 0.5) / layer1_size;
		sememe_bias_ada[a] = 1;
		for (b = 0; b < layer1_size; b++) {
			sememe_vec1[a * layer1_size + b] =  (rand() / (real) RAND_MAX - 0.5) / layer1_size;
			sememe_vec2[a * layer1_size + b] =  (rand() / (real) RAND_MAX - 0.5) / layer1_size;
			sememe_vec_ada1[a * layer1_size + b] = 1;
			sememe_vec_ada2[a * layer1_size + b] = 1;
		}
	} // rand()是否需要使用srand()？
}

void InitNet(int lang_id) {
	long long a, b, vocab_size = vocab_sizes[lang_id];
	real *syn0, *syn1neg, *syn0grad, *syn1negGrad;

	// 分别为输入和输出向量开空间
	a = posix_memalign((void **) &syn0, 128,
	                   (long long) vocab_size * layer1_size * sizeof(real));
	if (syn0 == NULL) {
		printf("Memory allocation failed\n");
		exit(1);
	} else
		syn0s[lang_id] = syn0;
	a = posix_memalign((void **) &syn1neg, 128,
	                   (long long) vocab_size * layer1_size * sizeof(real));
	if (syn1neg == NULL) {
		printf("Memory allocation failed\n");
		exit(1);
	} else
		syn1negs[lang_id] = syn1neg;

	// adagrad向量开空间
	if (adagrad) {
		a = posix_memalign((void **) &syn0grad, 128,
		                   (long long) vocab_size * layer1_size * sizeof(real));
		if (syn0grad == NULL) {
			printf("Memory allocation failed\n");
			exit(1);
		} else
			syn0grads[lang_id] = syn0grad;
		a = posix_memalign((void **) &syn1negGrad, 128,
		                   (long long) vocab_size * layer1_size * sizeof(real));
		if (syn1negGrad == NULL) {
			printf("Memory allocation failed\n");
			exit(1);
		} else
			syn1negGrads[lang_id] = syn1negGrad;
	}
	// 初始化
	for (b = 0; b < layer1_size; b++) {
		for (a = 0; a < vocab_size; a++) {
			syn1neg[a * layer1_size + b] = 0; // 为什么这里用0初始化？
			syn0[a * layer1_size + b] = (rand() / (real) RAND_MAX - 0.5) / layer1_size;

			if (adagrad) {
				syn0grad[a * layer1_size + b] = 0;
				syn1negGrad[a * layer1_size + b] = 0;
			}
		}
	}
}

char SubSample(int lang_id, long long word_id) {
	long long count = vocabs[lang_id][word_id].cn;
	real thresh = (sqrt(count / (sample * train_words[lang_id])) + 1)
	              * (sample * train_words[lang_id]) / count;
	next_random = next_random * (unsigned long long) 25214903917 + 11;
	if ((next_random & 0xFFFF) / (real) 65536 > thresh)
		return 1;
	else
		return 0;
}

/* Read a sentence into *sen using vocabulary for language lang_id
* Store processed words in *sen, returns (potentially subsampled)
* length of sentence */
int ReadSent(FILE * fi, int lang_id, long long * sen, char subsample) {
	long long word;
	int sentence_length = 0;
	//struct vocab_word *vocab = vocabs[lang_id];
	while (1) {
		word = ReadWordIndex(fi, lang_id);
		if (feof(fi))
			break;
		if (word == -1)
			continue;       // unknown
		if (word == 0)
			break;           // 换行符\n，表示 end-of-sentence
		// The subsampling randomly discards frequent words while keeping the
		// ranking the same.
		if (subsample && sample > 0) { // sample由输入设置，推荐值为1e-5
			if (SubSample(lang_id, word)) // 若降采样成功，则跳过该词
				continue;
		}
		sen[sentence_length] = word;
		sentence_length++;
		if (sentence_length >= MAX_SEN_LEN)
			break;
	}
	return sentence_length;
}

//Zm: @grads is only used for AdaGrad
void UpdateEmbeddings(real * embeddings, real * grads, int offset,
                      int num_updates, real * deltas, real weight) {
	int a;
	real step, epsilon = 1e-6;
	for (a = 0; a < num_updates; a++) {
		if (adagrad) {
			// Use Adagrad for automatic learning rate selection
			grads[offset + a] += (deltas[a] * deltas[a]);
			step = (alpha / fmax(epsilon, sqrt(grads[offset + a]))) * deltas[a];
		} else {
			// Regular SGD
			step = alpha * deltas[a];
		}
		if (step != step) {
			fprintf(stderr, "ERROR: step == NaN\n");
		}
		step = step * weight;
		if (CLIP_UPDATES != 0) {
			if (step > CLIP_UPDATES)
				step = CLIP_UPDATES;
			if (step < -CLIP_UPDATES)
				step = -CLIP_UPDATES;
		}
		embeddings[offset + a] += step;
	}
}

void LexiconUpdate(long long w_I, long long w_O, int lang_id1, int lang_id2, // w_I,w_O表示词典的一对词，
                   real lambda, real * delta) {
	long long l1, l2;
	int c;
	l1 = w_I * layer1_size;
	l2 = w_O * layer1_size;
	//update input vector
	for (c = 0; c < layer1_size; c++) {
		delta[c] = syn0s[lang_id1][c + l1] - syn0s[lang_id2][c + l2];
	}
	UpdateEmbeddings(syn0s[lang_id1], syn0grads[lang_id1], l1,
	                 layer1_size, delta, -lambda);
	UpdateEmbeddings(syn0s[lang_id2], syn0grads[lang_id2], l2,
	                 layer1_size, delta, lambda);
	//update output vector
	for (c = 0; c < layer1_size; c++) {
		delta[c] = syn1negs[lang_id1][c + l1] - syn1negs[lang_id2][c + l2];
	}
	UpdateEmbeddings(syn1negs[lang_id1], syn1negGrads[lang_id1], l1,
	                 layer1_size, delta, -lambda);
	UpdateEmbeddings(syn1negs[lang_id2], syn1negGrads[lang_id2], l2,
	                 layer1_size, delta, lambda);
}

void MatchUpdate(long long w_I, long long w_O, int lang_id1, int lang_id2,
                 real lambda, real * delta) {
	long long l1, l2;
	int c;
	l1 = w_I * layer1_size;
	l2 = w_O * layer1_size;

	for (c = 0; c < layer1_size; c++) {
		delta[c] = syn0s[lang_id1][c + l1] + syn1negs[lang_id1][c + l1] - syn0s[lang_id2][c + l2] - syn1negs[lang_id2][c + l2];
	} // 这个为什么不像Lexicon Update一样对两套词向量分别计算delta来更新？是因为这个matching是根据平均词向量最相近求出来的？

	//update input vector
	UpdateEmbeddings(syn0s[lang_id1], syn0grads[lang_id1], l1,
	                 layer1_size, delta, -lambda);
	UpdateEmbeddings(syn0s[lang_id2], syn0grads[lang_id2], l2,
	                 layer1_size, delta, lambda);
	//update output vector
	UpdateEmbeddings(syn1negs[lang_id1], syn1negGrads[lang_id1], l1,
	                 layer1_size, delta, -lambda);
	UpdateEmbeddings(syn1negs[lang_id2], syn1negGrads[lang_id2], l2,
	                 layer1_size, delta, lambda);
}

void *LexiconThread(void *id) {
	char LOCAL_ALL_MONO_DONE;
	int thread_id = (int) id % num_threads; // 没有必要，直接令thread_id=id即可
	long long entry;	//entry in the lexicon
	real deltas1[layer1_size], deltas2[layer1_size];

	entry = lexicon_size / num_threads * thread_id; // 各个线程读词表的起始位置

	// Continue training while monolingual models are still training
	while (1) { //MONO_DONE_TRAINING < NUM_LANG * num_threads) {
		pthread_rwlock_rdlock(&lock);
		LOCAL_ALL_MONO_DONE = ALL_MONO_DONE;
		pthread_rwlock_unlock(&lock);
		if (LOCAL_ALL_MONO_DONE) break; // 如果单语线程已经结束，那么退出；否则一直训练
		if (entry >= lexicon_size / num_threads * (thread_id + 1)) { // 读到了该线程对应词表的末位，则返回起始位置
			entry = lexicon_size / num_threads * thread_id;
			continue;
		}
		LexiconUpdate(lexicons[0][entry], lexicons[1][entry], 0, 1, LEXICON_LAMBDA, deltas1);
		//LexiconUpdate(lexicons[1][entry], lexicons[0][entry], 1, 0, LEXICON_LAMBDA, deltas1, deltas2);
		entry++;
	} // while training loop
	//	fprintf(stderr, "Exiting lexicon thread %d. ALL_MONO_DONE = %d\n", (int)id, ALL_MONO_DONE);
	pthread_exit(NULL);
	return NULL;
}
// 从HowNet中找当前词表中的词

int SearchHowNet(long long entry) {
	char * word = vocabs[1][entry].word;
	unsigned int hash = GetHowNetHash(word);
	while (1) {
		if (hownet_hash[hash] == -1)
			return -1;
		if (!strcmp(word, hownet[hownet_hash[hash]].word))
			return hownet_hash[hash];
		hash = (hash + 1) % HOWNET_HASH_SIZE;
	}
	return -1;
}
real ClipStep(real step) {
	if (CLIP_UPDATES > 0) {
		if (step > CLIP_UPDATES)
			return CLIP_UPDATES;
		if (step < -CLIP_UPDATES)
			return -CLIP_UPDATES;
	}
	return step;
}

void *SememeThread(void *id) {
	char LOCAL_ALL_MONO_DONE;
	int thread_id = (int) id;
	int hownet_idx, a, b, c, sememe_in_word;
	long long zh_entry, l0, l1, zh_vocab_size = vocab_sizes[1];
	real * syn0 = syn0s[1], *syn1neg = syn1negs[1], delta, sememe_grad, word_grad;

	zh_entry = zh_vocab_size / num_threads * thread_id;
	while (1) {
		pthread_rwlock_rdlock(&lock);
		LOCAL_ALL_MONO_DONE = ALL_MONO_DONE;
		pthread_rwlock_unlock(&lock);
		if (LOCAL_ALL_MONO_DONE) break;  // 如果单语线程已经结束，那么退出；否则一直训练

		if (zh_entry >= zh_vocab_size / num_threads * (thread_id + 1)) { // 读到了该线程对应词表的末位，则返回起始位置
			zh_entry = zh_vocab_size / num_threads * thread_id;
			continue;
		}
		hownet_idx = SearchHowNet(zh_entry);// 给定一个词在当前词表中的index，返回该词在hownet词典中的index
		if (hownet_idx == -1) {
			zh_entry++;
			continue;
		}
		l0 = zh_entry * layer1_size; //词的offset

		// 遍历义原
		for (a = 0; a < sememe_size; a++) {
			// 判断当前义原是否属于当前词
			sememe_in_word = 0;
			for (b = 0; b < hownet[hownet_idx].sememe_num; b++)
				if (hownet[hownet_idx].sememe_idx[b] == a) {
					sememe_in_word = 1;
					break;
				}
			if (sememe_in_word == 0) { // 当前义原不属于当前词，则随机丢弃
				srand((unsigned)time(NULL));
				if (rand() / (real) RAND_MAX > 0.005)
					continue;
			}
			l1 = a * layer1_size; // 义原的offset
			// 计算loss
			delta = 0;
			for (c = 0; c < layer1_size; c++)
				delta += (syn0[l0 + c] + syn1neg[l0 + c]) * (sememe_vec1[l1 + c] + sememe_vec2[l1 + c]) / 2; // 这里应不应该除以2？
			delta += word_bias[hownet_idx] + sememe_bias[a] - sememe_in_word;
			printf("The delta for word:%s  sememe:%s is %f\n", hownet[hownet_idx].word, sememes[a].word, delta);
			// 义原向量更新
			for (c = 0; c < layer1_size; c++) {
				sememe_grad = delta * 2 * (syn0[l0 + c] + syn1neg[l0 + c]) / 2;
				//printf("The %d-th grad for word %s's sememe:%s  is %f\n", c,  hownet[hownet_idx].word, sememes[a].word, sememe_grad);

				sememe_vec1[l1 + c] -= ClipStep(alpha * sememe_grad / sememe_vec_ada1[l1 + c]);
				sememe_vec2[l1 + c] -= ClipStep(alpha * sememe_grad / sememe_vec_ada2[l1 + c]);// AdaGrad写的对吗？

				sememe_vec_ada1[l1 + c] += ClipStep(sememe_grad * sememe_grad);
				sememe_vec_ada2[l1 + c] += ClipStep(sememe_grad * sememe_grad);
			}
			// bias更新
			word_bias[hownet_idx] -= ClipStep(2 * delta * alpha / word_bias_ada[hownet_idx]);
			word_bias_ada[hownet_idx] += ClipStep(4 * delta * delta);

			sememe_bias[a] -= ClipStep(2 * delta * alpha / sememe_bias_ada[a]);
			sememe_bias_ada[a] += ClipStep(4 * delta * delta);
			// 词向量更新
			for (c = 0; c < layer1_size; c++) {
				word_grad = delta * 2 * (sememe_vec1[l1 + c] + sememe_vec2[l1 + c]) / 2;
				//printf("The %d-th grad for sememe %s's word:%s is %f\n", c, sememes[a].word, hownet[hownet_idx].word, word_grad);
				syn0[l0 + c] -= ClipStep(alpha * word_grad * SEMEME_LAMBDA); // 这里就简单用负梯度可以么？
				syn1neg[l0 + c] -= ClipStep(alpha * word_grad * SEMEME_LAMBDA);
			}
		} // sememe end
		zh_entry++;
	}// while end
	pthread_exit(NULL);
	return NULL;
}

real dot_product(real * embeddings0, real * embeddings1, int offset0, int offset1, int length) {
	real result = 0;
	int i;
	for (i = 0; i < length; i++) {
		result += embeddings0[offset0 + i] * embeddings1[offset1 + i];
	}
	return result;
}

real add_dot_product(real * embeddings0, real * contexts0, real * embeddings1, real * contexts1, int offset0, int offset1, int length) {
	real result = 0;
	int i;
	for (i = 0; i < length; i++) {
		result += (embeddings0[offset0 + i] + contexts0[offset0 + i]) * (embeddings1[offset1 + i] + contexts1[offset1 + i]);
	}
	return result;
}

real lookupExpTable(real x) {
	if (x >= MAX_EXP) return expTable[EXP_TABLE_SIZE - 1];
	if (x < -MAX_EXP) return expTable[0];
	return expTable[(int) ((x + MAX_EXP) / MAX_EXP / 2 * EXP_TABLE_SIZE)];
}

void *MatchingT2SThread(void *id) {
	char LOCAL_ALL_MONO_DONE;
	int thread_id = (int) id % num_threads, j, m;
	long long max_src_entry, src_entry, max_tgt_entry, tgt_entry, l0, l1;
	long long src_vocab_size = vocab_sizes[0], tgt_vocab_size = vocab_sizes[1];
	real deltas1[layer1_size], deltas2[layer1_size];
	real src_norm, max_src_norm, tgt_norm, cos_sim, max_cos_sim, norm;
	char valid;

	//src_entry = src_vocab_size / num_threads * thread_id;
	tgt_entry = tgt_vocab_size / num_threads * thread_id; // 该线程处理的目标语言词表起始位置

	// Continue training while monolingual models are still training
	while (1) { //(MONO_DONE_TRAINING < NUM_LANG * num_threads) {
		pthread_rwlock_rdlock(&lock);
		LOCAL_ALL_MONO_DONE = ALL_MONO_DONE;
		pthread_rwlock_unlock(&lock);
		if (LOCAL_ALL_MONO_DONE) break;  // 如果单语线程已经结束，那么退出；否则一直训练
		if (tgt_entry >= tgt_vocab_size / num_threads * (thread_id + 1)) { // 读到了该线程对应词表的末位，则返回起始位置
			tgt_entry = tgt_vocab_size / num_threads * thread_id;
			continue;
		}
		if (tgtVocabInLexicon[tgt_entry]) { // 如果当前词在词典中，则跳过。
			tgt_entry++;
			continue;
		}
		l1 = tgt_entry * layer1_size;
		tgt_norm = sqrt(add_dot_product(syn0s[1], syn1negs[1], syn0s[1], syn1negs[1], l1, l1, layer1_size));// 这个不用除以2吗
		max_cos_sim = -1;
		for (src_entry = 1; src_entry < src_vocab_size; src_entry++) { // 对每个源语言词
			if (srcVocabInLexicon[src_entry]) continue; // 若该词在词典中，则跳过
			l0 = src_entry * layer1_size;
			src_norm = sqrt(add_dot_product(syn0s[0], syn1negs[0], syn0s[0], syn1negs[0], l0, l0, layer1_size));
			cos_sim = add_dot_product(syn0s[0], syn1negs[0], syn0s[1], syn1negs[1], l0, l1, layer1_size) / (src_norm * tgt_norm); // 求出源语言词和当前目标语言词的cos相似度
			if (cos_sim > max_cos_sim) { // 找到和当前目标语言词最相似的源语言词
				max_cos_sim = cos_sim;
				max_src_entry = src_entry;
				//				max_tgt_norm = tgt_norm;
			}
		}
		//		printf("max probability = %f\n", max_cos_sim);
		//		valid = 1;
		//		l0 = max_src_entry * layer1_size;
		//		for (j = 1; j < tgt_vocab_size; j++) {
		//			if (j == tgt_entry) continue;
		//			l1 = j * layer1_size;
		//			tgt_norm = sqrt(dot_product(syn0s[1], syn0s[1], l1, l1, layer1_size));
		//			cos_sim = dot_product(syn0s[0], syn0s[1], l0, l1, layer1_size) / (max_src_norm * tgt_norm);
		//			if (cos_sim > max_cos_sim) {
		//				valid = 0;
		//				break;
		//			}
		//		}
		if (/*valid && */max_cos_sim > threshold) { // 默认值为0.5
			//			printf("source - target - cos_sim: %s %s %f\n", vocabs[0][src_entry].word, vocabs[1][max_tgt_entry].word, max_cos_sim);
			printf("target - source - cos_sim: %s %s %f\n", vocabs[1][tgt_entry].word, vocabs[0][max_src_entry].word, max_cos_sim);
			for (m = 0; m < MSTEP_ITER; m++) {
				//				LexiconUpdate(src_entry, max_tgt_entry, 0, 1, MATCHING_LAMBDA, deltas1);
				MatchUpdate(max_src_entry, tgt_entry, 0, 1, MATCHING_LAMBDA * vocabs[1][tgt_entry].cn / train_words[1], deltas1);
			}
		}
		tgt_entry++;
	} // while training loop
	//	fprintf(stderr, "Exiting matching thread %d. ALL_MONO_DONE = %d\n", (int)id, ALL_MONO_DONE);
	pthread_exit(NULL);
	return NULL;
}

void *MatchingS2TThread(void *id) {
	char LOCAL_ALL_MONO_DONE;
	int thread_id = (int) id % num_threads, j, m;
	long long max_src_entry, src_entry, max_tgt_entry, tgt_entry, l0, l1;
	long long src_vocab_size = vocab_sizes[0], tgt_vocab_size = vocab_sizes[1];
	real deltas1[layer1_size], deltas2[layer1_size];
	real src_norm, max_src_norm, tgt_norm, cos_sim, max_cos_sim, norm;
	char valid;

	//src_entry = src_vocab_size / num_threads * thread_id;
	src_entry = src_vocab_size / num_threads * thread_id;

	// Continue training while monolingual models are still training
	while (1) { //(MONO_DONE_TRAINING < NUM_LANG * num_threads) {
		pthread_rwlock_rdlock(&lock);
		LOCAL_ALL_MONO_DONE = ALL_MONO_DONE;
		pthread_rwlock_unlock(&lock);
		if (LOCAL_ALL_MONO_DONE) break;
		if (src_entry >= src_vocab_size / num_threads * (thread_id + 1)) {
			src_entry = src_vocab_size / num_threads * thread_id;
			continue;
		}
		if (srcVocabInLexicon[src_entry]) {
			src_entry++;
			continue;
		}
		l0 = src_entry * layer1_size;
		src_norm = sqrt(add_dot_product(syn0s[0], syn1negs[0], syn0s[0], syn1negs[0], l0, l0, layer1_size));
		max_cos_sim = -1;
		for (tgt_entry = 1; tgt_entry < tgt_vocab_size; tgt_entry++) {
			if (tgtVocabInLexicon[tgt_entry]) continue;
			l1 = tgt_entry * layer1_size;
			tgt_norm = sqrt(add_dot_product(syn0s[1], syn1negs[1], syn0s[1], syn1negs[1], l1, l1, layer1_size));
			cos_sim = add_dot_product(syn0s[0], syn1negs[0], syn0s[1], syn1negs[1], l0, l1, layer1_size) / (src_norm * tgt_norm);
			if (cos_sim > max_cos_sim) {
				max_cos_sim = cos_sim;
				max_tgt_entry = tgt_entry;
				//				max_tgt_norm = tgt_norm;
			}
		}
		//		printf("max probability = %f\n", max_cos_sim);
		//		valid = 1;
		//		l0 = max_src_entry * layer1_size;
		//		for (j = 1; j < tgt_vocab_size; j++) {
		//			if (j == tgt_entry) continue;
		//			l1 = j * layer1_size;
		//			tgt_norm = sqrt(dot_product(syn0s[1], syn0s[1], l1, l1, layer1_size));
		//			cos_sim = dot_product(syn0s[0], syn0s[1], l0, l1, layer1_size) / (max_src_norm * tgt_norm);
		//			if (cos_sim > max_cos_sim) {
		//				valid = 0;
		//				break;
		//			}
		//		}
		if (/*valid && */max_cos_sim > threshold) {
			printf("source - target - cos_sim: %s %s %f\n", vocabs[0][src_entry].word, vocabs[1][max_tgt_entry].word, max_cos_sim);
			//			printf("target - source - cos_sim: %s %s %f\n", vocabs[1][tgt_entry].word, vocabs[0][max_src_entry].word, max_cos_sim);
			for (m = 0; m < MSTEP_ITER; m++) {
				MatchUpdate(src_entry, max_tgt_entry, 0, 1, MATCHING_LAMBDA * vocabs[0][src_entry].cn / train_words[0], deltas1);
				//				LexiconUpdate(max_src_entry, tgt_entry, 0, 1, MATCHING_LAMBDA*vocabs[1][tgt_entry].cn/train_words[1], deltas1);
			}
		}
		src_entry++;
	} // while training loop
	//	fprintf(stderr, "Exiting matching thread %d. ALL_MONO_DONE = %d\n", (int)id, ALL_MONO_DONE);
	pthread_exit(NULL);
	return NULL;
}

void SaveModel(int lang_id, char *name) {
	long a, b;
	struct vocab_word *vocab = vocabs[lang_id];
	real *syn0 = syn0s[lang_id];
	real *syn1neg = syn1negs[lang_id];
	FILE *fo = fopen(name, "wb");

	fprintf(stderr, "\nSaving model to file: %s\n", name);
	fprintf(fo, "%lld %lld\n", vocab_sizes[lang_id], layer1_size);
	for (a = 0; a < vocab_sizes[lang_id]; a++) {
		fprintf(fo, "%s ", vocab[a].word);
		if (binary) {
			fprintf(stderr, "Not supported!\n");
			//                      for (b = 0; b < layer1_size; b++)
			//                              fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
		} else
			for (b = 0; b < layer1_size; b++)
				fprintf(fo, "%lf ",
				        syn0[a * layer1_size + b]
				        + syn1neg[a * layer1_size + b]);  // 注意保存的是两套词向量的和
		fprintf(fo, "\n");
	}
	fclose(fo);
}
void SaveSememe() {
	int a, b;
	FILE *fo = fopen(save_sememe_file, "wb");
	fprintf(stderr, "\nSaving sememe embeddings to file: %s\n", save_sememe_file);
	fprintf(fo, "%lld %lld\n", sememe_size, layer1_size);
	for (a = 0; a < sememe_size; a++) {
		fprintf(fo, "%s ", sememes[a].word);
		if (binary) {
			fprintf(stderr, "Not supported!\n");
			//                      for (b = 0; b < layer1_size; b++)
			//                              fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
		} else
			for (b = 0; b < layer1_size; b++)
				fprintf(fo, "%lf ",
				        sememe_vec1[a * layer1_size + b]
				        + sememe_vec2[a * layer1_size + b]);
		fprintf(fo, "\n");
	}
	fclose(fo);
}

/* Monolingual training thread */
void *MonoModelThread(void *id) {
	long long a, b, d, word, last_word, sentence_length = 0, sentence_position =
	            0;
	long long word_count = 0, last_word_count = 0, all_train_words = 0;
	long long mono_sen[MAX_SEN_LEN + 1];
	long long l1, l2, c, target, label;
	int lang_id = (int) id / num_threads, thread_id = (int) id % num_threads, cw;
	char *train_file = mono_train_files[lang_id];
	long long vocab_size = vocab_sizes[lang_id];
	real f, g;
	clock_t now;
	real *neu1 = calloc(layer1_size, sizeof(real));
	real *neu1e = calloc(layer1_size, sizeof(real));
	real *syn1neg = syn1negs[lang_id]; // 输出向量
	real *syn1negDelta = calloc(layer1_size, sizeof(real));
	real *syn0 = syn0s[lang_id]; // 输出向量
	FILE *fi = fopen(train_file, "rb");

	if (!EARLY_STOP)
		// If two languages have different amounts of training data,
		// recycle the smaller language data while there is more data
		// for the other language
		all_train_words = max_train_words * 1 * NUM_LANG;	//Zm: always train for 1 epoch for each thread
	else {
		all_train_words = EARLY_STOP;
	}

	if (dump_every < 0) {
		dump_every = max_train_words / abs(dump_every);
	}

	fseek(fi, file_sizes[lang_id] / (long long) num_threads * thread_id,
	      SEEK_SET);
	while (1) {
		if (word_count - last_word_count > 10000) {
			word_count_actual += word_count - last_word_count; // word_count_actual为全局变量，记录各个线程的总训练词数
			last_word_count = word_count;
			if ((debug_mode > 1)) {
				now = clock();
				fprintf(stderr,
				        "%cAlpha: %f  Progress: %.2f%%  (epoch %lld) Updates (L1: %.2fM, "
				        "L2: %.2fM) Words/sec: %.2fK \n",
				        13, alpha,
				        word_count_actual / (real) (all_train_words + 1) * 100,
				        epoch[0], lang_updates[0] / (real) 1000000,
				        lang_updates[1] / (real) 1000000,
				        word_count_actual
				        / ((real) (now - start + 1)
				           / (real) CLOCKS_PER_SEC * 1000));
				fflush(stdout);
			}
			//			if (!adagrad) {
			//				if (word_count_actual < (all_train_words + 1)) {
			//					alpha = starting_alpha
			//							* (1.0
			//									- word_count_actual
			//											/ (real) (all_train_words + 1));
			//				} else
			//					alpha = starting_alpha * 0.0001;
			//				//if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
			//			}
		}
		if (sentence_length == 0) { // 当前没有句子，则读取一个句子
			sentence_length = ReadSent(fi, lang_id, mono_sen, 1);
			word_count += sentence_length;
			sentence_position = 0;
		}
		if (lang_updates[lang_id] > all_train_words / NUM_LANG) // 当前某语言的实时已训练词数已经大于两个语料中较大词数时，说明较大语料已经训练完成
			break;  // 这是跳出while循环，结束该线程的主要出口

		if (lang_updates[lang_id] > 0
		        && lang_updates[lang_id] % max_train_words == 0) {
			epoch[lang_id]++;
		}

		if (feof(fi) || (word_count > train_words[lang_id] / num_threads)) {  // 当前线程训练词数已经超过平均训练词数
			word_count_actual += word_count - last_word_count;
			word_count = 0;
			last_word_count = 0;
			sentence_length = 0;
			fseek(fi, file_sizes[lang_id] / (long long) num_threads * thread_id, // 从头开始继续训练
			      SEEK_SET);
			continue;
		}
		if (EARLY_STOP) {
			if (word_count_actual > EARLY_STOP) {
				fprintf(stderr, "EARLY STOP point reached (thread %d)\n",
				        (int) id);
				break;
			}
		}
		// 开始训练！
		word = mono_sen[sentence_position];
		if (word == -1)
			continue;
		for (c = 0; c < layer1_size; c++)
			neu1[c] = 0;
		for (c = 0; c < layer1_size; c++)
			neu1e[c] = 0;
		next_random = next_random * (unsigned long long) 25214903917 + 11;
		b = next_random % window;

		// CBOW ARCHITECTURE WITH NEGATIVE SAMPLING
		if (cbow) {
			// QFC: 原来代码没有求context word的平均embedding，这里补上
			cw = 0; //周围词的数目
			for (a = b; a < window * 2 + 1 - b; a++) //为什么要用随机窗口大小？
				if (a != window) {
					c = sentence_position - window + a;// sentence_position表示的是当前词的位置；窗口范围：[sentense_position-window+b,sentense_position+window-b]
					// 判断c是否越界
					if (c < 0) continue;
					if (c >= sentence_length) continue;

					last_word = mono_sen[c];// 找到c对应的索引
					if (last_word == -1) continue; //之前已经判断过不要把不在词库的词加入

					for (c = 0; c < layer1_size; c++)
						neu1[c] += syn0[c + last_word * layer1_size];// 把各个周围词向量累加
					cw++;
				}
			if (cw) {  //周围词数大于0，开始计算梯度并更新
				for (c = 0; c < layer1_size; c++)
					neu1[c] /= cw;// 计算均值

				for (d = 0; d < negative + 1; d++) {
					if (d == 0) { // 正样本
						target = word;
						label = 1;
					} else
					{
						next_random = next_random * (unsigned long long) 25214903917
						              + 11;
						target = tables[lang_id][(next_random >> 16) % table_size];
						if (target == 0) // 选出</S>，则重新选
							target = next_random % (vocab_size - 1) + 1;
						if (target == word) continue; //选出正样本，则重新选
						label = 0;
					}
					l2 = target * layer1_size;  // 选出的样本词的词向量偏置
					f = 0;
					for (c = 0; c < layer1_size; c++)
						f += neu1[c] * syn1neg[c + l2];
					// learning rate alpha is applied in UpdateEmbeddings()
					if (f >= MAX_EXP)
						g = (label - 1);
					else if (f < -MAX_EXP)
						g = (label - 0);
					else
						g = (label
						     - sigmoidTable[(int) ((f + MAX_EXP) / MAX_EXP / 2 * EXP_TABLE_SIZE)]);

					for (c = 0; c < layer1_size; c++)
						neu1e[c] += g * syn1neg[c + l2];

					//for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];
					for (c = 0; c < layer1_size; c++)
						syn1negDelta[c] = neu1[c] * g; // syn1neg的变化量
					UpdateEmbeddings(syn1neg, syn1negGrads[lang_id], l2,
					                 layer1_size, syn1negDelta, +1); // 修改负采样方法中的逻辑回归的参数
				}
				// hidden -> in 修改词向量
				for (a = b; a < window * 2 + 1 - b; a++)
					if (a != window) {
						c = sentence_position - window + a;
						if (c < 0)  continue;
						if (c >= sentence_length)  continue;
						last_word = mono_sen[c];
						if (last_word == -1)  continue;
						//for (c = 0; c < layer1_size; c++)
						//syn0[c + last_word * layer1_size] += neu1e[c];
						UpdateEmbeddings(syn0, syn0grads[lang_id],
						                 last_word * layer1_size, layer1_size, neu1e, +1);
					}
			}
		} else {
			// SKIPGRAM ARCHITECTURE WITH NEGATIVE SAMPLING
			// Zm: This does not seem to be consistent with the paper.
			// The roles of w_I and w_O seem exchanged.
			// This is equivalent, according to youdao (Page 14).
			for (a = b; a < window * 2 + 1 - b; a++) {
				if (a != window) {
					c = sentence_position - window + a;
					if (c < 0)
						continue;
					if (c >= sentence_length)
						continue;
					last_word = mono_sen[c];
					if (last_word == -1)
						continue;
					l1 = last_word * layer1_size;  // 当前周围词的词向量存储偏置
					for (c = 0; c < layer1_size; c++)
						neu1e[c] = 0;
					// NEGATIVE SAMPLING
					for (d = 0; d < negative + 1; d++) {
						if (d == 0) {
							target = word;
							label = 1;
						} else {
							next_random = next_random
							              * (unsigned long long) 25214903917 + 11;
							target = tables[lang_id][(next_random >> 16)
							                         % table_size];
							if (target == 0)
								target = next_random % (vocab_size - 1) + 1;
							if (target == word)
								continue;
							label = 0;
						}
						l2 = target * layer1_size; // 负采样词的偏置
						f = 0;
						for (c = 0; c < layer1_size; c++)
							f += syn0[c + l1] * syn1neg[c + l2];
						// We multiply with the learning rate in UpdateEmbeddings()
						if (f >= MAX_EXP)
							g = (label - 1);
						else if (f < -MAX_EXP)
							g = (label - 0);
						else
							g = (label
							     - sigmoidTable[(int) ((f + MAX_EXP) / MAX_EXP / 2 * EXP_TABLE_SIZE)]);
						for (c = 0; c < layer1_size; c++)
							neu1e[c] += g * syn1neg[c + l2];
						//for (c = 0; c < layer1_size; c++)
						//syn1neg[c + l2] += g * syn0[c + l1];
						for (c = 0; c < layer1_size; c++)
							syn1negDelta[c] = g * syn0[c + l1];
						UpdateEmbeddings(syn1neg, syn1negGrads[lang_id], l2,  // 更新参数
						                 layer1_size, syn1negDelta, +1);
					}
					// Learn weights input -> hidden
					//for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
					UpdateEmbeddings(syn0, syn0grads[lang_id], l1, layer1_size,
					                 neu1e, +1); // 更新词向量
				}
			}   // for
		}   // skipgram
		lang_updates[lang_id]++;
		sentence_position++;
		if (dump_every > 0) {
			if (lang_updates[lang_id] % dump_every == 0) {
				char save_name[MAX_STRING];
				sprintf(save_name, output_files[lang_id],
				        dump_iters[lang_id]++);
				SaveModel(lang_id, save_name);
			}
		}
		if (sentence_position >= sentence_length) {
			sentence_length = 0;
			continue;
		}
	}
	fclose(fi);
	free(neu1);
	free(neu1e);
	MONO_DONE_TRAINING++; // 已结束训练线程数
	pthread_exit(NULL);
	return NULL;
}

void InitLexiconWords() {
	int a, i, srcEntry, tgtEntry;
	for (i = 0; i < lexicon_size; i++) {
		srcEntry = lexicons[0][i];
		tgtEntry = lexicons[1][i];
		for (a = 0; a < layer1_size; a++) {
			syn0s[1][tgtEntry * layer1_size + a] = syn0s[0][srcEntry * layer1_size + a];
		}
	}
}

void TrainModel() {
	long a;
	int lang_id, i;
	pthread_t *mono_pt = malloc(NUM_LANG * num_threads * sizeof(pthread_t)); // 单语言训练线程
	pthread_t *lexicon_pt = malloc(num_threads * sizeof(pthread_t));
	pthread_t *sememe_pt = malloc(num_threads * sizeof(pthread_t));
	pthread_t *matching_t2s_pt = malloc(num_threads * sizeof(pthread_t));
	pthread_t *matching_s2t_pt = malloc(num_threads * sizeof(pthread_t));
	starting_alpha = alpha;

	expTable = malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
	sigmoidTable = malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
	for (i = 0; i < EXP_TABLE_SIZE; i++) {
		// Precompute the exp() table
		expTable[i] = exp((i / (real) EXP_TABLE_SIZE * 2 - 1) * MAX_EXP);
		// Precompute sigmoid f(x) = x / (x + 1)
		sigmoidTable[i] = expTable[i] / (expTable[i] + 1);
	}

	max_train_words = 0;
	for (lang_id = 0; lang_id < NUM_LANG; lang_id++) {
		vocabs[lang_id] = calloc(vocab_max_size, sizeof(struct vocab_word));
		vocab_hashes[lang_id] = calloc(vocab_hash_size, sizeof(int));
		if (read_vocab_files[lang_id][0] != 0) {
			fprintf(stderr, "Reading vocab\n");
			ReadVocab(lang_id);
		} else {
			fprintf(stderr, "Learning Vocab\n");
			LearnVocabFromTrainFile(lang_id);
			fprintf(stderr, "Done learning vocab\n");
		}
		if (save_vocab_files[lang_id][0] != 0) {
			fprintf(stderr, "Saving vocab\n");
			SaveVocab(lang_id);
		}
		if (!learn_vocab_and_quit && output_files[lang_id][0] == 0) { // 没有说只读取词表就可以，但是却没有指定输出文件
			printf("ERROR: No output name specified.\n");
			exit(1);
		}

		fprintf(stderr, "Initializing net..");
		InitNet(lang_id);
		fprintf(stderr, "..done.\n");

		fprintf(stderr, "Initializing unigram table..");
		InitUnigramTable(lang_id);
		fprintf(stderr, "..done.\n");

		if (train_words[lang_id] > max_train_words)
			max_train_words = train_words[lang_id]; // ？？这是啥意思？
	}
	fprintf(stderr, "Loading lexicon\n");
	LoadLexicon();
	fprintf(stderr, "..done.\n");

	//InitLexiconWords();
	if (learn_vocab_and_quit)
		exit(0);

	// 读取义原数组和HowNet词典
	fprintf(stderr, "Reading Sememes\n");
	ReadSememes();
	fprintf(stderr, "... done\n");

	fprintf(stderr, "Reading HowNet\n");
	ReadHowNet();
	fprintf(stderr, "... done\n");

	fprintf(stderr, "Initializing Sememe Embeddings\n");
	InitNetSememe();
	fprintf(stderr, "... done\n");


	pthread_rwlock_init(&lock, NULL); // 初始化读写锁，NULL表示使用缺省的读写锁属性
	start = clock();
	fprintf(stderr, "Starting training.\n");

	for (i = 0; i < NUM_EPOCHS; i++) {
		printf("Epoch = %d\n", i);
		alpha = starting_alpha * (NUM_EPOCHS - i) / NUM_EPOCHS; // 学习率递减
		ALL_MONO_DONE = 0;
		lang_updates[0] = 0;
		lang_updates[1] = 0;

		for (a = 0; a < NUM_LANG * num_threads; a++) {
			pthread_create(&mono_pt[a], NULL, MonoModelThread, (void *) a); // a是当前线程的序号，作为参数传入函数中
		}
		for (a = 0; a < num_threads; a++) {
			pthread_create(&lexicon_pt[a], NULL, LexiconThread, (void *) a);
		}
		for (a = 0; a < num_threads; a++) {
			pthread_create(&sememe_pt[a], NULL, SememeThread, (void *) a);
		}

		for (a = 0; a < num_threads; a++) {
			pthread_create(&matching_t2s_pt[a], NULL, MatchingT2SThread, (void *) a);
		}
		for (a = 0; a < num_threads; a++) {
			pthread_create(&matching_s2t_pt[a], NULL, MatchingS2TThread, (void *) a);
		}

		for (a = 0; a < NUM_LANG * num_threads; a++)
			pthread_join(mono_pt[a], NULL); // 以阻塞的形式等待该线程结束

		pthread_rwlock_wrlock(&lock);
		ALL_MONO_DONE = 1;
		pthread_rwlock_unlock(&lock);

		for (a = 0; a < num_threads; a++)
			pthread_join(lexicon_pt[a], NULL);
		for (a = 0; a < num_threads; a++)
			pthread_join(sememe_pt[a], NULL);

		for (a = 0; a < num_threads; a++)
			pthread_join(matching_t2s_pt[a], NULL);

		for (a = 0; a < num_threads; a++)
			pthread_join(matching_s2t_pt[a], NULL);
		// Save the word vectors
		for (lang_id = 0; lang_id < NUM_LANG; lang_id++) {
			char save_name[MAX_STRING];
			sprintf(save_name, output_files[lang_id], dump_iters[lang_id]++); // dump_iters用来记录保存的次数
			SaveModel(lang_id, save_name);
		}
		// 保存义原向量
		SaveSememe();
		//reset optimization
		//		long long c, d;
		//		if (adagrad) {
		//			for (lang_id = 0; lang_id < NUM_LANG; lang_id++) {
		//				for (d = 0; d < layer1_size; d++) {
		//					for (c = 0; c < vocab_sizes[lang_id]; c++) {
		//						syn0grads[lang_id][c * layer1_size + d] = 0;
		//						syn1negGrads[lang_id][c * layer1_size + d] = 0;
		//					}
		//				}
		//			}
		//		} else {
		//			alpha = starting_alpha;
		//			word_count_actual = 0;
		//		}
	}
	pthread_rwlock_destroy(&lock);
}

int ArgPos(char *str, int argc, char **argv) {
	int a;
	for (a = 1; a < argc; a++)
		if (!strcmp(str, argv[a])) {
			if (a == argc - 1) {
				printf("Argument missing for %s\n", str);
				exit(1);
			}
			return a;
		}
	return -1;
}

int main(int argc, char **argv) {
	int i, lang_id;
	if (argc == 1) {
		printf("Embedding matching cross-lingual word "
		       "vector estimation toolkit\n\n");
		printf("Options:\n");
		printf("Arguments for training:\n");

		printf("\t-mono-trainN <file>\n");
		printf("\t\tUse monolingual text data for language N from <file> to train\n");

		printf("\t-lexiconN <file>\n");
		printf("\t\tUse lexicon for language N from <file> to train\n"
		       "\t\tEach line is a word in language N.\n"
		       "\t\tLines should match for both languages.\n");

		// new, begin
		printf("\t-sememe <file>\n");
		printf("\t\tRead sememes from <file>\n");

		printf("\t-hownet <file>\n");
		printf("\t\tRead HowNet dictionary from <file>\n");

		printf("\t-sememe-lambda <float>\n");
		printf("\t\tSememe term weight (default = 1)\n");

		printf("\t-save-sememe <file>\n");
		printf("\t\tUse <file> to save the resulting semememe vectors\n");

		// new, end
		printf("\t-outputN <file>\n");
		printf("\t\tUse <file> to save the resulting word vectors for language N\n");

		printf("\t-size <int>\n");
		printf("\t\tSet size of word vectors; default is 100\n");

		printf("\t-window <int>\n");
		printf("\t\tSet max skip length between words; default is 5\n");

		printf("\t-sample <float>\n");
		printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency\n"
		       "\t\tin the training data will be randomly down-sampled; default is "
		       "0 (off), useful value is 1e-5\n");

		printf("\t-negative <int>\n");
		printf("\t\tNumber of negative examples; default is 5, common values are"
		       " 5 - 10 (0 = not used)\n");

		printf("\t-threads <int>\n");
		printf("\t\tUse <int> threads (default 1)\n");

		printf("\t-min-count <int>\n");
		printf("\t\tThis will discard words that appear less than <int> times; "
		       "default is 5\n");

		printf("\t-alpha <float>\n");
		printf("\t\tSet the starting learning rate; default is 0.025\n");

		printf("\t-debug <int>\n");
		printf("\t\tSet the debug mode (default = 2, more info during training)\n");

		printf("\t-binary <int>\n");
		printf("\t\tSave the resulting vectors in binary mode; default is 0 (off)\n");

		printf("\t-save-vocabN <file>\n");
		printf("\t\tThe vocabulary for language N will be saved to <file>\n");

		printf("\t-read-vocabN <file>\n");
		printf("\t\tThe vocabulary for language N will be read from <file>, not "
		       "constructed from the training data\n");

		printf("\t-epochs N\n");
		printf("\t\tTrain for N epochs (default = 1)\n");

		printf("\t-adagrad <int>\n");
		printf("\t\tUse Adagrad adaptive learning rate anealing (default = 1)\n");

		printf("\t-matching-lambda <float>\n");
		printf("\t\tMatching term weight (default = 1)\n");

		printf("\t-lexicon-lambda <float>\n");
		printf("\t\tLexicon term weight (default = 1)\n");

		printf("\t-threshold <float>\n");
		printf("\t\tThreshold for falling back to empty word (default = 0)\n");

		printf("\t-Mstep-iterations <int>\n");
		printf("\t\tNumber of iterations in each M step (default = 1)\n");

		printf("\t-dump-every N\n");
		printf("\t\tSave intermediate embeddings during training every N steps if N>0, else every epoch/N steps\n");

		printf("\t-learn-vocab-and-quit <int>\n");
		printf("\t\tLearn and save vocab only\n");

		printf("\nExample:\n");
		printf("./embeddingMatching -mono-train1 data.e -mono-train2 data.f -lexicon1 "
		       "lexicon.e -lexicon2 lexicon.f -output1 vec.e -output2 vec.f -size 200"
		       " -window 5 -sample 1e-4 -negative 5\n\n");
		return 0;
	}

	for (lang_id = 0; lang_id < NUM_LANG; lang_id++) {
		mono_train_files[lang_id] = calloc(MAX_STRING, sizeof(char));
		lexicon_files[lang_id] = calloc(MAX_STRING, sizeof(char));
		output_files[lang_id] = calloc(MAX_STRING, sizeof(char));
		save_vocab_files[lang_id] = calloc(MAX_STRING, sizeof(char));
		read_vocab_files[lang_id] = calloc(MAX_STRING, sizeof(char));
		lang_updates[lang_id] = 0;
		dump_iters[lang_id] = 0;
	}
	sememe_file = calloc(MAX_STRING, sizeof(char));
	hownet_file = calloc(MAX_STRING, sizeof(char));
	save_sememe_file = calloc(MAX_STRING, sizeof(char));

	if ((i = ArgPos((char *) "-size", argc, argv)) > 0)
		layer1_size = atoi(argv[i + 1]);
	if ((i = ArgPos((char *) "-mono-train1", argc, argv)) > 0)
		strcpy(mono_train_files[0], argv[i + 1]);
	if ((i = ArgPos((char *) "-mono-train2", argc, argv)) > 0)
		strcpy(mono_train_files[1], argv[i + 1]);
	if ((i = ArgPos((char *) "-lexicon1", argc, argv)) > 0)
		strcpy(lexicon_files[0], argv[i + 1]);
	if ((i = ArgPos((char *) "-lexicon2", argc, argv)) > 0)
		strcpy(lexicon_files[1], argv[i + 1]);

	if ((i = ArgPos((char *) "-sememe", argc, argv)) > 0)
		strcpy(sememe_file, argv[i + 1]);
	if ((i = ArgPos((char *) "-hownet", argc, argv)) > 0)
		strcpy(hownet_file, argv[i + 1]);
	if ((i = ArgPos((char *) "-sememe-lambda", argc, argv)) > 0)
		SEMEME_LAMBDA = atof(argv[i + 1]);
	if ((i = ArgPos((char *) "-save-sememe", argc, argv)) > 0)
		strcpy(save_sememe_file, argv[i + 1]);

	if ((i = ArgPos((char *) "-output1", argc, argv)) > 0)
		strcpy(output_files[0], argv[i + 1]);
	if ((i = ArgPos((char *) "-output2", argc, argv)) > 0)
		strcpy(output_files[1], argv[i + 1]);

	if ((i = ArgPos((char *) "-save-vocab1", argc, argv)) > 0)
		strcpy(save_vocab_files[0], argv[i + 1]);
	if ((i = ArgPos((char *) "-read-vocab1", argc, argv)) > 0)
		strcpy(read_vocab_files[0], argv[i + 1]);
	if ((i = ArgPos((char *) "-save-vocab2", argc, argv)) > 0)
		strcpy(save_vocab_files[1], argv[i + 1]);
	if ((i = ArgPos((char *) "-read-vocab2", argc, argv)) > 0)
		strcpy(read_vocab_files[1], argv[i + 1]);
	if ((i = ArgPos((char *) "-cbow", argc, argv)) > 0)
		cbow = atoi(argv[i + 1]);
	if ((i = ArgPos((char *) "-debug", argc, argv)) > 0)
		debug_mode = atoi(argv[i + 1]);
	if ((i = ArgPos((char *) "-binary", argc, argv)) > 0)
		binary = atoi(argv[i + 1]);
	if ((i = ArgPos((char *) "-alpha", argc, argv)) > 0)
		alpha = atof(argv[i + 1]);

	if ((i = ArgPos((char *) "-window", argc, argv)) > 0)
		window = atoi(argv[i + 1]);
	if ((i = ArgPos((char *) "-sample", argc, argv)) > 0)
		sample = atof(argv[i + 1]);
	if ((i = ArgPos((char *) "-negative", argc, argv)) > 0)
		negative = atoi(argv[i + 1]);
	if ((i = ArgPos((char *) "-threads", argc, argv)) > 0)
		num_threads = atoi(argv[i + 1]);
	if ((i = ArgPos((char *) "-min-count", argc, argv)) > 0)
		min_count = atoi(argv[i + 1]);
	if ((i = ArgPos((char *) "-early-stop", argc, argv)) > 0)
		EARLY_STOP = atoi(argv[i + 1]);
	if ((i = ArgPos((char *) "-epochs", argc, argv)) > 0)
		NUM_EPOCHS = atoi(argv[i + 1]);
	if ((i = ArgPos((char *) "-adagrad", argc, argv)) > 0)
		adagrad = atoi(argv[i + 1]);
	if ((i = ArgPos((char *) "-matching-lambda", argc, argv)) > 0)
		MATCHING_LAMBDA = atof(argv[i + 1]);
	if ((i = ArgPos((char *) "-lexicon-lambda", argc, argv)) > 0)
		LEXICON_LAMBDA = atof(argv[i + 1]);
	if ((i = ArgPos((char *) "-threshold", argc, argv)) > 0)
		threshold = atof(argv[i + 1]);
	if ((i = ArgPos((char *) "-Mstep-iterations", argc, argv)) > 0)
		MSTEP_ITER = atoi(argv[i + 1]);
	if ((i = ArgPos((char *) "-dump-every", argc, argv)) > 0)
		dump_every = atoi(argv[i + 1]);
	if ((i = ArgPos((char *) "-learn-vocab-and-quit", argc, argv)) > 0)
		learn_vocab_and_quit = atoi(argv[i + 1]);

	TrainModel();
	return 0;
}

