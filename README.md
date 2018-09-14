# Cross-lingual Lexical Sememe Prediction
This is the open-source code of the EMNLP 2018 paper **Cross-lingual Lexical Sememe Prediction**.

Sememes are defined as the minimum semantic units of human languages. As important knowledge sources, sememe-based linguistic knowledge bases have been widely used in many NLP tasks. However, most languages still do not have sememe-based linguistic knowledge bases. Thus we present a task of cross-lingual lexical sememe prediction (**CLSP**), aiming to automatically predict sememes for words in other languages. We propose a novel framework to model correlations between sememes and multi-lingual words in low-dimensional semantic space for sememe prediction. Experimental results on real-world datasets show that our proposed model achieves consistent and significant improvements as compared to baseline methods in cross-lingual sememe prediction.
## Usage
	
	bash run.sh
	
To change the training corpus, please just switch the `-mono-train1` and `-mono-train2` parameters in `bash.sh`. Notice that `lang1` refers to the source language and `lang2` refers to the target language.
## Datasets
<table border="1">
	<tr>
		<td align="center">Process</td>
		<td align="center">Type</td>
		<td align="center">Source</td>
		<td align="center">Target</td>
	</tr>
	<tr>
		<td align="center"  rowspan="3">Training</td>
		<td align="center">Corpus</td>
		<td align="center">Sogou-T</td>
		<td align="center">Wikipedia</td>
	</tr>
	<tr>
		<td align="center">Seed Lexicon</td>
		<td align="center" colspan="2"> Google Translate API</td>
	</tr>
	<tr>
		<td align="center">Sememe-based KB</td>
		<td align="center">HowNet_zh</td>
		<td align="center">-</td>
	</tr>
	<tr>
		<td align="center" = rowspan="4">Testing</td>
		<td align="center">Sememe Prediction</td>
		<td align="center">-</td>
		<td align="center">HowNet_en</td>
	</tr>
	<tr>
		<td align="center">Bilingual Lexicon Induction</td>
		<td align="center" colspan="2">Chinese-English Translation Lexicon 3.0 Version</td>
	</tr>
	<tr>
		<td align="center"  rowspan="2"> Word Similarity Computation</td>
		<td align="center">Wordsim-240</td>
		<td align="center">WordSim-353</td>
	</tr>
	<tr>
		<td align="center">WordSim-297</td>
		<td align="center">SimLex-999</td>
	</tr>
</table>


## Cite

If the codes or datasets help you, please cite the following paper:

	@InProceedings{qi2018cross,
	  Title      = {Cross-lingual lexical sememe prediction},
	  Author     = {Qi, Fanchao and Lin, Yankai and Sun, Maosong and Zhu, Hao and Xie, Ruobing and Liu, Zhiyuan},
	  Booktitle  = {Proceedings of EMNLP},
	  Year       = {2018},
	}
