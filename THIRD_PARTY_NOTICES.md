# Third-party notices

This file identifies source code, datasets and model dependencies. Their original licenses remain applicable; inclusion here does not relicense them. Generated checkpoints and output artifacts are not claimed to exist merely because their dependencies are listed.

## Reference FastSpeech 2 and bundled HiFi-GAN

Reference: [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2), commit `d4e79eb52e8b01d24703b2dfc0385544092958f3`.

- FastSpeech2 code: MIT, Copyright (c) 2020 Chung-Ming Chien. Full license retained at `vendor/FastSpeech2/LICENSE`.
- Bundled HiFi-GAN code: MIT, Copyright (c) 2020 Jungil Kong. Full license retained at `vendor/FastSpeech2/hifigan/LICENSE`.
- The original reference modules remain vendored. Project adapters reside separately in `tts/`.
- Bundled pretrained universal HiFi-GAN is a waveform decoder dependency; it does not count as either assignment-trained FastSpeech2 acoustic model. Do not confuse vocoder pretraining with assignment acoustic-model training.

Cite Yi Ren et al., *FastSpeech 2: Fast and High-Quality End-to-End Text to Speech*, ICLR 2021, [paper](https://arxiv.org/abs/2006.04558); and Jungil Kong et al., *HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis*, NeurIPS 2020, [paper](https://arxiv.org/abs/2010.05646).

## Svarah

[AI4Bharat/Svarah](https://huggingface.co/datasets/ai4bharat/Svarah) declares [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).

Attribution: Tahir Javed, Sakshi Joshi, Vignesh Nagarajan, Sai Sundaresan, Janki Nawale, Abhigyan Raman, Kaushal Santosh Bhogale, Pratyush Kumar and Mitesh M. Khapra. *Svarah: Evaluating English ASR Systems on Indian Accents*, INTERSPEECH 2023, pp. 5087–5091. [Paper](https://arxiv.org/abs/2305.15760).

Changes include import, filtering, profile-based conditioning, and new assignment-specific partitions. No original dataset audio is bundled in the public project. Authorization/access requirements remain the reproducer's responsibility. No dataset-author endorsement is implied.

## IndicTTS Telugu

[SPRINGLab/IndicTTS_Telugu](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu) derives from the IIT Madras IndicTTS corpus and points to the [IITM license](https://www.iitm.ac.in/donlab/indictts/downloads/license.pdf). Retain this required notice:

> COPYRIGHT 2016 TTS Consortium, TDIL, Meity represented by Hema A Murthy & S Umesh, DEPARTMENT OF Computer Science and Engineering and Electrical Engineering, IIT Madras. ALL RIGHTS RESERVED

Acknowledge the Speech Technology Consortium, Department of Computer Science and Engineering and Electrical Engineering, IIT Madras, Bhashini/MeitY, Prof. Hema A Murthy and Prof. S Umesh.

The original license server timed out at audit time. The inspected EULA came from a [third-party mirror of the document](https://huggingface.co/k-m-irfan/Fastspeech2_HS_Flask_API/blob/main/license.pdf); byte identity against the official server has not been established. Its section 2.1 allows copying/modifying data and free derivative distribution; it also contains notice-retention and downstream-sale restrictions. Do not replace the corpus license with a generic MIT or CC BY claim. Original corpus audio is not included in this project.

## Alignment and pronunciation dependencies

- [Bournemouth Forced Aligner](https://github.com/tabahi/bournemouth-forced-aligner), installed package `bournemouth-forced-aligner==2.0.1`: package metadata declares GPLv3.
- [Tabahi/CUPE-3-en](https://huggingface.co/Tabahi/CUPE-3-en), revision `cf87be29456f00f71bcfcd0f1c0e37d48e719e33`.
- [Tabahi/CUPE-3-mls7](https://huggingface.co/Tabahi/CUPE-3-mls7), revision `7c1cfc128fdd9799b247b912c264822d8efcf993`: model card declares AGPL-3.0. This model has no Telugu training data; Telugu alignment is zero-shot and needs boundary-quality review.
- [Phonemizer](https://github.com/bootphon/phonemizer): GPLv3-or-later, as stated in its installed package license.
- [eSpeak NG](https://github.com/espeak-ng/espeak-ng): use according to its GPL license and preserve its notices when distributing binaries.

Alignment and phonemizer models/software are fetched or installed as dependencies, not relicensed as part of the reference implementation. Their copyleft terms must be considered for the complete distribution. The list above is not a claim that all dependencies share the MIT license of FastSpeech2. Preserve package licenses and source availability obligations when packaging them. A complete dependency lock/requirements file supplements this notice.
