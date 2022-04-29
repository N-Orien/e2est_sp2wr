# End-to-end Japanese-English Speech-to-text Translation with Spoken-to-Written Style Conversion

[lasertagger](lasertagger) is for spoken-to-written style conversion and [speech_translation](speech_translation) is for speech translation with interactive-attention-based multi-decoder models.

## Spoken-to-Written Style Conversion

You can use the [yml file](lasertagger/exp/lasertagger.yml) to build the conda environment.

Refer to [the example data directory](lasertagger/data_conv) to prepare data for training.

[train.sh](lasertagger/exp/train.sh) and [predict.sh](lasertagger/exp/predict.sh) are the scripts for training and inference.
Please define your own $DATA_DIR, $OUTPUT_DIR, $BERT_BASE_DIR and $INPUT_DIR in the scripts.

If you have the ST data for training ASR-ST Dual Decoder ([example](speech_translation/data_st/dual_sp)), you can use [convert_st_data.sh](lasertagger/exp/convert_st_data.sh) to build the data directories for training Conversion-ST Dual Decoder and Triple Decoder.

## Multi-decoder Speech-to-text Translation with Interactive Attention

You can use the [yml file](speech_translation/joint_asr_st.yml) to build the conda environment.

Please refer to [this repository](https://github.com/formiel/speech-translation) for environment building, data preparation and model training.
Noted that the suffix for languages in our inplementation is different. For dual-decoder, we use "sp" for source language and "en" for target language. For triple-decoder, we use "sp" for spoken-style source language, "wr" for written-style source language and "en" for target language. 
You can refer to [the example data directory](speech_translation/data_st) for data preparation.
