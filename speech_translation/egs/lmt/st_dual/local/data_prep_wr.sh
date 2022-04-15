#!/bin/bash

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

export LC_ALL=C

. utils/parse_options.sh || exit 1;

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <src-dir>"
    echo "e.g.: $0 dataset_path"
    exit 1;
fi

for set in train dev test; do
    src=$1/data/${set}
    dst=data/local/${set}

    [ ! -d ${src} ] && echo "$0: no such directory ${src}" && exit 1;

    wav_dir=${src}/wav
    trans_dir=${src}/txt
    yml=${trans_dir}/${set}.yaml
    en=${trans_dir}/${set}.en
    sp=${trans_dir}/${set}.sp
    wr=${trans_dir}/${set}.wr


    mkdir -p ${dst} || exit 1;

    [ ! -d ${wav_dir} ] && echo "$0: no such directory ${wav_dir}" && exit 1;
    [ ! -d ${trans_dir} ] && echo "$0: no such directory ${trans_dir}" && exit 1;
    [ ! -f ${yml} ] && echo "$0: expected file ${yml} to exist" && exit 1;
    [ ! -f ${en} ] && echo "$0: expected file ${en} to exist" && exit 1;
    [ ! -f ${sp} ] && echo "$0: expected file ${sp} to exist" && exit 1;
    [ ! -f ${wr} ] && echo "$0: expected file ${wr} to exist" && exit 1;


    wav_scp=${dst}/wav.scp; [[ -f "${wav_scp}" ]] && rm ${wav_scp}
    trans_en=${dst}/text.en; [[ -f "${trans_en}" ]] && rm ${trans_en}
    trans_sp=${dst}/text.sp; [[ -f "${trans_sp}" ]] && rm ${trans_sp}
    trans_wr=${dst}/text.wr; [[ -f "${trans_wr}" ]] && rm ${trans_wr}
    utt2spk=${dst}/utt2spk; [[ -f "${utt2spk}" ]] && rm ${utt2spk}
    spk2utt=${dst}/spk2utt; [[ -f "${spk2utt}" ]] && rm ${spk2utt}
    segments=${dst}/segments; [[ -f "${segments}" ]] && rm ${segments}

    # error check
    n=$(cat ${yml} | grep duration | wc -l)
    n_en=$(cat ${en} | wc -l)
    n_sp=$(cat ${sp} | wc -l)
    n_wr=$(cat ${wr} | wc -l)
    [ ${n} -ne ${n_en} ] && echo "Warning: expected ${n} en data files, found ${n_en}" && exit 1;
    [ ${n} -ne ${n_sp} ] && echo "Warning: expected ${n} sp data data files, found ${n_sp}" && exit 1;
    [ ${n} -ne ${n_wr} ] && echo "Warning: expected ${n} wr data data files, found ${n_wr}" && exit 1;


    # (1a) Transcriptions and translations preparation
    # make basic transcription file (add segments info)
    cp ${yml} ${dst}/.yaml0
    grep duration ${dst}/.yaml0 > ${dst}/.yaml1
    awk '{
        duration=$3; offset=$5; spkid=$7;
        gsub(",","",duration);
        gsub(",","",offset);
        gsub(",","",spkid);
	gsub("spk.","",spkid);
        duration=sprintf("%.7f", duration);
        if ( duration < 0.2 ) extendt=sprintf("%.7f", (0.2-duration)/2);
        else extendt=0;
        offset=sprintf("%.7f", offset);
        startt=offset-extendt;
        endt=offset+duration+extendt;
        printf("kyodai_jimaku_%s_%07.0f_%07.0f\n", spkid, int(1000*startt+0.5), int(1000*endt+0.5));
    }' ${dst}/.yaml1 > ${dst}/.yaml2
    # NOTE: Extend the lengths of short utterances (< 0.2s) rather than exclude them

    cp ${en} ${dst}/en.org
    cp ${sp} ${dst}/sp.org
    cp ${wr} ${dst}/wr.org

    for lang in sp wr en; do
        # normalize punctuation
        normalize-punctuation.perl -l ${lang} < ${dst}/${lang}.org > ${dst}/${lang}.norm

        # lowercasing
        lowercase.perl < ${dst}/${lang}.norm > ${dst}/${lang}.norm.lc
        cp ${dst}/${lang}.norm ${dst}/${lang}.norm.tc

        # remove punctuation
        local/remove_punctuation.pl < ${dst}/${lang}.norm.lc > ${dst}/${lang}.norm.lc.rm

        # tokenization
        tokenizer.perl -l ${lang} -q < ${dst}/${lang}.norm.tc > ${dst}/${lang}.norm.tc.tok
        tokenizer.perl -l ${lang} -q < ${dst}/${lang}.norm.lc > ${dst}/${lang}.norm.lc.tok
        tokenizer.perl -l ${lang} -q < ${dst}/${lang}.norm.lc.rm > ${dst}/${lang}.norm.lc.rm.tok

        paste -d " " ${dst}/.yaml2 ${dst}/${lang}.norm.tc.tok | sort > ${dst}/text.tc.${lang}
        paste -d " " ${dst}/.yaml2 ${dst}/${lang}.norm.lc.tok | sort > ${dst}/text.lc.${lang}
        paste -d " " ${dst}/.yaml2 ${dst}/${lang}.norm.lc.rm.tok | sort > ${dst}/text.lc.rm.${lang}

        # save original and cleaned punctuation
        lowercase.perl < ${dst}/${lang}.org | text2token.py -s 0 -n 1 | tr " " "\n" \
            | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' > ${dst}/punctuation.${lang}
        lowercase.perl < ${dst}/${lang}.norm.tc | text2token.py -s 0 -n 1 | tr " " "\n" \
            | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' > ${dst}/punctuation.clean.${lang}
    done


    # error check
    n=$(cat ${dst}/.yaml2 | wc -l)
    n_en=$(cat ${dst}/en.norm.tc.tok | wc -l)
    n_sp=$(cat ${dst}/sp.norm.tc.tok | wc -l)
    n_wr=$(cat ${dst}/wr.norm.tc.tok | wc -l)
    [ ${n} -ne ${n_en} ] && echo "Warning: expected ${n} data data files, found ${n_en}" && exit 1;
    [ ${n} -ne ${n_sp} ] && echo "Warning: expected ${n} data data files, found ${n_sp}" && exit 1;
    [ ${n} -ne ${n_wr} ] && echo "Warning: expected ${n} data data files, found ${n_wr}" && exit 1;


    # (1c) Make segments files from transcript
    #segments file format is: utt-id start-time end-time, e.g.:
    #ku_00001_0003501_0003684 ku_0001 003.501 0003.684
    awk '{
        segment=$1; split(segment,S,"[_]");
        spkid=S[1] "_" S[2] "_" S[3] "_" S[4] "_" S[5]; startf=S[6]; endf=S[7];
        printf("%s %s %.2f %.2f\n", segment, spkid, startf/1000, endf/1000);
    }' < ${dst}/text.tc.en | uniq | sort > ${dst}/segments

    awk '{
        segment=$1; split(segment,S,"[_]");
        spkid=S[1] "_" S[2] "_" S[3] "_" S[4] "_" S[5];
        printf("%s cat '${wav_dir}'/%s.wav |\n", spkid, spkid);
    }' < ${dst}/text.tc.en | uniq | sort > ${dst}/wav.scp

    awk '{
        segment=$1; split(segment,S,"[_]");
        spkid=S[1] "_" S[2] "_" S[3] "_" S[4] "_" S[5]; print $1 " " spkid
    }' ${dst}/segments | uniq | sort > ${dst}/utt2spk

    cat ${dst}/utt2spk | utils/utt2spk_to_spk2utt.pl | sort > ${dst}/spk2utt

    # error check
    n_en=$(cat ${dst}/text.tc.en | wc -l)
    n_sp=$(cat ${dst}/text.tc.sp | wc -l)
    n_wr=$(cat ${dst}/text.tc.wr | wc -l)
    [ ${n_sp} -ne ${n_wr} ] && echo "Warning: expected ${n_sp} wr data files, found ${n_wr}" && exit 1;
    [ ${n_sp} -ne ${n_en} ] && echo "Warning: expected ${n_sp} en data files, found ${n_en}" && exit 1;

    # Copy stuff into its final locations [this has been moved from the format_data script]
    mkdir -p data/${set}

    # remove duplicated utterances (the same offset)
    echo "remove duplicate lines..."
    cut -d ' ' -f 1 ${dst}/text.tc.en | sort | uniq -c | sort -n -k1 -r | grep -v '1 kyodai_jimaku' \
        | sed 's/^[ \t]*//' > ${dst}/duplicate_lines
    cut -d ' ' -f 1 ${dst}/text.tc.en | sort | uniq -c | sort -n -k1 -r | grep '1 kyodai_jimaku' \
        | cut -d '1' -f 2- | sed 's/^[ \t]*//' > ${dst}/reclist
    reduce_data_dir.sh ${dst} ${dst}/reclist data/${set}
    for l in en sp wr; do
        for case in tc lc lc.rm; do
            cp ${dst}/text.${case}.${l} data/${set}/text.${case}.${l}
        done
    done
    utils/fix_data_dir.sh --utt_extra_files \
        "text.tc.en text.lc.en text.lc.rm.en text.tc.sp text.lc.sp text.lc.rm.sp text.tc.wr text.lc.wr text.lc.rm.wr" \
        data/${set}

    # error check
    n_seg=$(cat data/${set}/segments | wc -l)
    n_text=$(cat data/${set}/text.tc.en | wc -l)
    [ ${n_seg} -ne ${n_text} ] && echo "Warning: expected ${n_seg} data data files, found ${n_text}" && exit 1;

    echo "$0: successfully prepared data in ${dst}"
done
