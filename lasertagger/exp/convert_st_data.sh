#!/bin/bash

sp_path=../../speech_translation/data_st/dual_sp/data
temp_path=temp
wr_path=../../speech_translation/data_st/dual_wr/data
triple_path=../../speech_translation/data_st/triple/data

for setname in dev test train; do
	mkdir -p ${wr_path}/${setname}
	mkdir -p ${triple_path}/${setname}
	mkdir -p ${wr_path}/${setname}/txt
	mkdir -p ${triple_path}/${setname}/txt
	ln -s ${sp_path}/${setname}/wav ${wr_path}/${setname}/
	ln -s ${sp_path}/${setname}/wav ${triple_path}/${setname}/
done

for setname in dev test train; do
	awk '{print $0"\t"NR}' ${temp_path}/${setname}.sp.zen > ${temp_path}/${setname}.tsv
done

bash predict.sh

for setname in dev test train; do
	cut -f 2 -d "	" ${temp_path}/${setname}_pred.tsv > ${temp_path}/${setname}.wr
	cp ${temp_path}/${setname}.wr ${wr_path}/${setname}/txt/${setname}.sp
	cp ${temp_path}/${setname}.wr ${triple_path}/${setname}/txt/
	cp ${sp_path}/${setname}/txt/${setname}.en ${wr_path}/${setname}/txt/${setname}.en
	cp ${sp_path}/${setname}/txt/${setname}.en ${triple_path}/${setname}/txt/${setname}.en
	cp ${sp_path}/${setname}/txt/${setname}.yaml ${wr_path}/${setname}/txt/${setname}.yaml
	cp ${sp_path}/${setname}/txt/${setname}.yaml ${triple_path}/${setname}/txt/${setname}.yaml
done

