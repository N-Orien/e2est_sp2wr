MAIN_ROOT=/mnt/elm/zd-yang/joint-asr-st/speech-translation
KALDI_ROOT=$MAIN_ROOT/tools/kaldi

[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sctk/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

export PATH=$MAIN_ROOT/tools/moses/scripts/tokenizer/:$MAIN_ROOT/tools/moses/scripts/generic/:$PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$MAIN_ROOT/tools/chainer_ctc/ext/warp-ctc/build
if [ -e $MAIN_ROOT/tools/venv/etc/profile.d/conda.sh ]; then
    source $MAIN_ROOT/tools/venv/etc/profile.d/conda.sh && conda deactivate && conda activate
elif [ -e $MAIN_ROOT/tools/venv/bin/activate ]; then
    source $MAIN_ROOT/tools/venv/bin/activate
fi
export PATH=$MAIN_ROOT/utils:$MAIN_ROOT/espnet/bin:$PATH

export OMP_NUM_THREADS=1

# check extra module installation
if ! which tokenizer.perl > /dev/null; then
    echo "Error: it seems that moses is not installed." >&2
    echo "Error: please install moses as follows." >&2
    echo "Error: cd ${MAIN_ROOT}/tools && make moses.done" >&2
    return 1
fi

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8
