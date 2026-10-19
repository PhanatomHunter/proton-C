#!/bin/sh
export PYTHONPATH="$(pwd)/code"
cd "$(dirname "$0")"
python3 tests_suite_full/test_decoder_safety.py
python3 tests2_decoder_safety/test_decoder_safety.py
python3 tests_suite_full/test_seek_consistency.py
python3 tests_suite_full/test_kernels.py
python3 tests_suite_full/test_native.py
python3 tests_suite_full/test_rans.py
python3 tests_suite_full/fuzz_decoder.py
