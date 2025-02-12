#/bin/bash
git clone https://github.com/ggerganov/llama.cpp.git .
python3 llama.cpp/convert_hf_to_gguf.py ~/llm_test2/output_merge/Qwen2.5/ --outtype f16 --outfile ~/llm_test2/output_gguf/Qwen2.5/
sudo apt  install cmake
cmake -B build
cmake --build build --config Release
./llama-cli -m ~/llm_test2/output_gguf/Qwen2.5/Qwen2.5-494M-F16.gguf
