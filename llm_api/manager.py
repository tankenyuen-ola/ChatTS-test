# llm/manager.py
import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoProcessor
import sys
import ChatTS.chatts.vllm.chatts_vllm


_global_llm_instance = None
_global_tokenizer_instance = None
_global_processor_instance = None

def initialize_llm(model_path: str):
    print(model_path)
    global _global_llm_instance, _global_tokenizer_instance, _global_processor_instance

    # Only return if *all three* are already set
    if (_global_llm_instance is not None
        and _global_tokenizer_instance is not None
        and _global_processor_instance is not None):
        return _global_llm_instance, _global_tokenizer_instance, _global_processor_instance

    try:
        _global_llm_instance = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.9,
            max_model_len=8192,
            limit_mm_per_prompt={"timeseries": 50},
            max_num_seqs=128,
            dtype='half'
        )
        _global_tokenizer_instance = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=True, device_map='auto'
        )
        _global_processor_instance = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            tokenizer=_global_tokenizer_instance
        )
    except Exception as e:
        # You can choose to log or re-raise
        print(f"[ERROR] initialize_llm failed: {e}", file=sys.stderr)
        return None, None, None

    return _global_llm_instance, _global_tokenizer_instance, _global_processor_instance

def get_llm_instance():
    return _global_llm_instance, _global_tokenizer_instance, _global_processor_instance

def release_llm():
    global _global_llm_instance, _global_tokenizer_instance, _global_processor_instance
    _global_llm_instance     = None
    _global_tokenizer_instance = None
    _global_processor_instance = None