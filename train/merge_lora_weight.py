from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
import argparse
import shutil
import os
import torch

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True, 
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--adapter_model_path", type=str, required=True, help="Path to adapter model")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output model")
    parser.add_argument("--save_dtype", type=str, choices=['bf16', 'fp32', 'fp16'], 
                        default='fp32', help="In which dtype to save, fp32, bf16 or fp16.")
    args = parser.parse_args()

    name2dtype = {'bf16': torch.bfloat16, 'fp32': torch.float32, 'fp16': torch.float16}

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path, device_map='cpu', 
        trust_remote_code=True, torch_dtype=name2dtype[args.save_dtype]
    )
    model = PeftModel.from_pretrained(model, args.adapter_model_path, trust_remote_code=True)

    model = model.merge_and_unload()

    model.save_pretrained(args.output_path, safe_serialization=False)

    # Copy tokenizer, config and other non-weight files from base model
    # Skip model weight files (.safetensors, .bin, .pt) and index files
    _SKIP_SUFFIXES = ('.safetensors', '.bin', '.pt', '.pth')
    _SKIP_NAMES = {'model.safetensors.index.json', 'pytorch_model.bin.index.json'}

    for fname in os.listdir(args.base_model_path):
        src = os.path.join(args.base_model_path, fname)
        if not os.path.isfile(src):
            continue
        if fname in _SKIP_NAMES or fname.endswith(_SKIP_SUFFIXES):
            continue
        dst = os.path.join(args.output_path, fname)
        if not os.path.exists(dst):
            shutil.copy(src, dst)
            print(f'Copied {fname}')

    print(f'Merged model weight is saved to {args.output_path}')
    
if __name__ == "__main__":
    main()
