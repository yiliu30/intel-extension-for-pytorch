import intel_extension_for_pytorch
from intel_extension_for_pytorch.quantization.fp8 import (
    fp8_autocast,
    DelayedScaling,
    Format,
    prepare_fp8,
)

import torch

import time
import logging


def dump_elapsed_time(customized_msg=""):
    """Get the elapsed time for decorated functions.

    Args:
        customized_msg (string, optional): The parameter passed to decorator. Defaults to None.
    """

    def f(func):
        def fi(*args, **kwargs):
            start = time.time()
            res = func(*args, **kwargs)
            end = time.time()
            print(
                "%s elapsed time: %s ms"
                % (
                    customized_msg if customized_msg else func.__qualname__,
                    round((end - start) * 1000, 2),
                )
            )
            return res

        return fi

    return f


import os


# @dump_elapsed_time()
# @torch.no_grad()
# def batch_gen_text(model, tokenizer, msg="", prompt="What's AI?", max_tokens = 20, device="cpu"):
#     model = model.to(device)
#     inputs = tokenizer(prompt, return_tensors="pt") #, padding=True, truncation=True)
#     # inputs = move_data_to_device(inputs, device)
#     new_tokens = model.generate(**inputs.to(device), max_length=max_tokens, cache_implementation="static")
#     text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
#     for i, t in enumerate(text):
#         print(f"Generated text ({msg}): {t}")

# model_name = "/home/yliu7/Llama-2-7b-chat-hf"
# import transformers
# model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
# model_dtype = next(model.parameters()).dtype
# print(f"model dtype: {model_dtype}")
# tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
# # tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# prompt="What's AI?"
# inputs = tokenizer(prompt, return_tensors="pt") #, padding=True, truncation=True)
# input_ids = inputs.input_ids
# model = prepare_fp8(model)
# with fp8_autocast(enabled=False, calibrating=True, fp8_recipe=DelayedScaling(fp8_format=Format.E4M3), device="cpu"):
#     # output = fp8_model(input_data)
#     model(input_ids)
# # breakpoint()
# print(model)

# # model = torch.compile(model)
# for i in range(10):
#     batch_gen_text(model, tokenizer)
#     # batch_gen_text(model, tokenizer)
#     # batch_gen_text(model, tokenizer)

device = torch.device("xpu")
model = torch.nn.Linear(10, 10).to(device)
input_data = torch.randn(1, 10).to(device)
float_out = model(input_data)
fp8_model = prepare_fp8(model)
with fp8_autocast(
    enabled=False, calibrating=True, fp8_recipe=DelayedScaling(fp8_format=Format.E4M3)
):
    output = fp8_model(input_data)

# Error
# fp8_model = torch.compile(fp8_model, backend="ipex")


output = fp8_model(input_data)

print(f"output: {output}, dtype: {output.dtype}")
print(f"float output: {float_out}")


# """
# w/o fp8_autocast
# Generated text (): What's AI?
# batch_gen_text elapsed time: 295256.62 ms
# Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.


# w/o fp8_autocast
# (hqq) (base) [yliu7@aia-sdp-spr-117706 fp8]$ python example.py
# Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00,  2.71it/s]
# model dtype: torch.float16
# Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
# Generated text (): What's AI?

# Artificial intelligence (AI) refers to the development of computer systems able to perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation. AI systems use algorithms and machine learning techniques to analyze data, learn from it, and make decisions or predictions based on that data.

# There are several types of AI, including:

# 1. Narrow or weak AI
# batch_gen_text elapsed time: 16000.43 ms
# Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
# Generated text (): What's AI?

# Artificial intelligence (AI) refers to the development of computer systems able to perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation. AI systems use algorithms and machine learning techniques to analyze data, learn from it, and make decisions or predictions based on that data.

# There are several types of AI, including:

# 1. Narrow or weak AI
# batch_gen_text elapsed time: 14468.42 ms
# Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
# Generated text (): What's AI?

# Artificial intelligence (AI) refers to the development of computer systems able to perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation. AI systems use algorithms and machine learning techniques to analyze data, learn from it, and make decisions or predictions based on that data.

# There are several types of AI, including:

# 1. Narrow or weak AI
# batch_gen_text elapsed time: 12835.57 ms
# Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
# Generated text (): What's AI?

# Artificial intelligence (AI) refers to the development of computer systems able to perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation. AI systems use algorithms and machine learning techniques to analyze data, learn from it, and make decisions or predictions based on that data.

# There are several types of AI, including:

# 1. Narrow or weak AI
# batch_gen_text elapsed time: 14261.04 ms
# Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
# Generated text (): What's AI?

# Artificial intelligence (AI) refers to the development of computer systems able to perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation. AI systems use algorithms and machine learning techniques to analyze data, learn from it, and make decisions or predictions based on that data.

# There are several types of AI, including:

# 1. Narrow or weak AI
# batch_gen_text elapsed time: 14475.32 ms
# Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
# Generated text (): What's AI?

# Artificial intelligence (AI) refers to the development of computer systems able to perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation. AI systems use algorithms and machine learning techniques to analyze data, learn from it, and make decisions or predictions based on that data.

# There are several types of AI, including:

# 1. Narrow or weak AI
# batch_gen_text elapsed time: 14258.79 ms
# Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
# Generated text (): What's AI?

# Artificial intelligence (AI) refers to the development of computer systems able to perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation. AI systems use algorithms and machine learning techniques to analyze data, learn from it, and make decisions or predictions based on that data.

# There are several types of AI, including:

# 1. Narrow or weak AI
# batch_gen_text elapsed time: 13662.58 ms
# Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
# Generated text (): What's AI?

# Artificial intelligence (AI) refers to the development of computer systems able to perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation. AI systems use algorithms and machine learning techniques to analyze data, learn from it, and make decisions or predictions based on that data.

# There are several types of AI, including:

# 1. Narrow or weak AI
# batch_gen_text elapsed time: 14111.66 ms
# Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
# Generated text (): What's AI?

# Artificial intelligence (AI) refers to the development of computer systems able to perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation. AI systems use algorithms and machine learning techniques to analyze data, learn from it, and make decisions or predictions based on that data.

# There are several types of AI, including:

# 1. Narrow or weak AI
# batch_gen_text elapsed time: 13967.98 ms
# Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.


# """
