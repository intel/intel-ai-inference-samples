import transformers
from transformers import GPTJForCausalLM, AutoTokenizer
import torch
import time
import json
import pathlib
import argparse
import numpy as np
from itertools import chain

# args
parser = argparse.ArgumentParser("GPT-J generation script", add_help=False)
parser.add_argument(
    "-m",
    "--model-id",
    type=str,
    choices=[
        "EleutherAI/gpt-j-6B",
    ],
    default="EleutherAI/gpt-j-6B",
)
parser.add_argument(
    "--dtype",
    type=str,
    help="bfloat16 or float32",
    default="bfloat16",
)
parser.add_argument(
    "--max-new-tokens", default=32, type=int, help="output max new tokens"
)
parser.add_argument("--greedy", action="store_true")
parser.add_argument("--ipex", action="store_true")
parser.add_argument("--jit", action="store_true")
parser.add_argument("--input-tokens", default="32", type=str)
parser.add_argument("--prompt", default=None, type=str)
parser.add_argument("--num-iter", default=100, type=int, help="num iter")
parser.add_argument("--num-warmup", default=10, type=int, help="num warmup")
parser.add_argument("--batch-size", default=1, type=int, help="batch size")
parser.add_argument("--token-latency", action="store_true")
parser.add_argument("--use-tpp", action="store_true")
args = parser.parse_args()
print(args)


# device
device = torch.device('cpu')
# import extension
if args.ipex:
    import intel_extension_for_pytorch as ipex

# dtype
if args.dtype == "bfloat16":
    amp_enabled = True
    amp_dtype = torch.bfloat16
else:
    amp_enabled = False
    amp_dtype = torch.float32

# generate args
if args.greedy:
    generate_kwargs = dict(do_sample=False, temperature=0.9)
else:
    generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=4)
if args.jit:
    torch._C._jit_set_texpr_fuser_enabled(False)
    generate_kwargs["jit"] = True
if args.token_latency:
    generate_kwargs["token_latency"] = True
if args.use_tpp:
    generate_kwargs["use_tpp"] = True

# load model
model_id = args.model_id
model = GPTJForCausalLM.from_pretrained(
    model_id, low_cpu_mem_usage=True, return_dict=not args.jit, torch_dtype=amp_dtype
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = model.eval().to(device)
model = model.to(memory_format=torch.channels_last)

# to ipex
if args.ipex:
    model = ipex.optimize(model.eval(), dtype=amp_dtype, inplace=True)

# use tpp
if args.use_tpp:
    from tpp_pytorch_extension.llm.fused_gptj_infer import FixGPTJBlock, block
    for m in model.modules():
        if isinstance(m, transformers.models.gptj.modeling_gptj.GPTJBlock):
            # FixGPTJBlock(m, 16, 16, torch.bfloat16)
            FixGPTJBlock(m, 16, 64, amp_dtype)
    # block(model)

# input prompt
current_path = pathlib.Path(__file__).parent.resolve()
with open(str(current_path) + "/prompt.json") as f:
    prompt_pool = json.load(f)
if args.prompt is not None:
    prompt = args.prompt
elif args.input_tokens in prompt_pool:
    prompt = prompt_pool[args.input_tokens]
else:
    raise SystemExit("[ERROR] Plese use --prompt if want to use custom input.")

input_size = tokenizer(prompt, return_tensors="pt").input_ids.size(dim=1)
print("---- Prompt size:", input_size)

# start
total_time = 0.0
num_iter = args.num_iter
num_warmup = args.num_warmup
prompt = [prompt] * args.batch_size
total_list = []
with torch.inference_mode(), torch.no_grad(), torch.autocast(
    device_type='cpu',
    enabled=amp_enabled,
    dtype=amp_dtype if amp_enabled else None,
):
    for i in range(num_iter):
        tic = time.time()
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        output = model.generate(
            input_ids, max_new_tokens=args.max_new_tokens, **generate_kwargs
        )
        gen_ids = output[0] if args.token_latency else output
        gen_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        toc = time.time()
        print(gen_text, flush=True)
        if i >= num_warmup:
            total_time += toc - tic
            if args.token_latency:
                total_list.append(output[1])

print("\n", "-" * 10, "Summary:", "-" * 10)
latency = total_time / (num_iter - num_warmup)
print("Inference latency: %.3f sec." % latency)
if args.token_latency:
    first_latency = np.mean([x[0] for x in total_list])
    average_2n = list(chain(*[x[1:] for x in total_list]))
    average_2n.sort()
    average_2n_latency = np.mean(average_2n)
    p90_latency = average_2n[int(len(average_2n) * 0.9)]
    p99_latency = average_2n[int(len(average_2n) * 0.99)]
    print("First token average latency: %.3f sec." % first_latency)
    print("Average 2... latency: %.3f sec." % average_2n_latency)
    print("P99 2... latency: %.3f sec." % p90_latency)
