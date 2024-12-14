from entropixing.llama_cpp_impl import generate_response
from llama_cpp import Llama
from datasets import load_dataset

PROMPT_FORMAT = """以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。

### 指示:
{}
 
### 応答:
"""

ENSEMBLE_FORMAT = """以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。

### 指示:
質問と3つの回答案を与えるので、それらを元に最も優れた回答を作成してください。複数の回答案を元にしていることを悟られずに、回答のみ応答してください。
質問: {}
回答案1: {}
回答案2: {}
回答案3: {}
 
### 応答:
"""

ENSEMBLE_2_FORMAT = """以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。

### 指示:
質問と3つの回答案を与えます。それらのうち最も優れた回答を元により優れた回答を作成してください。回答のみ応答してください。
質問: {}
回答案1: {}
回答案2: {}
回答案3: {}
 
### 応答:
"""

def generate(weights, inputs, max_length) -> str:
    it = generate_response(
        weights,
        inputs,
        max_length,
        stop=["<bos>"],
    )
    text = ""
    for token in it:
        print(token, end="")
        text += token
    print()
    return text.strip()

def answer_function(item, weights, max_length):
    input_prompt = PROMPT_FORMAT.format(item["input"])
    output1 = generate(weights, input_prompt, max_length)
    output2 = generate(weights, input_prompt, max_length)
    output3 = generate(weights, input_prompt, max_length)
    emsemble_prompt = ENSEMBLE_2_FORMAT.format(item["input"], output1, output2, output3)
    item["output"] = generate(weights, emsemble_prompt, max_length)
    return item


def main():
    from argparse import ArgumentParser

    global device
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True, default="./model.gguf")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--context_length", type=int, default=2048)
    parser.add_argument("--ngl", type=int, default=0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--flash_attn", action="store_true", default=False)
    parser.add_argument("--data_file", type=str, required=True)
    args = parser.parse_args()
    weights = Llama(
        args.model,
        n_gpu_layers=args.ngl,
        n_ctx=args.context_length,
        verbose=False,
        flash_attn=args.flash_attn
    )

    # ds = load_dataset("json", data_files=args.data_file, split="train").filter(lambda example, idx: idx % 100 == 0, with_indices=True)
    ds = load_dataset("json", data_files=args.data_file, split="train")
    ds = ds.map(lambda item: answer_function(item, weights, args.max_length))
    ds.to_json("output.jsonl", force_ascii=False)

if __name__ == "__main__":
    main()
