from entropixing.llama_cpp_impl import generate_response
from llama_cpp import Llama
from datasets import load_dataset

PROMPT_FORMAT = """以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。

### 指示:
{}

### 応答:
"""

CHOSE_FORMAT = """以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。

### 指示:
以下の問題と2つの回答を読み、より良い回答を選択してください。
問題: {}
回答1: {}
回答2: {}

出力は必ず1または2です。

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
        # print(token, end="")
        text += token
    return text.strip()

def chose(weights, question, outputs) ->int:
    input_prompt = CHOSE_FORMAT.format(question, outputs[0], outputs[1])
    try:
        output_number = generate(weights, input_prompt, 1)
        print(output_number)
        output_number = int(output_number) - 1
    except Exception as e:
        return chose(weights, question, outputs)
    if not output_number in [0,1]:
        return chose(weights, question, outputs)
    return output_number

def answer_function(item, weights, max_length):
    input_prompt = PROMPT_FORMAT.format(item["input"])
    outputs = ["", ""]
    outputs[0] = generate(weights, input_prompt, max_length)
    outputs[1] = generate(weights, input_prompt, max_length)
    output_number = chose(weights, item["input"], outputs)
    item["output"] = outputs[output_number]
    return item


def main():
    from argparse import ArgumentParser

    global device
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True, default="./model.gguf")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--context_length", type=int, default=2048)
    parser.add_argument("--ngl", type=int, default=-1)
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

    ds = load_dataset("json", data_files=args.data_file, split="train")
    ds = ds.map(lambda item: answer_function(item, weights, args.max_length))
    ds.to_json("output.jsonl", force_ascii=False)

if __name__ == "__main__":
    main()
