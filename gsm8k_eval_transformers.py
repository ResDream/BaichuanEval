import re
import torch
import argparse
import jsonlines
import numpy as np
import datasets
import os
from datasets import load_from_disk, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def check_prompt_file(prompt_path):
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found at {prompt_path}")
    return open(prompt_path).read()


def doc_to_text(doc, prompt):
    return (
            prompt
            + "\nQuestion: "
            + doc["question"]
            + "\nLet's think step by step\n"
    )


def decode(tokens_list, tokenizer, raw_text_len):
    sents = []
    for tokens in tokens_list:
        tokens = tokens.cpu().numpy().tolist()
        sent = tokenizer.tokenizer.decode(tokens[raw_text_len:])
        sent = sent.split("<|endoftext|>")[0]
        sent = sent.split("\n\n\n")[0]
        sent = sent.split("\n\n")[0]
        sent = sent.split("Question:")[0]
        sents.append(sent)
    return sents


def load_model_and_tokenizer(checkpoint_path):
    print("Loading tokenizer ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path, trust_remote_code=True
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer: {str(e)}")

    print("Loading model ...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            device_map="auto",
            trust_remote_code=True
        ).eval()
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

    return model, tokenizer


def generate_sample(model, tokenizer, input_txt):
    input_ids = tokenizer.encode(input_txt)
    raw_text_len = len(input_ids)
    context_enc = torch.tensor([input_ids]).to(model.device)
    # print(f"Input text: {input_txt}\n")

    try:
        # 直接在generate()中设置参数
        outputs = model.generate(
            context_enc,
            max_length=2048,
            do_sample=False,
        )
        output_text = decode(outputs, tokenizer, raw_text_len)[0]
        # print(f"\nOutput text: {output_text}\n")
        return output_text
    except Exception as e:
        print(f"Generation failed: {str(e)}")
        return ""


def extract_answer(completion):
    # First try to extract answer in the standard format
    match = ANS_RE.search(completion)
    if match:
        try:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            return eval(match_str)
        except:
            pass

    # Fall back to looking for the last number in the text
    try:
        last_number = re.findall(r"\d+", completion)[-1]
        return eval(last_number)
    except:
        return INVALID_ANS


def is_correct(completion, answer):
    try:
        gold = extract_answer(answer)
        if gold == INVALID_ANS:
            print("Warning: No ground truth answer found in the document.")
            return False
        return extract_answer(completion) == gold
    except Exception as e:
        print(f"Error comparing answers: {str(e)}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HF checkpoint.")
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        help="Checkpoint path",
        default="baichuan-inc/Baichuan2-7B-Base",
    )
    parser.add_argument("-f", "--sample-input-file", type=str, default=None)
    parser.add_argument(
        "-o", "--sample-output-file", type=str, default="gsm8k_res.jsonl"
    )
    parser.add_argument(
        "-p", "--prompt-file", type=str, default="gsm8k_prompt.txt"
    )

    args = parser.parse_args()

    # Load and verify prompt file
    try:
        fewshot_prompt = check_prompt_file(args.prompt_file)
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        print("Please ensure the prompt file exists at the specified location")
        exit(1)

    # Load dataset
    try:
        if args.sample_input_file is not None:
            dataset = load_from_disk(args.sample_input_file)
        else:
            config = datasets.DownloadConfig(resume_download=True, max_retries=100)
            dataset = load_dataset("gsm8k", "main", download_config=config)
        test = dataset["test"]
    except Exception as e:
        print(f"Failed to load dataset: {str(e)}")
        exit(1)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.checkpoint_path)

    # Process samples
    f_output = jsonlines.Writer(open(args.sample_output_file, "w", encoding="utf-8"))
    tot_length = test.num_rows
    acc_res = []

    try:
        for i, doc in enumerate(test):
            print(f"\nProcessing sample {i + 1}/{tot_length}")
            context = doc_to_text(doc, fewshot_prompt)
            completion = generate_sample(model, tokenizer, context)
            answer = doc["answer"]
            acc = is_correct(completion, answer)
            doc["completion"] = completion
            doc["acc"] = acc
            f_output.write(doc)
            acc_res.append(acc)

            if (i + 1) % 10 == 0:
                print(f"Current accuracy: {np.mean(acc_res):.4f}")

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
    finally:
        f_output.close()
        if acc_res:
            print(f"\nFinal accuracy: {np.mean(acc_res):.4f}")