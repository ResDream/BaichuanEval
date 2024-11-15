import argparse
import torch
import numpy as np
from categories import subcategories, categories
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

choices = ["A", "B", "C", "D"]

# Define available subjects
SUBJECTS = [
    'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge',
    'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics',
    'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics',
    'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic',
    'global_facts', 'high_school_biology', 'high_school_chemistry',
    'high_school_computer_science', 'high_school_european_history', 'high_school_geography',
    'high_school_government_and_politics', 'high_school_macroeconomics',
    'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics',
    'high_school_psychology', 'high_school_statistics', 'high_school_us_history',
    'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law',
    'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing',
    'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition',
    'philosophy', 'prehistory', 'professional_accounting', 'professional_law',
    'professional_medicine', 'professional_psychology', 'public_relations',
    'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions'
]


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(example, include_answer=True):
    """Format a single example from the dataset"""
    prompt = example['question']
    for j, choice in enumerate(choices):
        prompt += f"\n{choice}. {example[f'choices'][j]}"
    prompt += "\nAnswer:"
    if include_answer:
        prompt += f" {example['answer']}\n\n"
    return prompt


def gen_prompt(train_examples, subject, k=-1):
    """Generate prompt from training examples"""
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = len(train_examples)
    for i in range(k):
        prompt += format_example(train_examples[i])
    return prompt


@torch.no_grad()
def eval(args, subject, model, tokenizer, dev_examples, test_examples):
    cors = []
    all_probs = []

    # 添加选项到数字的映射
    letter_to_number = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    number_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

    print(f"\n{'=' * 20} Evaluating {subject} {'=' * 20}")

    for i, example in enumerate(test_examples):
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example(example, include_answer=False)
        train_prompt = gen_prompt(dev_examples, subject, k)
        prompt = train_prompt + prompt_end

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # 确保不超过模型最大长度
        if inputs.input_ids.shape[-1] > model.config.max_position_embeddings:
            while inputs.input_ids.shape[-1] > model.config.max_position_embeddings:
                k -= 1
                train_prompt = gen_prompt(dev_examples, subject, k)
                prompt = train_prompt + prompt_end
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # 生成回答
        outputs = model(**inputs)
        logits = outputs.logits[0, -1]  # 取最后一个token的logits

        # 获取各个选项的概率
        choice_probs = []
        for choice in choices:
            choice_id = tokenizer.encode(" " + choice, add_special_tokens=False)[0]  # 添加空格避免subword问题
            choice_probs.append(logits[choice_id].item())

        probs = torch.nn.functional.softmax(torch.tensor(choice_probs), dim=0).numpy()
        pred_letter = choices[np.argmax(choice_probs)]
        pred_number = letter_to_number[pred_letter]

        label = int(example['answer'])  # 确保标准答案是数字
        label_letter = number_to_letter[label]
        cor = pred_number == label
        cors.append(cor)
        all_probs.append(probs)

        # Print detailed results for this example
        # print(f"\nQuestion {i + 1}:")
        # print(f"Problem: {example['question']}")
        # print("Options:")
        # for j, choice in enumerate(choices):
            # print(f"{choice}. {example['choices'][j]} (Probability: {probs[j]:.3f})")
        # print(f"Model prediction: {pred_letter} ({pred_number})")
        # print(f"Correct answer: {label_letter} ({label})")
        # print(f"Correct: {'✓' if cor else '✗'}")
        # print("-" * 50)

    acc = np.mean(cors)
    cors = np.array(cors)
    all_probs = np.array(all_probs)

    print(f"\nSubject: {subject}")
    print(f"Average accuracy: {acc:.3f}")
    print(f"Total examples: {len(test_examples)}")
    print(f"Correct predictions: {sum(cors)}")
    print("=" * 50)

    return cors, acc, all_probs

def main(args):
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="cuda:0"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    model.eval()

    # Initialize result storage
    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    # Evaluate each subject
    overall_correct = 0
    overall_total = 0

    for subject in SUBJECTS:
        try:
            dataset = load_dataset("cais/mmlu", subject)

            # Convert datasets to lists
            dev_examples = list(dataset['dev'])
            if args.ntrain > len(dev_examples):
                print(f"Warning: requested {args.ntrain} examples but only {len(dev_examples)} available for {subject}")
                dev_examples = dev_examples[:len(dev_examples)]
            else:
                dev_examples = dev_examples[:args.ntrain]

            test_examples = list(dataset['test'])

            if not dev_examples or not test_examples:
                print(f"Skipping {subject} - no examples found")
                continue

            cors, acc, probs = eval(args, subject, model, tokenizer, dev_examples, test_examples)

            # Update overall statistics
            overall_correct += sum(cors)
            overall_total += len(cors)

            # Store results
            if subject in subcategories:
                subcats = subcategories[subject]
                for subcat in subcats:
                    subcat_cors[subcat].append(cors)
                    for key in categories.keys():
                        if subcat in categories[key]:
                            cat_cors[key].append(cors)
            all_cors.append(cors)

        except Exception as e:
            print(f"Error processing subject {subject}: {str(e)}")
            continue

    # Print final summary results
    print("\n" + "=" * 20 + " Final Results " + "=" * 20)
    print(f"Overall Accuracy: {overall_correct / overall_total:.3f}")
    print(f"Total Correct: {overall_correct}")
    print(f"Total Questions: {overall_total}")

    print("\nResults by subcategory:")
    for subcat in subcat_cors:
        if subcat_cors[subcat]:
            subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
            print(f"{subcat}: {subcat_acc:.3f}")

    print("\nResults by category:")
    for cat in cat_cors:
        if cat_cors[cat]:
            cat_acc = np.mean(np.concatenate(cat_cors[cat]))
            print(f"{cat}: {cat_acc:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="baichuan-inc/Baichuan2-7B-Base"
    )
    args = parser.parse_args()
    main(args)