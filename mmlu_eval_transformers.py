import argparse
import torch
import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path
from categories import subcategories, categories
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# 常量定义保持不变
choices = ["A", "B", "C", "D"]
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
    return " ".join(l)


def format_example(example, include_answer=True):
    """格式化单个示例"""
    prompt = example['question']
    for j, choice in enumerate(choices):
        prompt += f"\n{choice}. {example[f'choices'][j]}"
    prompt += "\nAnswer:"
    if include_answer:
        prompt += f" {example['answer']}\n\n"
    return prompt


def gen_prompt(train_examples, subject, k=-1):
    """生成提示文本"""
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
    predictions = []  # 存储详细预测结果

    letter_to_number = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    number_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

    print(f"\n{'=' * 20} 正在评估 {subject} {'=' * 20}")

    for i, example in enumerate(test_examples):
        k = args.ntrain
        prompt_end = format_example(example, include_answer=False)
        train_prompt = gen_prompt(dev_examples, subject, k)
        prompt = train_prompt + prompt_end

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        if inputs.input_ids.shape[-1] > model.config.max_position_embeddings:
            while inputs.input_ids.shape[-1] > model.config.max_position_embeddings:
                k -= 1
                train_prompt = gen_prompt(dev_examples, subject, k)
                prompt = train_prompt + prompt_end
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        outputs = model(**inputs)
        logits = outputs.logits[0, -1]

        choice_probs = []
        for choice in choices:
            choice_id = tokenizer.encode(" " + choice, add_special_tokens=False)[0]
            choice_probs.append(logits[choice_id].item())

        probs = torch.nn.functional.softmax(torch.tensor(choice_probs), dim=0).numpy()
        pred_letter = choices[np.argmax(choice_probs)]
        pred_number = letter_to_number[pred_letter]

        label = int(example['answer'])
        label_letter = number_to_letter[label]
        cor = pred_number == label

        cors.append(cor)
        all_probs.append(probs)

        # 存储详细预测结果
        predictions.append({
            'question': example['question'],
            'choices': example['choices'],
            'prediction': pred_letter,
            'correct_answer': label_letter,
            'probabilities': probs.tolist(),
            'correct': cor
        })

    acc = np.mean(cors)
    print(f"\n科目: {subject}")
    print(f"平均准确率: {acc:.3f}")
    print(f"总样本数: {len(test_examples)}")
    print(f"正确预测数: {sum(cors)}")
    print("=" * 50)

    return {
        'accuracy': acc,
        'total_examples': len(test_examples),
        'correct_count': sum(cors),
        'detailed_predictions': predictions,
        'raw_cors': cors,
        'probabilities': [p.tolist() for p in all_probs]
    }


def save_results(args, results_data, timestamp):
    """保存评估结果到JSON文件"""
    try:
        # 创建保存目录
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 创建包含模型名称和时间戳的文件名
        model_name = args.model.split('/')[-1]
        filename = f"mmlu_results_{model_name}_{timestamp}.json"
        save_path = save_dir / filename

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存至: {save_path}")

    except Exception as e:
        print(f"\n保存结果时出错: {str(e)}")


def main(args):
    # 记录开始时间和时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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

    # 初始化结果存储
    results_data = {
        "metadata": {
            "model": args.model,
            "num_train_examples": args.ntrain,
            "timestamp": timestamp,
        },
        "overall_results": {
            "total_correct": 0,
            "total_questions": 0,
            "overall_accuracy": 0
        },
        "subject_results": {},
        "subcategory_results": {
            subcat: {"correct": 0, "total": 0, "accuracy": 0}
            for subcat_lists in subcategories.values()
            for subcat in subcat_lists
        },
        "category_results": {
            cat: {"correct": 0, "total": 0, "accuracy": 0}
            for cat in categories
        }
    }

    # 评估每个科目
    for subject in SUBJECTS:
        try:
            dataset = load_dataset("cais/mmlu", subject)

            dev_examples = list(dataset['dev'])
            if args.ntrain > len(dev_examples):
                print(f"警告: 要求{args.ntrain}个示例，但{subject}只有{len(dev_examples)}个可用")
                dev_examples = dev_examples[:len(dev_examples)]
            else:
                dev_examples = dev_examples[:args.ntrain]

            test_examples = list(dataset['test'])

            if not dev_examples or not test_examples:
                print(f"跳过{subject} - 未找到示例")
                continue

            # 获取该科目的评估结果
            subject_results = eval(args, subject, model, tokenizer, dev_examples, test_examples)
            results_data["subject_results"][subject] = subject_results

            # 更新总体统计
            results_data["overall_results"]["total_correct"] += subject_results["correct_count"]
            results_data["overall_results"]["total_questions"] += subject_results["total_examples"]

            # 更新子类别和类别统计
            if subject in subcategories:
                subcats = subcategories[subject]
                for subcat in subcats:
                    results_data["subcategory_results"][subcat]["correct"] += subject_results["correct_count"]
                    results_data["subcategory_results"][subcat]["total"] += subject_results["total_examples"]

                    for key in categories.keys():
                        if subcat in categories[key]:
                            results_data["category_results"][key]["correct"] += subject_results["correct_count"]
                            results_data["category_results"][key]["total"] += subject_results["total_examples"]

        except Exception as e:
            print(f"处理科目 {subject} 时出错: {str(e)}")
            continue

    # 计算最终准确率
    total_correct = results_data["overall_results"]["total_correct"]
    total_questions = results_data["overall_results"]["total_questions"]
    results_data["overall_results"]["overall_accuracy"] = total_correct / total_questions if total_questions > 0 else 0

    # 计算子类别和类别准确率
    for subcat in results_data["subcategory_results"]:
        total = results_data["subcategory_results"][subcat]["total"]
        if total > 0:
            results_data["subcategory_results"][subcat]["accuracy"] = \
                results_data["subcategory_results"][subcat]["correct"] / total

    for cat in results_data["category_results"]:
        total = results_data["category_results"][cat]["total"]
        if total > 0:
            results_data["category_results"][cat]["accuracy"] = \
                results_data["category_results"][cat]["correct"] / total

    # 打印最终结果
    print("\n" + "=" * 20 + " 最终结果 " + "=" * 20)
    print(f"总体准确率: {results_data['overall_results']['overall_accuracy']:.3f}")
    print(f"总正确数: {total_correct}")
    print(f"总题目数: {total_questions}")

    print("\n子类别结果:")
    for subcat, results in results_data["subcategory_results"].items():
        if results["total"] > 0:
            print(f"{subcat}: {results['accuracy']:.3f}")

    print("\n类别结果:")
    for cat, results in results_data["category_results"].items():
        if results["total"] > 0:
            print(f"{cat}: {results['accuracy']:.3f}")

    # 保存结果
    save_results(args, results_data, timestamp)


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

