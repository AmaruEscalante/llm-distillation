import argparse
import torch
from transformers import AutoTokenizer, LlamaForCausalLM
from peft import PeftModel
import pandas as pd
from tqdm import tqdm
from utils.nofitications import send_telegram_message


def load_model(base_model, peft_model, device):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = LlamaForCausalLM.from_pretrained(
        base_model, trust_remote_code=True, device_map=device
    )
    model = PeftModel.from_pretrained(model, peft_model)
    model = model.eval()
    return tokenizer, model


def process_chunk(chunk, tokenizer, model, device):
    results = []
    for _, row in chunk.iterrows():
        prompt = f"Instruction:{row['instruction']}\nInput: {row['input']}\nAnswer: "
        tokens = tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)

        with torch.no_grad():
            res = model.generate(**tokens, max_length=512)

        res_sentence = tokenizer.decode(res[0], skip_special_tokens=True)
        out_text = res_sentence.split("Answer: ")[-1].strip()
        results.append(out_text)
    return results


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Send start notification
    send_telegram_message("FinGPT Evaluation process started.")

    print("Loading model...")
    tokenizer, model = load_model(args.base_model, args.peft_model, device)
    model = model.to(device)

    print("Loading dataset...")
    df = pd.read_parquet(args.input_file)

    # Add test mode logic
    if args.test:
        print("Running in test mode. Using only the first 100 rows.")
        df = df.head(100)

    results = []
    chunk_size = args.chunk_size

    print("Processing data...")
    for i in tqdm(range(0, len(df), chunk_size)):
        chunk = df.iloc[i : i + chunk_size]
        chunk_results = process_chunk(chunk, tokenizer, model, device)
        results.extend(chunk_results)

    df["predicted_sentiment"] = results

    print("Saving results...")
    df.to_csv(args.output_file, index=False)
    print(f"Results saved to {args.output_file}")

    # Send finish notification
    send_telegram_message(
        f"FinGPT Evaluation process completed. Results saved to {args.output_file}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinGPT Evaluation Script")
    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
        help="Base model name or path",
    )
    parser.add_argument(
        "--peft_model",
        type=str,
        default="FinGPT/fingpt-mt_llama3-8b_lora",
        help="PEFT model name or path",
    )
    parser.add_argument(
        "--input_file", type=str, required=True, help="Input parquet file path"
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Output CSV file path"
    )
    parser.add_argument(
        "--chunk_size", type=int, default=100, help="Chunk size for processing"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (evaluate only 100 rows)",
    )

    args = parser.parse_args()
    main(args)
