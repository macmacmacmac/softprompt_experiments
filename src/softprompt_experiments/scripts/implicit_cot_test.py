import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

def run(args_list):
    # Load the implicit CoT model
    implicit_cot_model_name = 'yuntian-deng/implicit-cot-math-mistral7b'
    implicit_cot_model = AutoModelForCausalLM.from_pretrained(implicit_cot_model_name, dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(implicit_cot_model_name)
    
    implicit_cot_model.to('cuda' if torch.cuda.is_available() else 'cpu')
    implicit_cot_model.eval()

    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str)
    parser.add_argument("--max_new_tokens", type=int, default=10)
    parser.set_defaults(auto_split=True)

    args, _ = parser.parse_known_args(args_list)

    # Constants
    MAX_RESULT_TOKENS = 10

    def predict_answer(question):
        try:
            input_text = ' '.join(question.split()).strip() + ' ' + tokenizer.eos_token
            print (input_text)
            inputs = tokenizer(input_text, return_tensors='pt').to('cuda' if torch.cuda.is_available() else 'cpu')
            implicit_cot_model.to('cuda' if torch.cuda.is_available() else 'cpu')
        
            input_ids = inputs['input_ids']
            #print (input_ids)
            attention_mask = torch.ones((1, input_ids.shape[1]), dtype=torch.long, device=implicit_cot_model.device)
            print(input_ids.shape)
            print(attention_mask.shape)
            outputs = implicit_cot_model.generate(input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=MAX_RESULT_TOKENS,
                    do_sample=False)
            #print (outputs)
        
            prediction = tokenizer.decode(outputs[0, input_ids.shape[-1]:], skip_special_tokens=True)
        except Exception as e:
            prediction = f'{e}'

        return prediction

    question = args.question if args.question else (
        "Asumi's bookshelf has 120 books. "
        "She has 10 books on history, "
        "twice that many books on literature, and the rest are science books. "
        "How many science books does Asumi have?"
    )
    print(predict_answer(question))