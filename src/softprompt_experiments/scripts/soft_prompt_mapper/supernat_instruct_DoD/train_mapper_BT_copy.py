import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import evaluate

# PyTorch Dataset wrapper on the Mapper Dataset from Soft Prompts to Hard Prompts
class MapperDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # Return the (20, 4096) tensor and the target string
        return (
            self.data[idx]["soft_prompt"], 
            self.data[idx]["hard_prompt"], 
            self.data[idx]["soft_prompt_init_embeddings"],
            self.data[idx]["instances"],
            self.data[idx]["task_name"]
        )


# Custom Data Collator for the Mapper Dataset
class MapperCollator:
    def __init__(self, tokenizer, soft_prompt_length=20):
        self.tokenizer = tokenizer
        self.soft_prompt_length = soft_prompt_length

    def __call__(self, batch):
        # Retrieve list of soft_prompts and hard_prompts (in that order)
        soft_prompts, hard_prompts, softprompt_init, instances, task_name = zip(*batch)
        
        # Stack the frozen soft prompts into a batch: (batch_size, soft_prompt_len, embed_dim)
        soft_prompts = torch.stack(soft_prompts)        # (batch_size, soft_prompt_len, embed_dim)
        softprompt_init = torch.stack(softprompt_init)        # (batch_size, soft_prompt_len, embed_dim)

        # Explicitly append the EOS token so the model learns when to stop
        hard_prompts = [prompt + self.tokenizer.eos_token for prompt in hard_prompts]
    
        # Tokenize the target hard prompts
        tokenized = self.tokenizer(
            hard_prompts, 
            padding=True, 
            truncation=True, 
            max_length=300, # TODO: Test this value
            return_tensors="pt",
            add_special_tokens=True
        )
        
        input_ids = tokenized["input_ids"]              # (batch_size, seq_len)
        attention_mask = tokenized["attention_mask"]    # (batch_size, seq_len)
        
        # Create labels and mask the padding tokens with -100
        labels = input_ids.clone()                      # (batch_size, seq_len)
        labels[attention_mask == 0] = -100              # (batch_size, seq_len)

        return {
            "soft_prompts": soft_prompts,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "init":softprompt_init,
            "instances":instances,
            "task_name":task_name
        }


# Driver Code
def run(args_list=None):
    exp_name = os.path.basename(__file__)
    print(
        "="*100, "\n", 
        f"\t\t\t\tRunning script: {exp_name}", "\n",
        "="*100,"\n"
    )

    # Perform CLI Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_tokens", type=int, default=20)
    parser.add_argument("--mapper_dataset_path", type=str, default="./datasets/mapper_training_dataset/SUPER-NATURALINSTRUCTIONS-english-filtered_original_instructions")
    parser.add_argument("--save_dir", type=str, default="./mapper_lora_weights")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--optim_weight_decay", type=float, default=0.1) 
    args, _ = parser.parse_known_args(args_list)

    # Parse all the arguments into Variables
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    MAPPER_DATASET_PATH = args.mapper_dataset_path
    DB_NAME = MAPPER_DATASET_PATH.split('/')[-1]
    SAVE_DIR = args.save_dir
    LR = args.lr
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    NUM_TOKENS = args.num_tokens
    LORA_RANK = args.lora_rank
    LORA_DROPOUT = args.lora_dropout
    OPTIM_WEIGHT_DECAY = args.optim_weight_decay

    # Determine DEVICE and DTYPE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token       # Llama doesn't have a default pad token, so we map it to EOS

    # Init Rouge Metric
    ROUGE_METRIC = evaluate.load("rouge")

    # ┌───────────────────────────────────────────────┐
    # │                   DATASET PREP                │
    # └───────────────────────────────────────────────┘
    print("Loading Train and Validation datasets ...")
    train_dataset = torch.load(os.path.join(MAPPER_DATASET_PATH, 'train_mapper_dataset.pt'), map_location="cpu", weights_only=True)
    val_dataset = torch.load(os.path.join(MAPPER_DATASET_PATH, 'val_mapper_dataset.pt'), map_location="cpu", weights_only=True)
    
    print(f"Train Dataset size: {len(train_dataset)} | Validation Dataset size: {len(val_dataset)}")

    # Init Collator
    collator = MapperCollator(
        tokenizer = tokenizer, 
        soft_prompt_length = NUM_TOKENS
    )    
    # Init Validation Dataloader
    val_dataloader = DataLoader(
        MapperDataset(val_dataset), 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collator
    )

    def prep_instances(instances, taskname):
        full_text = [f"Input: {instance["input"]}\nOutput: {instance["output"]}" for instance in instances]
        tokenized = tokenizer(
            full_text, 
            padding=True, 
            truncation=True,
            max_length=512, 
            return_tensors="pt",
            add_special_tokens=True
        ).to(DEVICE)

        input_ids = tokenized["input_ids"]                              # (batch_size, seq_len)
        attention_mask = tokenized["attention_mask"]                    # (batch_size,)

        # Create the labels tensor
        labels = input_ids.clone()                                      # (batch_size, seq_len)
        
        # Mask out the input text and the padding tokens with -100
        for i, instance in enumerate(instances):
            
            # Tokenize just the input text to find out how long it is
            inp_len = len(tokenizer.encode(f"Input: {instance["input"]}\nOutput:", add_special_tokens=True))
            out_len = len(tokenizer.encode(instance["output"], add_special_tokens=False))
            inp_out_len = len(tokenizer.encode((f"Input: {instance["input"]}\nOutput: {instance["output"]}"), add_special_tokens=True))


            # Mask the input portion so loss is not calculated on it
            labels[i, :inp_len] = -100
            # Mask any padding tokens added to the end of the sequence
            labels[i, attention_mask[i] == 0] = -100

            # print(f"this is labels: {(labels[i])}")
            # print(f"this is labels bool: {(labels[i]!=-100)}")
            # print(f"this is labels sum: {sum(labels[i]!=-100)}")
            # print("")

            # if(sum(labels[i]!=-100) == 0):
            print(f"ALERT!!!! LABELS IS FULL -100s")
            print(f"This is inp_len {inp_len}")
            print(f"This is out_len {out_len}")
            print(f"This is inp_out_len {inp_out_len}")
            print(f"input_ids len: {len(input_ids[i])}")
            print(f"labels len: {len(labels[i])}")

            # print(f"task name: {taskname}")
            print(f"this was original input:<START>{instance["input"]}</END>")
            print(f"this was original input tokenized{tokenizer.encode(instance["input"], add_special_tokens=True)}\n")
            print(f"this was original output:<START>{instance["output"]}</END>")
            print(f"this was original output tokenized{tokenizer.encode(instance["output"], add_special_tokens=False)}\n")
            print(f"this was original input+output:<START>{f"{instance["input"]}{instance["output"]}"}</END>")
            print(f"this was original output tokenized{tokenizer.encode(f"{instance["input"]}{instance["output"]}", add_special_tokens=True)}\n")
            break 
        
        return input_ids, attention_mask, labels
        

    with torch.no_grad():

        for batch in (val_dataloader):

            # Move inputs to DEVICE
            batch_instances = batch["instances"]
            taskname = batch["task_name"]
            for i, instances in enumerate(batch_instances):
                if taskname[i] != "task220_rocstories_title_classification":
                    continue
                instance_input_ids, instance_attn_mask, instance_labels = prep_instances(instances, taskname[i])
                

    # ┌───────────────────────────────────────────────┐
    # │               SAVE LORA ADAPTERS              │
    # └───────────────────────────────────────────────┘
    print(f"Mapper training complete! PEFT LoRA weights saved to")