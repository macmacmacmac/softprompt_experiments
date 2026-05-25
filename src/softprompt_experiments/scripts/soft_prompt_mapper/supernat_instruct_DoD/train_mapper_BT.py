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
            self.data[idx]["instances"]
        )


# Custom Data Collator for the Mapper Dataset
class MapperCollator:
    def __init__(self, tokenizer, soft_prompt_length=20):
        self.tokenizer = tokenizer
        self.soft_prompt_length = soft_prompt_length

    def __call__(self, batch):
        # Retrieve list of soft_prompts and hard_prompts (in that order)
        soft_prompts, hard_prompts, softprompt_init, instances = zip(*batch)
        
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
            "instances":instances
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
    parser.add_argument("--epochs", type=int, default=5)
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
    # │                 LORA MODEL PREP               │
    # └───────────────────────────────────────────────┘
    print(f"Loading base model {MODEL_NAME}...")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device_map=DEVICE)
    base_model.gradient_checkpointing_enable()

    # Configure LoRA Config to target the key linear layers of attention and feed-forward networks
    lora_config = LoraConfig(
        r = LORA_RANK, 
        lora_alpha = 2 * LORA_RANK,
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout = LORA_DROPOUT,
        bias = "none",
        task_type = TaskType.CAUSAL_LM
    )
    
    # Attach the LoRA adapters to the base model
    model = get_peft_model(base_model, lora_config)
    
    # Print out exactly how many params are trainable
    model.print_trainable_parameters() 
    
    # Get the Llama Model's Word Embedding Mappings
    llama_word_embeddings = model.get_base_model().get_input_embeddings()

    # Init Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr = LR, 
                                  weight_decay = OPTIM_WEIGHT_DECAY)


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

    # Init Training Dataloader
    train_dataloader = DataLoader(
        MapperDataset(train_dataset), 
        batch_size = BATCH_SIZE, 
        shuffle = True, 
        collate_fn = collator
    )
    
    # Init Validation Dataloader
    val_dataloader = DataLoader(
        MapperDataset(val_dataset), 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collator
    )

    with torch.no_grad():
        SOFT_MARKER = llama_word_embeddings(tokenizer("<SOFT:>", add_special_tokens=False, return_tensors='pt').to(DEVICE)['input_ids']).detach()
        HARD_MARKER = llama_word_embeddings(tokenizer("<HARD:>", add_special_tokens=False, return_tensors='pt').to(DEVICE)['input_ids']).detach()
        INIT_MARKER = llama_word_embeddings(tokenizer("<INIT:>", add_special_tokens=False, return_tensors='pt').to(DEVICE)['input_ids']).detach()

    def soft_to_hard(soft_hat=None, **kwargs):
        """
            Computes -log p(hard|soft) from kwargs, supply soft_hat manually to overwrite
            Returns: 
                soft_to_hard_loss,
                rouge_results
        """
        # For back translation when we want to supply a sampled soft prompt instead for loss
        soft_prompts = soft_hat if soft_hat is not None else kwargs.get('soft_prompts')

        attention_mask = kwargs['attention_mask']
        labels = kwargs['labels']
        init = kwargs['init']
        soft_marker = kwargs['soft_marker']
        hard_marker = kwargs['hard_marker']
        init_marker = kwargs['init_marker']
        text_embeds = kwargs['text_embeds']
        batchsize = kwargs['batchsize']

        # build sequence (init + soft + hard)
        inputs_embeds = torch.cat([init_marker, init, soft_marker, soft_prompts, hard_marker, text_embeds], dim=1)               # (batch_size, soft_prompt_len + seq_len, embed_dim)
        prefix_len = init_marker.shape[1] + init.shape[1] + soft_marker.shape[1] + soft_prompts.shape[1] + hard_marker.shape[1]

        # build sequence (soft + hard)
        # inputs_embeds = torch.cat([soft_marker, soft_prompts, hard_marker, text_embeds], dim=1)               # (batch_size, soft_prompt_len + seq_len, embed_dim)
        # prefix_len = soft_marker.shape[1] + soft_prompts.shape[1] + hard_marker.shape[1]

        # Concatenate Attention Masks (Add `1`s for the soft prompt so Llama Model pays attention to it)
        soft_prompt_mask = torch.ones((batchsize, prefix_len), dtype=attention_mask.dtype, device=DEVICE)   # (batch_size, soft_prompt_len)
        full_attention_mask = torch.cat([soft_prompt_mask, attention_mask], dim=1)           # (batch_size, soft_prompt_len + seq_len)
        
        # Concatenate Labels (-100s for the soft prompt so loss isn't calculated on it)
        soft_prompt_labels = torch.full((batchsize, prefix_len), -100, dtype=labels.dtype, device=DEVICE)         # (batch_size, soft_prompt_len)
        full_labels = torch.cat([soft_prompt_labels, labels], dim=1)                         # (batch_size, soft_prompt_len + seq_len)
        
        # Forward Pass
        outputs = model(
            inputs_embeds = inputs_embeds,
            attention_mask = full_attention_mask,
            labels = full_labels
        )

        soft_to_hard_loss = outputs.loss

        #========================= rouge L eval ================================
        # TEACHER FORCING
        # Soft to hard ROUGE-L
        # Extract the logits
        logits = outputs.logits

        # Shift logits and labels so token i predicts i + 1
        shifted_logits = logits[..., :-1, :].contiguous()
        shifted_labels = full_labels[..., 1:].contiguous()

        # Get the predicted token ids
        preds = torch.argmax(shifted_logits, dim = -1)

        # Replace -100 in the labels as we can't decode -100
        shifted_labels = torch.where(
            shifted_labels != -100, 
            shifted_labels, 
            tokenizer.pad_token_id
        )

        # Decode preds and labels into strings
        # skip_special_tokens=True removes EOS and Padding tokens from the text
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(shifted_labels, skip_special_tokens=True)

        # Calculate ROUGE-L for the current batch
        rouge_results = ROUGE_METRIC.compute(
            predictions=decoded_preds, 
            references=decoded_labels, 
            use_stemmer=True
        )
        
        return soft_to_hard_loss, rouge_results

    def hard_to_soft(hard_hat=None, hard_hat_attn_mask=None, **kwargs):

        # For back translation when we want to supply a sampled hard prompt instead for loss
        text_embeds = hard_hat if hard_hat is not None else kwargs.get('text_embeds')
        attention_mask = hard_hat_attn_mask if hard_hat_attn_mask is not None else kwargs.get('attention_mask')

        soft_prompts = kwargs['soft_prompts']
        init = kwargs['init']
        soft_marker = kwargs['soft_marker']
        hard_marker = kwargs['hard_marker']
        init_marker = kwargs['init_marker']
        batchsize = kwargs['batchsize']

        # build sequence (init + hard + soft)
        inputs_embeds = torch.cat([init_marker, init, hard_marker, text_embeds, soft_marker, soft_prompts], dim=1)               # (batch_size, soft_prompt_len + seq_len, embed_dim)
        prefix_len = init_marker.shape[1] + init.shape[1] + hard_marker.shape[1]
        
        # build sequence (hard + soft)
        # inputs_embeds = torch.cat([hard_marker, text_embeds, soft_marker, soft_prompts], dim=1)               # (batch_size, soft_prompt_len + seq_len, embed_dim)
        # prefix_len = hard_marker.shape[1] 

        soft_len = soft_marker.shape[1] + soft_prompts.shape[1]

        # Concatenate Attention Masks (Add `1`s for the soft prompt so Llama Model pays attention to it)
        prefix_mask = torch.ones((batchsize, prefix_len), dtype=attention_mask.dtype, device=DEVICE)  
        softprompt_mask = torch.ones((batchsize, soft_len), dtype=attention_mask.dtype, device=DEVICE)   
        full_attention_mask = torch.cat([prefix_mask, attention_mask, softprompt_mask], dim=1)           
             
        # Forward Pass
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            output_hidden_states=True
        ).hidden_states[-1]  # (B, T, D)

        # predictions (shifted left)
        pred = outputs[:, :-1, :]              # (B, T-1, D)

        # take only positions that predict soft prompt
        pred_soft = pred[:, -soft_prompts.shape[1]:, :]     # (B, k, D)
        hard_to_soft_loss = torch.mean((soft_prompts - pred_soft)**2)
        
        return hard_to_soft_loss

    def hard_to_soft_v2(hard_hat=None, hard_hat_attn_mask=None, give_baseline=False,**kwargs):

        # For back translation when we want to supply a sampled hard prompt instead for loss
        text_embeds = hard_hat if hard_hat is not None else kwargs.get('text_embeds')
        attention_mask = hard_hat_attn_mask if hard_hat_attn_mask is not None else kwargs.get('attention_mask')

        soft_prompts = kwargs['soft_prompts']
        init = kwargs['init']
        soft_marker = kwargs['soft_marker']
        hard_marker = kwargs['hard_marker']
        init_marker = kwargs['init_marker']
        batchsize = kwargs['batchsize']
        batch_instances = kwargs['instances']

        # build sequence (init + hard + soft)
        # inputs_embeds = torch.cat([init_marker, init, hard_marker, text_embeds, soft_marker, soft_prompts], dim=1)               # (batch_size, soft_prompt_len + seq_len, embed_dim)
        # prefix_len = init_marker.shape[1] + init.shape[1] + hard_marker.shape[1]
        
        # build sequence (hard + soft)
        inputs_embeds = torch.cat([hard_marker, text_embeds, soft_marker, soft_prompts], dim=1)               # (batch_size, soft_prompt_len + seq_len, embed_dim)
        prefix_len = hard_marker.shape[1]

        # build sequence (hard + init)
        # inputs_embeds = torch.cat([hard_marker, text_embeds, soft_marker, init], dim=1)               # (batch_size, soft_prompt_len + seq_len, embed_dim)
        # prefix_len = hard_marker.shape[1]


        soft_len = soft_marker.shape[1] + soft_prompts.shape[1]

        # Concatenate Attention Masks (Add `1`s for the soft prompt so Llama Model pays attention to it)
        prefix_mask = torch.ones((batchsize, prefix_len), dtype=attention_mask.dtype, device=DEVICE)  
        softprompt_mask = torch.ones((batchsize, soft_len), dtype=attention_mask.dtype, device=DEVICE)   
        full_attention_mask = torch.cat([prefix_mask, attention_mask, softprompt_mask], dim=1)           
             
        # Forward Pass
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            output_hidden_states=True
        ).hidden_states[-1]  # (B, T, D)

        # predictions (shifted left)
        pred = outputs[:, :-1, :]              # (B, T-1, D)

        # take only positions that predict soft prompt
        pred_softs = pred[:, -soft_prompts.shape[1]:, :]     # (B, k, D)

        flat_instances = [item for sublist in batch_instances for item in sublist[:3]]
        instance_input_ids, instance_attn_mask, instance_labels = prep_instances(flat_instances)

        hard_to_soft_loss = compare_softprompts(
            soft_prompts,
            pred_softs,
            instance_input_ids,
            instance_attn_mask,
            instance_labels
        )

        if give_baseline:
            baseline_loss = compare_softprompts(
                soft_prompts,
                init,
                instance_input_ids,
                instance_attn_mask,
                instance_labels
            )
            return hard_to_soft_loss, baseline_loss

        return hard_to_soft_loss
    
    def generate_hard_from_soft(**kwargs):
        soft_prompts = kwargs['soft_prompts']
        attention_mask = kwargs['attention_mask']
        init = kwargs['init']
        soft_marker = kwargs['soft_marker']
        hard_marker = kwargs['hard_marker']
        init_marker = kwargs['init_marker']
        batchsize = kwargs['batchsize']

        # build input sequence (init + soft)
        inputs_embeds = torch.cat([init_marker, init, soft_marker, soft_prompts, hard_marker], dim=1)               # (batch_size, soft_prompt_len + seq_len, embed_dim)
        prefix_len = init_marker.shape[1] + init.shape[1] + soft_marker.shape[1] + soft_prompts.shape[1] + hard_marker.shape[1]
        
        # Concatenate Attention Masks (Add `1`s for the soft prompt so Llama Model pays attention to it)
        attention_mask = torch.ones((batchsize, prefix_len), dtype=DTYPE, device=DEVICE)   # (batch_size, soft_prompt_len)
        
        # Generate the predicted tokens for the whole batch
        hard_hat_idxs = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
        hard_hat_attn_mask = (hard_hat_idxs != tokenizer.pad_token_id).long()
        hard_hat_embeds = llama_word_embeddings(hard_hat_idxs).detach()  

        return hard_hat_embeds, hard_hat_attn_mask
    
    def generate_soft_from_hard(**kwargs):
        attention_mask = kwargs['attention_mask']
        init = kwargs['init']
        soft_marker = kwargs['soft_marker']
        hard_marker = kwargs['hard_marker']
        init_marker = kwargs['init_marker']
        text_embeds = kwargs['text_embeds']
        batchsize = kwargs['batchsize']

        # build input sequence (init + soft)
        inputs_embeds = torch.cat([init_marker, init, hard_marker, text_embeds, soft_marker], dim=1)               # (batch_size, soft_prompt_len + seq_len, embed_dim)
        prefix_len = init_marker.shape[1] + init.shape[1] + hard_marker.shape[1]
        soft_len = soft_marker.shape[1]
        
        # Concatenate Attention Masks 
        prefix_mask = torch.ones((batchsize, prefix_len), dtype=attention_mask.dtype, device=DEVICE)  
        softprompt_mask = torch.ones((batchsize, soft_len), dtype=attention_mask.dtype, device=DEVICE)   
        full_attention_mask = torch.cat([prefix_mask, attention_mask, softprompt_mask], dim=1)           
        
        # Initial forward pass (full prefix, no soft tokens yet)
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            use_cache=True,
            output_hidden_states=True
        )

        past_key_values = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1:, :]  # (B, 1, D)

        generated = []

        for _ in range(soft_len):
            outputs = model(
                inputs_embeds=last_hidden,   # ONLY last token
                attention_mask=None,         # not needed with cache in most models
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True
            )

            past_key_values = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1:, :]  # (B, 1, D)

            generated.append(last_hidden)
        soft_hat_embeds = inputs_embeds[:, -soft_prompts.shape[1]:, :]     # (B, k, D)

        return soft_hat_embeds
    
    # def generate_soft_from_hard( 
    #     text_embeds, attention_mask, 
    #     init, soft_marker, hard_marker, 
    #     init_marker, batchsize
    # ): 
    #     # build input sequence (init + soft) 
    #     inputs_embeds = torch.cat([init_marker, init, hard_marker, text_embeds, soft_marker], dim=1) #(batch_size, soft_prompt_len + seq_len, embed_dim) 

    #     prefix_len = init_marker.shape[1] + init.shape[1] + hard_marker.shape[1] 

    #     soft_len = soft_marker.shape[1] # Concatenate Attention Masks

    #     prefix_mask = torch.ones((batchsize, prefix_len), dtype=attention_mask.dtype, device=DEVICE)
        
    #     softprompt_mask = torch.ones((batchsize, soft_len), dtype=attention_mask.dtype, device=DEVICE) 

    #     full_attention_mask = torch.cat([prefix_mask, attention_mask, softprompt_mask], dim=1) 
        
    #     # Manually auto regress generate a soft prompt 
    #     for _ in range(soft_len): 
    #         output = model( 
    #             inputs_embeds=inputs_embeds, 
    #             attention_mask=full_attention_mask, 
    #             output_hidden_states=True 
    #         ).hidden_states[-1] # (B, T, D) 

    #         pred = output[:,-1,:].unsqueeze(1) 
    #         inputs_embeds = torch.cat([inputs_embeds, pred], dim=1) 
    #         new_prompt_token_mask = torch.ones((batchsize, 1), dtype=DTYPE, device=DEVICE) 
    #         full_attention_mask = torch.cat([full_attention_mask, new_prompt_token_mask], dim=1) 

    #     soft_hat_embeds = inputs_embeds[:, -soft_prompts.shape[1]:, :] # (B, k, D) 
    #     return soft_hat_embeds

    def prep_instances(instances):
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
            # out_len = len(tokenizer.encode(instance["output"], add_special_tokens=False))
            # inp_out_len = len(tokenizer.encode((f"{instance["input"]}{instance["output"]}"), add_special_tokens=True))

            # print(f"This is inp_len {inp_len}")
            # print(f"This is out_len {out_len}")
            # print(f"This is inp_out_len {inp_out_len}")

            # Mask the input portion so loss is not calculated on it
            labels[i, :inp_len] = -100
            # Mask any padding tokens added to the end of the sequence
            labels[i, attention_mask[i] == 0] = -100

            # print(f"this is labels: {(labels[i])}")
            # print(f"this is labels bool: {(labels[i]!=-100)}")
            # print(f"this is labels sum: {sum(labels[i]!=-100)}")
            # print("")

            if(sum(labels[i]!=-100) == 0):
                print(f"ALERT!!!! LABELS IS FULL -100s")
                print(f"this was original input:<START>{instance["input"]}</END>")
                print(f"this was original input tokenized{tokenizer.encode(instance["input"], add_special_tokens=True)}")
                print(f"this was original output:<START>{instance["output"]}</END>")
                print(f"this was original output tokenized{tokenizer.encode(instance["output"], add_special_tokens=False)}")
                print(f"this was original input+output:<START>{f"{instance["input"]}{instance["output"]}"}</END>")
                print(f"this was original output tokenized{tokenizer.encode(f"{instance["input"]}{instance["output"]}", add_special_tokens=True)}")

        
        return input_ids, attention_mask, labels
        

    def eval_softprompt_on_instances(soft_prompt, instance_input_ids, instance_attn_mask, instance_labels):
        batchsize = instance_labels.shape[0]
        softprompt_len = soft_prompt.shape[1]

        soft_prompt_labels = torch.full((batchsize,softprompt_len), -100, dtype=torch.long, device=DEVICE)  # (batch_size, soft_prompt_len)
        full_labels = torch.cat([soft_prompt_labels, instance_labels], dim=1)                                         # (batch_size, soft_prompt_len + seq_len)

        soft_prompt_attn_mask = torch.ones((batchsize, softprompt_len), dtype=instance_attn_mask.dtype, device=DEVICE)
        full_attn_mask = torch.cat([soft_prompt_attn_mask, instance_attn_mask], dim=1)

        input_embeds = llama_word_embeddings(instance_input_ids)
        full_embeds = torch.cat([soft_prompt.expand(batchsize, -1, -1), input_embeds], dim=1)

        outputs = model(
            inputs_embeds=full_embeds,
            attention_mask=full_attn_mask,
            labels=full_labels
        )

        return outputs.logits, full_labels
    
    def batched_eval_softprompt_on_instances(soft_prompts, instance_input_ids, instance_attn_mask, instance_labels):
        """
            Batched evaluation of softprompts logits
            Say there are S softprompts and K example instances per softprompt
            Expects soft_prompts to be [S, num_soft_tokens, dim]
            Expects instances to be [B*S, T]
        """
        batchsize = instance_labels.shape[0]
        softprompt_len = soft_prompts.shape[1]
        num_soft_prompts = soft_prompts.shape[0]
        num_instances_per_soft = batchsize//num_soft_prompts

        soft_prompt_labels = torch.full((batchsize,softprompt_len), -100, dtype=torch.long, device=DEVICE)  # (batch_size, soft_prompt_len)
        full_labels = torch.cat([soft_prompt_labels, instance_labels], dim=1)                                         # (batch_size, soft_prompt_len + seq_len)

        soft_prompt_attn_mask = torch.ones((batchsize, softprompt_len), dtype=instance_attn_mask.dtype, device=DEVICE)
        full_attn_mask = torch.cat([soft_prompt_attn_mask, instance_attn_mask], dim=1)

        input_embeds = llama_word_embeddings(instance_input_ids)
        expanded_softprompts = soft_prompts.repeat_interleave(num_instances_per_soft, dim=0)
        full_embeds = torch.cat([expanded_softprompts, input_embeds], dim=1)

        with model.disable_adapter():
            outputs = base_model(
                inputs_embeds=full_embeds,
                attention_mask=full_attn_mask,
                labels=full_labels
            )

        return outputs, full_labels

    
    def compare_softprompts(real_softprompt, pred_softprompt, instance_input_ids, instance_attn_mask, instance_labels):
        # This takes a predicted softprompt compares how closely its logits over the target
        # sequence matches the real softprompt's logits over the target sequence
        # This is done via cross entropy, treating the real softprompt's logits as
        # pseudo soft labels

        # [B, T, V] shifted logits over the entire sequence
        real_outputs, full_labels = batched_eval_softprompt_on_instances(real_softprompt, instance_input_ids, instance_attn_mask, instance_labels)
        pred_outputs, full_labels = batched_eval_softprompt_on_instances(pred_softprompt, instance_input_ids, instance_attn_mask, instance_labels)

        real_logits = real_outputs.logits
        pred_logits = pred_outputs.logits

        # Get real probs from
        soft_targets = torch.nn.functional.softmax(real_logits, dim=-1)
        pred_log_probs = torch.nn.functional.log_softmax(pred_logits, dim=-1)

        # Cross entropy with soft labels
        loss = -(soft_targets * pred_log_probs).sum(dim=-1)
        # Mask out ignored label positions
        valid_mask = (full_labels != -100).float()

        if not valid_mask.sum() > 0:
            print(f"valid mask sum: {valid_mask.sum()}")
            print(f"labels: {instance_labels}")
            print(f"decoded: {tokenizer.batch_decode(instance_input_ids)}")


        loss = (loss * valid_mask).sum() / valid_mask.sum()

        return loss


    # ┌───────────────────────────────────────────────┐
    # │                 TRAINING LOOP                 │
    # └───────────────────────────────────────────────┘

    # Phase configs, set me to 0 and 0 for vanilla one way only training for comparison
    NUM_EPOCHS_TO_PRETRAIN = 2
    NUM_EPOCHS_TO_BACKTRANSLATE = 0

    # Loop EPOCHS times
    # ==================================================
    # | The training pipeline is broken up into 3 phases
    # | (1) Pretraining
    # | (2) Backtranslation
    # | (3) Fine-tuning
    # | Details on each are given below in the comments
    # ==================================================
    for epoch in range(EPOCHS):

        # Set the LoRA Model in Training Mode
        model.train()

        total_train_soft_to_hard_loss = 0
        total_train_hard_to_soft_loss = 0
        total_train_rouge_l = 0
        
        soft_to_hard_loss, hard_to_soft_loss, bt_soft_to_hard_loss, bt_hard_to_soft_loss = 0., 0., 0., 0.

        # Init Progress Bar
        dataset_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        # batch_no = 0
        # val_batch_no = 0
        for batch in dataset_pbar:
            # Reset Gradients
            optimizer.zero_grad()
            
            # Move inputs to GPU
            soft_prompts = batch["soft_prompts"].to(DEVICE, dtype=DTYPE)                # (batch_size, soft_prompt_len, embed_dim)
            input_ids = batch["input_ids"].to(DEVICE)                                   # (batch_size, seq_len)
            attention_mask = batch["attention_mask"].to(DEVICE)                         # (batch_size, seq_len)
            labels = batch["labels"].to(DEVICE)                                         # (batch_size, seq_len)
            init = batch["init"].to(DEVICE)
            instances = batch["instances"]
            
            # Get embeddings for the discrete text
            with torch.no_grad():
                text_embeds = llama_word_embeddings(input_ids).detach()                 # (batch_size, seq_len, embed_dim)
            
            batchsize = soft_prompts.shape[0]
            soft_marker = SOFT_MARKER.expand(batchsize, -1, -1)
            hard_marker = HARD_MARKER.expand(batchsize, -1, -1)
            init_marker = INIT_MARKER.expand(batchsize, -1, -1)

            kwargs = {
                'soft_prompts': soft_prompts,
                'attention_mask':attention_mask,
                'labels':labels,
                'init':init,
                'soft_marker':soft_marker,
                'hard_marker':hard_marker,
                'init_marker':init_marker,
                'text_embeds':text_embeds,
                'batchsize':batchsize,
                'instances':instances
            }
            
            # ========================================
            # PHASE 1: Pretraining
            # |--- Pretrain on bi-directional objective
            # |--- loss = -log p(soft|hard) + -log p(hard|soft)
            # |--- Map from soft to hard, and hard to soft
            # |--- We use marker sequences to cue the LLM into knowing which it should do
            # |--- "<INIT:>(init here...)<HARD:>(hard here...)<SOFT:>" cues the LLM into doing hard -> soft
            # |--- "<INIT:>(init here...)<SOFT:>(soft here...)<HARD:>" cues the LLM into doing soft -> hard
            # |--- This is kind of a hyper parameter we can mess around with I guess
            # ========================================
            if epoch < NUM_EPOCHS_TO_PRETRAIN:

                # Computes bi-directional translation loss
                #    -log p(hard | soft)
                #    -log p(soft | hard)
                soft_to_hard_loss, rouge_results = soft_to_hard(**kwargs)
                hard_to_soft_loss = hard_to_soft(**kwargs)
                loss = soft_to_hard_loss + hard_to_soft_loss
                
                # Logging
                current_rouge_l = rouge_results['rougeL']
                total_train_rouge_l += current_rouge_l
                total_train_soft_to_hard_loss += soft_to_hard_loss.item()
                total_train_hard_to_soft_loss += hard_to_soft_loss.item()

                # batch_no+=1
                # if batch_no>5:
                #     break

                batch_log = {
                    "PHASE 1 Loss": f"{loss.item():.2f}",
                    "S->H": f"{soft_to_hard_loss:.2f}",
                    "H->S": f"{hard_to_soft_loss:.2f}",
                }
            # ========================================
            # PHASE 2: Backtranslation
            # |--- After we get a decent-ish bidirectional mapper, we use that to sample new sequences
            # |--- We sample soft_hat and hard_hat
            # |--- Then we just compute -log p(soft|hard_hat.detach()) and -log p(hard|soft.detach()
            # |--- Full loss also inclues the original bi-directional objective just in case
            # |--- Not sure if needed, but probably is
            # |--- This is painfully slow, I have no clue why generate takes so long here
            # ========================================
            # elif epoch < NUM_EPOCHS_TO_PRETRAIN + NUM_EPOCHS_TO_BACKTRANSLATE:

            #     # Disable gradient checkpointing so we can do kv cache generation
            #     # This reduces time from 26s/it to about 18s/it ish
            #     base_model.gradient_checkpointing_disable()
            #     base_model.config.use_cache = True
            #     model.eval()
                
            #     # Samples:
            #     #    soft_hat ~ p(soft|hard)
            #     #    hard_hat ~ p(hard|soft)

            #     with torch.no_grad():
            #         hard_hat, hard_attn_mask = generate_hard_from_soft(**kwargs)
            #         soft_hat = generate_soft_from_hard(**kwargs)

            #     # Re-enably gradient checkpointing before building grad graph
            #     base_model.gradient_checkpointing_enable()
            #     base_model.config.use_cache = False
            #     model.train()

            #     # Computes back translated bi-directional translation loss
            #     #    -log p(hard | soft_hat.detach())
            #     #    -log p(soft | hard_hat.detach())
            #     bt_soft_to_hard_loss, _ = soft_to_hard(
            #         soft_hat = soft_hat.detach(), 
            #         **kwargs
            #     )
            #     bt_hard_to_soft_loss = hard_to_soft(
            #         hard_hat = hard_hat.detach(),
            #         hard_attn_mask = hard_attn_mask.detach(),
            #         **kwargs
            #     )

            #     # Also same as before bi-directional translation loss on real samples for grounding
            #     # might not be necessary idk
            #     soft_to_hard_loss, rouge_results = soft_to_hard(**kwargs)
            #     hard_to_soft_loss = hard_to_soft(**kwargs)

            #     loss = soft_to_hard_loss + hard_to_soft_loss + bt_soft_to_hard_loss + bt_hard_to_soft_loss

            #     # Logging
            #     current_rouge_l = rouge_results['rougeL']
            #     total_train_rouge_l += current_rouge_l
            #     total_train_soft_to_hard_loss += soft_to_hard_loss.item()
            #     total_train_hard_to_soft_loss += hard_to_soft_loss.item()

            #     batch_log = {
            #         "PHASE 2 Loss": f"{loss.item():.2f}",
            #         "S->H": f"{soft_to_hard_loss:.2f}",
            #         "H->S": f"{hard_to_soft_loss:.2f}",
            #         "S->H (BT)": f"{bt_soft_to_hard_loss:.2f}",
            #         "H->S (BT)": f"{bt_hard_to_soft_loss:.2f}",
            #     }
            # ========================================
            # PHASE 2: Finetune
            # |--- Finetunes it on the actual objective
            # |--- Just -log p(hard|soft)
            # ========================================
            else:
                # Just one way map now
                soft_to_hard_loss, rouge_results = soft_to_hard(**kwargs)
                loss = soft_to_hard_loss

                current_rouge_l = rouge_results['rougeL']
                total_train_rouge_l += current_rouge_l
                total_train_soft_to_hard_loss += soft_to_hard_loss.item()

                # Logging
                batch_log = {
                    "PHASE 3 Loss": f"{loss.item():.2f}",
                    "S->H": f"{soft_to_hard_loss:.2f}",
                }

                
            # Backpropagate Loss and Update the Parameters
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            dataset_pbar.set_postfix(batch_log)

            
        avg_train_soft_to_hard_loss = total_train_soft_to_hard_loss / len(train_dataloader)
        avg_train_hard_to_soft_loss = total_train_hard_to_soft_loss / len(train_dataloader)
        avg_train_rouge_l = total_train_rouge_l / len(train_dataloader)
        
        
        # ┌───────────────────────────────────────────────┐
        # │                 VALIDATION LOOP               │
        # └───────────────────────────────────────────────┘
        model.eval()
        total_val_soft_to_hard_loss = 0
        total_val_hard_to_soft_loss = 0
        # total_val_hard_to_soft_baseline = 0
        total_val_rouge_l = 0

        # Freeze all weights
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):

                # Move inputs to DEVICE
                soft_prompts = batch["soft_prompts"].to(DEVICE, dtype=DTYPE)                # (batch_size, soft_prompt_len, embed_dim)
                input_ids = batch["input_ids"].to(DEVICE)                                   # (batch_size, seq_len)
                attention_mask = batch["attention_mask"].to(DEVICE)                         # (batch_size, seq_len)
                labels = batch["labels"].to(DEVICE)                                         # (batch_size, seq_len)
                init = batch["init"].to(DEVICE)
                instances = batch["instances"]
                
                # Get the text embeddings from Llama for the soft prompt token ids
                text_embeds = llama_word_embeddings(input_ids).detach()                 # (batch_size, seq_len, embed_dim)

                # "<SOFT:>" + softprompt + "<HARD>:" + hardprompt
                batchsize = soft_prompts.shape[0]
                soft_marker = SOFT_MARKER.expand(batchsize, -1, -1)
                hard_marker = HARD_MARKER.expand(batchsize, -1, -1)
                init_marker = INIT_MARKER.expand(batchsize, -1, -1)

                # val_batch_no+=1
                # if val_batch_no>5:
                #     break

                # print(init)

                kwargs = {
                    'soft_prompts': soft_prompts,
                    'attention_mask':attention_mask,
                    'labels':labels,
                    'init':init,
                    'soft_marker':soft_marker,
                    'hard_marker':hard_marker,
                    'init_marker':init_marker,
                    'text_embeds':text_embeds,
                    'batchsize':batchsize,
                    'instances':instances
                }

                val_soft_to_hard_loss, rouge_results = soft_to_hard(**kwargs)
                val_hard_to_soft_loss = hard_to_soft(**kwargs)

                # val_hard_to_soft_loss, baseline_loss = hard_to_soft(give_baseline=True,**kwargs)

                                
                # Accumulate validation loss
                total_val_soft_to_hard_loss += val_soft_to_hard_loss.item()
                total_val_hard_to_soft_loss += val_hard_to_soft_loss.item()
                # total_val_hard_to_soft_baseline += baseline_loss.item()

                # Accumulate metrics (initialize `total_train_rouge_l = 0` before the epoch instead of tokens)
                current_rouge_l = rouge_results['rougeL']
                total_val_rouge_l += current_rouge_l
                
        avg_val_soft_to_hard_loss = total_val_soft_to_hard_loss / len(val_dataloader)
        avg_val_hard_to_soft_loss = total_val_hard_to_soft_loss / len(val_dataloader)
        # avg_val_hard_to_soft_loss = total_val_hard_to_soft_baseline / len(val_dataloader)
        avg_val_rouge_l = total_val_rouge_l / len(val_dataloader)

        tqdm.write(f"\nEpoch {epoch + 1} Summary:")
        tqdm.write(
            f"Train losses:\n"
            f"\tsoft to hard loss{avg_train_soft_to_hard_loss: .4f}\n"
            f"\thard to soft loss{avg_train_hard_to_soft_loss: .4f}\n"
            f"\tROUGE-L: {avg_train_rouge_l: .2f}\n"
        )
        tqdm.write(
            f"Val losses:\n"
            f"\tsoft to hard loss{avg_val_soft_to_hard_loss: .4f}\n"
            f"\thard to soft loss{avg_val_hard_to_soft_loss: .4f}\n"
            # f"\thard to soft loss (baseline){total_val_hard_to_soft_baseline: .4f}\n"
            f"\tROUGE-L: {avg_val_rouge_l: .2f}\n"
        )

    # ┌───────────────────────────────────────────────┐
    # │               SAVE LORA ADAPTERS              │
    # └───────────────────────────────────────────────┘
    lora_weights_save_dir = os.path.join(SAVE_DIR, DB_NAME)
    os.makedirs(lora_weights_save_dir, exist_ok=True)
    model.save_pretrained(lora_weights_save_dir)
    tokenizer.save_pretrained(lora_weights_save_dir)
    print(f"Mapper training complete! PEFT LoRA weights saved to {lora_weights_save_dir}")