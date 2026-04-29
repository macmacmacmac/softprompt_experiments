"""

"""

import torch
import argparse
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from tqdm.auto import tqdm

import numpy as np

from softprompt_experiments.models.softprompt import SoftPrompt
from softprompt_experiments.models.squishyprompt import SquishyPrompt
from softprompt_experiments.models.priors.GMM_prior import GMM_prior
from softprompt_experiments.models.priors.LM_inverter_prior import LM_inverter_prior

from softprompt_experiments.utils import (
    get_train_test_from_tokenized, 
    train_softprompt_from_tokenized,
    eval_softprompt,
    eval_softprompt_regression,
    log_json
)

from peft import PromptTuningInit, PromptTuningConfig, get_peft_model
import logging

# --scripts dataset_nl_custom squishyprompt_generator_regression softprompt_lm_inversion --model_name 'meta-llama/Llama-2-7b-hf' --save_directory ./datasets/logit_prior_inv_1 --verbose --lambd 0.1

def run(args_list):
    exp_name = os.path.basename(__file__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--init", type=str, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--num_tokens", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lambd", type=float, default=0.0)
    parser.add_argument("--no_auto_split",dest="auto_split",action="store_false")
    parser.set_defaults(auto_split=True)
    parser.add_argument("--save_directory", type=str, default="./datasets/math_dataset")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--verbose", action="store_true", help="enable verbose logging")
    parser.set_defaults(verbose=False)
    parser.add_argument("--verbose_level", type=str, default='epoch')
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8b-Instruct")
    
    args, _ = parser.parse_known_args(args_list)
    
    MODEL_NAME = args.model_name
    SAVE_DIR = args.save_directory
    AUTO_SPLIT = args.auto_split
    VERBOSE = args.verbose
    VERBOSE_LEVEL = args.verbose_level
    INIT = args.init
    LR = args.lr
    EPOCHS = args.epochs
    NUM_TOKENS = args.num_tokens
    BATCH_SIZE = args.batch_size
    SEED = args.seed
    LAMBD = args.lambd

    logging.getLogger().setLevel(logging.WARNING)

    logger = logging.getLogger(f"{exp_name}")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            # logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            logging.Formatter("%(message)s")
        )

        # File handler
        file_handler = logging.FileHandler(os.path.join(SAVE_DIR,f"{exp_name}.log"), mode="w")
        file_handler.setFormatter(
            # logging.Formatter("%(levelname)s - %(message)s")
            logging.Formatter("%(message)s")
        )
        file_handler.flush = file_handler.stream.flush

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    logger.propagate = False

    # logging.getLogger("transformers").setLevel(logging.INFO)
    # logging.getLogger("torch").setLevel(logging.INFO)

    logger.info(
        f"{'='*100}\n\t\t\t\tRunning script: {exp_name}\n{'='*100}"
    )

    logger.info("Args: %s", vars(args))    

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=dtype
    ).to(device)
    model.eval()
    word_embeddings = model.get_input_embeddings()

    value_head = torch.nn.Sequential(
        torch.nn.Linear(word_embeddings.embedding_dim, word_embeddings.embedding_dim),
        torch.nn.Tanh(),
        torch.nn.Linear(word_embeddings.embedding_dim, 1)
    )
    value_head = value_head.to(dtype=dtype)
    value_head.to(device)

    # Get dataset sub directories
    dataset_dirs = []
    for entry in os.scandir(SAVE_DIR):
        if entry.is_dir():  # Check if the entry is a directory
            if "dataset_" in entry.name:
                dataset_dirs.append(entry.path)

    num_datasets = len(dataset_dirs)
    dataset_dirs = [
        os.path.join(SAVE_DIR, f"dataset_{i}")
        for i in range(num_datasets)
    ]
    if num_datasets > 0:
        logger.info(f"\nFound ({num_datasets}) datasets in directory")
    else:
        raise ValueError("path to directory has no datasets")

    for dataset_dir in tqdm(dataset_dirs):
        # load dataset
        _, test_dataset, train_loader, test_loader = get_train_test_from_tokenized(
            dataset_dir,
            BATCH_SIZE,
            train_portion = 0.8,
            auto_split=AUTO_SPLIT
        )
        del _

        # initialize softprompt
        if SEED is not None:
            vocab_size = word_embeddings.num_embeddings
            rng = np.random.default_rng(seed=SEED)
            init_token_ids = torch.from_numpy(
                rng.integers(0, vocab_size, size=NUM_TOKENS, dtype=np.int64)
            ).to(model.device)
            init = tokenizer.decode(init_token_ids)
        else:
            init = INIT
        
        # logger.info("Initial tokens: ", init)
        # logits_prior = GMM_prior()
        softprompt = SoftPrompt(
            model=model, 
            init=init,
            tokenizer=tokenizer, 
            word_embeddings=word_embeddings, 
            num_tokens=NUM_TOKENS,
        )
        value_head = torch.nn.Sequential(
            torch.nn.Linear(word_embeddings.embedding_dim, word_embeddings.embedding_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(word_embeddings.embedding_dim, 1)
        )
        value_head = value_head.to(dtype=dtype)
        value_head.to(device)
        if AUTO_SPLIT:
            hardprompt = torch.load(
                os.path.join(dataset_dir,'dataset.pt'),
                weights_only=False
            )['hardprompt']
        else:
            hardprompt = torch.load(
                os.path.join(dataset_dir,'train_dataset.pt'),
                weights_only=False
            )['hardprompt']

        
        # begin training
        if VERBOSE:
            logger.info(hardprompt)
        # train_loss, test_loss, entropy = train_softprompt_from_tokenized(
        #     squishyprompt, LR, EPOCHS, train_loader, test_loader, 
        #     verbose=VERBOSE, verbose_level=VERBOSE_LEVEL,
        #     entropy_reg_constant=LAMBD, logger=logger
        # )
        model = softprompt._model
        tokenizer = softprompt._tokenizer
        word_embeddings = softprompt._word_embeddings
        dtype = model.dtype
        device = model.device

        # Freeze LM
        model.requires_grad_(False)
        softprompt.to(device)

        # Only train the softprompt parameters
        optimizer = torch.optim.AdamW(softprompt.parameters(), lr=LR)
        v_optimizer = torch.optim.AdamW(value_head.parameters(), lr=LR)

        def prep_inputs_for_z_loss(input_ids, labels):
            batchsize = input_ids.size(0)
            # softprompt embeddings
            sp_embeds = softprompt.forward()   # [1, soft_len, dim]
            sp_embeds = sp_embeds.expand(batchsize, -1, -1)
            input_embeds = word_embeddings(input_ids).to(dtype=dtype)  #
            full_embeds = torch.cat([sp_embeds, input_embeds], dim=1)

            # Shift labels to align with concatenated softprompt
            pad_prefix = torch.full(
                (labels.shape[0], sp_embeds.shape[1]),
                -100,
                dtype=labels.dtype,
                device=device
            )
            labels_adjusted = torch.cat([pad_prefix, labels], dim=1)

            # build and shift attention mask
            attention_mask = (input_ids != tokenizer.pad_token_id).long()
            attn_prefix = torch.ones(
                (labels.shape[0], sp_embeds.shape[1]),
                dtype=labels.dtype,
                device=device
            )
            attention_mask = torch.cat([attn_prefix, attention_mask], dim=1)
            return full_embeds, labels_adjusted, attention_mask
        
        def prep_input_for_zprime_sample(input_ids_x_y, labels_x_y, input_id_p):
            """
                input_ids_x_y: a [B,T] tensor of tokenized input ids               
            """
            list_input_ids_x_p = []
            for i in range(len(input_ids_x_y)):
                mask = (labels_x_y[i]==-100).to(device) & (input_ids_x_y[i] != tokenizer.pad_token_id)
                input_id_x = input_ids_x_y[i][mask]
                input_id_x_p = torch.cat([input_id_x, input_id_p], dim=0)
                list_input_ids_x_p.append(input_id_x_p)

            tokenized_x_p = tokenizer.pad(
                {"input_ids": list_input_ids_x_p},
                padding=True,          # pad to longest in batch
                return_tensors="pt"
            ).to(device)

            input_ids_x_p, attn_mask_x_p = tokenized_x_p['input_ids'], tokenized_x_p['attention_mask']
            sp_embeds = softprompt.forward()   # [1, soft_len, dim]
            sp_embeds = sp_embeds.expand(input_ids_x_p.shape[0], -1, -1)
            embeds_x_p = word_embeddings(input_ids_x_p).to(dtype=dtype)
            embeds_z_x_p = torch.cat([sp_embeds, embeds_x_p], dim=1)

            # build and shift attention mask
            attn_prefix = torch.ones(
                (input_ids_x_p.shape[0], sp_embeds.shape[1]),
                dtype=dtype,
                device=device
            )
            attn_mask_z_x_p = torch.cat([attn_prefix, attn_mask_x_p], dim=1)

            return embeds_z_x_p, attn_mask_z_x_p

        
        def prep_inputs_for_zprime_loss(input_ids_x_y, labels_x_y, input_ids_zprime):
            """
                Builds embedding matrix that concats [x;z;y] along seq dim
                input_ids_x_y: a [B,T] tensor of input ids [x;y] 
                input_ids_zprime a [B,T] tensor of input ids        
                labels_x_y: [B,T] tensor of labels

            """
            list_input_ids_x_zprime_y = []
            prefix_lengths = []
            for i in range(len(input_ids_x_y)):
                # get y first
                mask_y = (labels_x_y[i]!=-100).to(device)
                mask_x = (~mask_y) & (input_ids_x_y[i] != tokenizer.pad_token_id)
                input_id_y = input_ids_x_y[i][mask_y]
                input_id_x = input_ids_x_y[i][mask_x]

                # get [x;p;z'], remove padding tokens from the end first before joining y
                mask_zprime = (input_ids_zprime[i] != tokenizer.pad_token_id).to(device)
                input_id_zprime_no_pad = input_ids_zprime[i][mask_zprime].to(device)
                prefix_lengths.append(len(input_id_zprime_no_pad))

                # concatenate to get [x;p;z';y]
                input_id_x_zprime_y = torch.cat([input_id_x, input_id_zprime_no_pad, input_id_y], dim=0)
                list_input_ids_x_zprime_y.append(input_id_x_zprime_y)

            input_ids_x_zprime_y = tokenizer.pad(
                {"input_ids": list_input_ids_x_zprime_y},
                padding=True,          # pad to longest in batch
                return_tensors="pt"
            ).to(device)

            # build labels
            labels_x_zprime_y = input_ids_x_zprime_y['input_ids'].clone()
            for i in range(len(labels_x_zprime_y)):
                prefix_length = prefix_lengths[i]
                labels_x_zprime_y[i][:prefix_length] = -100

                mask_pad = (labels_x_zprime_y[i]==tokenizer.pad_token_id).long().to(device)
                labels_x_zprime_y[i][mask_pad] = -100
            
            embeds_x_zprime_y = word_embeddings(input_ids_x_zprime_y['input_ids']).to(dtype=dtype)

            return embeds_x_zprime_y, input_ids_x_zprime_y['attention_mask'], labels_x_zprime_y

        def prep_inputs_for_zprime_loss_STE(input_ids_x_y, labels_x_y, input_ids_zprime):
            """
                input_ids_x_y: a [B,T] tensor of tokenized input ids              
            """
            list_input_ids_x_zprime_y = []
            prefix_lengths = []
            for i in range(len(input_ids_x_y)):
                # get y first
                mask_y = (labels_x_y[i]!=-100).to(device)
                mask_x = (~mask_y) & (input_ids_x_y[i] != tokenizer.pad_token_id)
                input_id_y = input_ids_x_y[i][mask_y]
                input_id_x = input_ids_x_y[i][mask_x]

                # get [x;p;z'], remove padding tokens from the end first before joining y
                mask_zprime = (input_ids_zprime[i] != tokenizer.pad_token_id).to(device)
                input_id_zprime_no_pad = input_ids_zprime[i][mask_zprime].to(device)
                prefix_lengths.append(len(input_id_zprime_no_pad))

                # concatenate to get [x;p;z';y]
                input_id_x_zprime_y = torch.cat([input_id_x, input_id_zprime_no_pad, input_id_y], dim=0)
                list_input_ids_x_zprime_y.append(input_id_x_zprime_y)

            input_ids_x_zprime_y = tokenizer.pad(
                {"input_ids": list_input_ids_x_zprime_y},
                padding=True,          # pad to longest in batch
                return_tensors="pt"
            ).to(device)

            # build labels
            labels_x_zprime_y = input_ids_x_zprime_y['input_ids'].clone()
            for i in range(len(labels_x_zprime_y)):
                prefix_length = prefix_lengths[i]
                labels_x_zprime_y[i][:prefix_length] = -100

                mask_pad = (labels_x_zprime_y[i]==tokenizer.pad_token_id).long().to(device)
                labels_x_zprime_y[i][mask_pad] = -100
            
            embeds_x_zprime_y = word_embeddings(input_ids_x_zprime_y['input_ids']).to(dtype=dtype)

            return embeds_x_zprime_y, input_ids_x_zprime_y['attention_mask'], labels_x_zprime_y

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        def manual_loss(logits, labels):
            B, T = labels.shape
            shift_logits = logits[:, :-1]
            shift_labels = labels[:, 1:]
            token_loss = loss_fct(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1)
            ).view(B, T-1)
            loss_per_example = token_loss.sum(dim=1) / (shift_labels != -100).sum(dim=1)
            return loss_per_example
        
        def value_pred(outputs_z_x_y, attn_mask_z_x_y):
            lengths = attn_mask_z_x_y.sum(dim=1) - 1  # [B]
            batch_idx = torch.arange(outputs_z_x_y.size(0), device=device)
            last_hidden = outputs_z_x_y[batch_idx, lengths]
            # pooled_hidden = (outputs_z_x_y * attn_mask_z_x_y.unsqueeze(-1)).sum(1) / attn_mask_z_x_y.sum(1)
            value_pred = value_head(last_hidden).squeeze(-1)
            return value_pred

        def update_critic(outputs_z_x_y, attn_mask_z_x_y, input_ids_x_y, labels_x_y):
            #======================
            # critic update
            #======================
            # Sample z'
            # we use this suffix prompt to get the LLM to hallucinate
            # CoT context based on the input and softprompt z
            suffix_prompt = "First, I should"
            input_id_p = tokenizer(
                suffix_prompt, 
                add_special_tokens=False, 
                return_tensors='pt'
            )['input_ids'][0].to(device)
            embeds_z_x_p, attn_mask_z_x_p = prep_input_for_zprime_sample(input_ids_x_y, labels_x_y, input_id_p)
            input_ids_zprime = model.generate(
                inputs_embeds=embeds_z_x_p,
                attention_mask=attn_mask_z_x_p,
                max_new_tokens=64,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

            sample_zprime = tokenizer.decode(
                input_ids_zprime[0],
                skip_special_tokens=True
            )

            embeds_x_zprime_y, attn_mask_x_zprime_y, labels_x_zprime_y = prep_inputs_for_zprime_loss(
                input_ids_x_y,
                labels_x_y,
                input_ids_zprime
            )

            # -log p(y|x,z')
            outputs_x_p_zprime_y = model( 
                inputs_embeds=embeds_x_zprime_y,
                attention_mask=attn_mask_x_zprime_y,
            )
            loss_under_zprime = manual_loss(outputs_x_p_zprime_y.logits, labels_x_zprime_y)
            # v(z,x,y)
            # re-use outputs for value prediction
            predicted_loss_under_zprime = value_pred(outputs_z_x_y.hidden_states[-1].detach(), attn_mask_z_x_y)

            critic_loss = ((predicted_loss_under_zprime - loss_under_zprime)**2).mean(dim=-1)
            critic_loss.backward()
            v_optimizer.step()
            v_optimizer.zero_grad()
            optimizer.zero_grad()

            logger.info(f"Critic loss: {critic_loss}")
            
            return sample_zprime  

        def zprime_loss(outputs_z_x_y, attn_mask_z_x_y, input_ids_x_y, labels_x_y):
            suffix_prompt = "First, I should"
            input_id_p = tokenizer(
                suffix_prompt, 
                add_special_tokens=False, 
                return_tensors='pt'
            )['input_ids'][0].to(device)
            embeds_z_x_p, attn_mask_z_x_p = prep_input_for_zprime_sample(input_ids_x_y, labels_x_y, input_id_p)
            input_ids_zprime = model.generate(
                inputs_embeds=embeds_z_x_p,
                attention_mask=attn_mask_z_x_p,
                max_new_tokens=64,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

            sample_zprime = tokenizer.decode(
                input_ids_zprime[0],
                skip_special_tokens=True
            )

            embeds_x_zprime_y, attn_mask_x_zprime_y, labels_x_zprime_y = prep_inputs_for_zprime_loss(
                input_ids_x_y,
                labels_x_y,
                input_ids_zprime
            )

            # -log p(y|x,z')
            outputs_x_p_zprime_y = model( 
                inputs_embeds=embeds_x_zprime_y,
                attention_mask=attn_mask_x_zprime_y,
            )
            loss_under_zprime = manual_loss(outputs_x_p_zprime_y.logits, labels_x_zprime_y)
            # v(z,x,y)
            # re-use outputs for value prediction
            predicted_loss_under_zprime = value_pred(outputs_z_x_y.hidden_states[-1].detach(), attn_mask_z_x_y)

            critic_loss = ((predicted_loss_under_zprime - loss_under_zprime)**2).mean(dim=-1)
            critic_loss.backward()
            v_optimizer.step()
            v_optimizer.zero_grad()
            optimizer.zero_grad()

            logger.info(f"Critic loss: {critic_loss}")
            
            return sample_zprime  


        # Actual train loop
        train_loss = 0.0
        test_loss = 0.0
        for epoch in range(EPOCHS):
            softprompt.train()
            for i, batch in enumerate(train_loader):
                input_ids_x_y, labels_x_y = [b.to(device) for b in batch]

                embeds_z_x_y, labels_z_x_y, attn_mask_z_x_y = prep_inputs_for_z_loss(input_ids_x_y, labels_x_y)

                # get outputs under softprompt
                outputs_z_x_y = model(
                    inputs_embeds=embeds_z_x_y,
                    attention_mask=attn_mask_z_x_y,
                    labels=labels_z_x_y,
                    output_hidden_states=True
                )
                if epoch > 6:
                    # ===============
                    # Critic update
                    # - update value function only
                    # - gradient shouldnt softprompt update here
                    # ===============
                    NUM_CRITIC_UPDATES = 6
                    for _ in range(NUM_CRITIC_UPDATES):
                        sample_zprime = update_critic(outputs_z_x_y, attn_mask_z_x_y, input_ids_x_y, labels_x_y)
                else:
                    sample_zprime = ""
                # ==============
                # Softprompt update
                # - update soft prompt only
                # - should not update value head at all here
                # ==============

                # -log p(y|x,z)
                loss_under_z = manual_loss(outputs_z_x_y.logits, labels_z_x_y).mean()

                if epoch > 6:
                    # loss_under_z = outputs_z_x_y.loss
                    predicted_loss_under_zprime = value_pred(outputs_z_x_y.hidden_states[-1], attn_mask_z_x_y).mean()
                else:
                    predicted_loss_under_zprime = torch.tensor(0.0
                                                               )
                annealing = 0.1 if loss_under_z > 4.0 else 1.0
                # annealing = 0.0

                loss = loss_under_z + annealing*predicted_loss_under_zprime

                logger.info(
                    f"TRAIN---Epoch: {epoch}, Batch: {i}\n"
                    f"\tloss under z: {loss_under_z}\n" 
                    f"\tloss under z': {predicted_loss_under_zprime}\n"
                    f"\tsample z': {sample_zprime}\n"
                )
                train_loss = loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                v_optimizer.zero_grad()

            softprompt.eval()
            for i, batch in enumerate(test_loader):
                with torch.no_grad():
                    input_ids_x_y, labels_x_y = [b.to(device) for b in batch]

                    embeds_z_x_y, labels_z_x_y, attn_mask_z_x_y = prep_inputs_for_z_loss(input_ids_x_y, labels_x_y)

                    # get outputs under softprompt
                    outputs_z_x_y = model(
                        inputs_embeds=embeds_z_x_y,
                        attention_mask=attn_mask_z_x_y,
                        labels=labels_z_x_y,
                        output_hidden_states=True
                    )
                    if epoch > 3:
                        # ===============
                        # Critic update
                        # - update value function only
                        # - gradient shouldnt softprompt update here
                        # ===============
                        NUM_CRITIC_UPDATES = 6
                        for _ in range(NUM_CRITIC_UPDATES):
                            sample_zprime = update_critic(outputs_z_x_y, attn_mask_z_x_y, input_ids_x_y, labels_x_y)
                    else:
                        sample_zprime = ""

                    # ==============
                    # Softprompt update
                    # - update soft prompt only
                    # - should not update value head at all here
                    # ==============

                    # -log p(y|x,z)
                    loss_under_z = manual_loss(outputs_z_x_y.logits, labels_z_x_y).mean()

                    if epoch > 3:
                        # loss_under_z = outputs_z_x_y.loss
                        predicted_loss_under_zprime = value_pred(outputs_z_x_y.hidden_states[-1], attn_mask_z_x_y).mean()
                    else:
                        predicted_loss_under_zprime = torch.tensor(0.0
                                                                )
                    annealing = 0.1 if loss_under_z > 4.0 else 1.0
                    # annealing = 0.0

                    loss = loss_under_z + annealing*predicted_loss_under_zprime

                    logger.info(
                        f"TEST---Epoch: {epoch}, Batch: {i}\n"
                        f"\tloss under z: {loss_under_z}\n" 
                        f"\tloss under z': {predicted_loss_under_zprime}\n"
                        f"\tsample z': {sample_zprime}\n"
                    )
                    test_loss = loss

        # # if verbose: generate sample output predictions using eval_softprompt
        # if VERBOSE:
        #     outputs = eval_softprompt_regression(softprompt, test_dataset, dataset_dir)
        #     logger.info(outputs)
        #     performance = {
        #         'hardprompt':hardprompt,
        #         'train loss':train_loss,
        #         'test_loss':test_loss,
        #         'outputs': outputs
        #     }
        #     log_json(os.path.join(dataset_dir,'softprompt_performance.json'), performance)
        # else:
        #     performance = {
        #         'hardprompt':hardprompt,
        #         'train loss':train_loss,
        #         'test_loss':test_loss,
        #     }
        #     log_json(os.path.join(dataset_dir,'softprompt_performance.json'), performance)

        softprompt.save_softprompt(dataset_dir)

    logger.info(
        f"{'='*100}\n\t\t\t\tCompleted script: {exp_name}\n{'='*100}"
    )









