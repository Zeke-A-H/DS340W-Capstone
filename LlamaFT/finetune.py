from pathlib import Path
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from PIL import Image
import torch
from trl import SFTTrainer, SFTConfig
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from huggingface_hub import login
import argparse
import os
import sys
import json
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if __package__:
    from .data import load_scene_samples
else:
    from data import load_scene_samples


def format_data(sample):
    if sample.get('task') == 'placement':
        from SceneAssembler.placement import focus_image
        image = focus_image(sample)
    else:
        image = Image.open(sample["image_path"]).convert("RGB")
    if image is None:
        print(sample["image_path"])
    return {"messages": [

                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": sample.get("input", prompt),
                        },{
                            "type": "image",
                            "image": image,
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": sample["response"]}],
                },
            ],
        }

# Define a custom collator function
def collate_fn(examples):
    """Keep prompt/image context in the inputs; supervise only the CAD answer."""
    for example in examples:
        messages = example["messages"]
        if not messages or messages[-1]["role"] != "assistant":
            raise ValueError("Each training conversation must end with an assistant response")
        if not any(item.get("type") == "text" and item.get("text", "").strip()
                   for item in messages[-1]["content"]):
            raise ValueError("The assistant CAD response must not be empty")

    texts = [processor.apply_chat_template(example["messages"], tokenize=False)
             for example in examples]
    prompt_texts = [processor.apply_chat_template(
        example["messages"][:-1], tokenize=False, add_generation_prompt=True)
        for example in examples]
    image_inputs = [process_vision_info(example["messages"])[0] for example in examples]

    batch = processor(text=texts, images=image_inputs, return_tensors="pt",
                      add_special_tokens=False, padding=True)
    # Prompt masking needs token IDs only; avoid processing the images twice.
    prompt_batch = processor.tokenizer(prompt_texts, return_tensors="pt",
                                       add_special_tokens=False, padding=True)
    labels = torch.full_like(batch["input_ids"], -100)

    for index in range(len(examples)):
        # Attention masks locate real tokens with either left or right padding.
        positions = batch["attention_mask"][index].nonzero(as_tuple=True)[0]
        full_ids = batch["input_ids"][index, positions]
        prompt_ids = prompt_batch["input_ids"][index][
            prompt_batch["attention_mask"][index].bool()]
        prefix_length = prompt_ids.numel()
        if not torch.equal(full_ids[:prefix_length], prompt_ids):
            raise ValueError("Prompt tokens do not match the start of the full conversation")
        answer_positions = positions[prefix_length:]
        labels[index, answer_positions] = batch["input_ids"][index, answer_positions]

    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    labels[batch["input_ids"] == image_token_id] = -100
    if not (labels != -100).any(dim=1).all():
        raise ValueError("Every sample must contain supervised response tokens")
    batch["labels"] = labels
    return batch


def generate_description(sample_image, model, processor):
    messages = [
        {"role": "user", "content": [
            {"type": "text", 
            "text": prompt,
            },
            {"type": "image", 
            "image": sample_image,
            }, 
        ]},
    ]
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        add_special_tokens=False,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=1024, top_p=1.0, temperature=1.0, do_sample=False)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', type=str, default='table')
    parser.add_argument('--task', choices=['structure', 'placement'], default='structure')
    parser.add_argument('--dataset_root', type=Path, default=Path(__file__).resolve().parents[1]/'data/scene_dataset_v2')
    parser.add_argument('--init_adapter', type=Path, help='Initialize trainable LoRA weights; does not resume optimizer state')
    parser.add_argument('--object_count', type=int, choices=[1, 2, 3, 4], default=None,
                        help='Select crops from scenes containing this many objects; omit for all scene sizes')
    parser.add_argument('--dataset', choices=['original', 'scenes'], default='original',
                        help='Use original single-object data or scene-derived crops')
    parser.add_argument('--nepochs', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--precision', choices=['auto', 'fp16', 'bf16'], default='auto')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--smoke_test', action='store_true',
                        help='Run two training steps with a small validation subset and separate checkpoints')
    parser.add_argument('--resume_from_checkpoint', nargs='?', const='latest', default=None,
                        help='Resume a checkpoint path, or latest in this output directory when no path is given')
    parser.add_argument('--save_steps', type=int, default=25,
                        help='Save a recoverable training checkpoint every N optimizer steps')
    args = parser.parse_args()
    if args.task == 'placement' and args.category != 'scene':
        parser.error('Placement training requires --category scene')
    if args.init_adapter and args.resume_from_checkpoint:
        parser.error('Choose init_adapter or resume_from_checkpoint, not both')
    if args.category == 'scene':
        args.dataset = 'scenes'
    if args.object_count is not None and args.category != 'scene':
        parser.error('--object_count for training requires --category scene')
    if args.save_steps < 1:
        parser.error('--save_steps must be positive')
    if args.batch_size < 1:
        parser.error('--batch_size must be positive')

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    category = args.category
    nepochs = args.nepochs
    prompt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prompt.txt')
    prompt = open(prompt_path).read()

    if args.task == 'placement':
        from SceneAssembler.placement import load_samples, prompt as placement_prompt, response as placement_response, fit_statistics
        train_samples = load_samples(args.dataset_root, 'train', args.object_count)
        val_samples = load_samples(args.dataset_root, 'val', args.object_count)
        placement_statistics = fit_statistics(train_samples)
        def placement_examples(samples):
            return [format_data(dict(sample, task='placement', input=placement_prompt(sample),
                                     response=placement_response(sample))) for sample in samples]
        train_dataset = placement_examples(train_samples)
        val_dataset = placement_examples(val_samples)
    elif args.dataset == 'scenes':
        train_samples = load_scene_samples(category, 'train', args.object_count)
        val_samples = load_scene_samples(category, 'val', args.object_count)
        train_dataset = [format_data(sample) for sample in train_samples]
        val_dataset = [format_data(sample) for sample in val_samples]
    else:
        train_ids = open('{}/data/splits/{}_train_ids.txt'.format(project_dir, category)).read().splitlines()
        val_ids = open('{}/data/splits/{}_val_ids.txt'.format(project_dir, category)).read().splitlines()

        valid_ids = [fn.stem for fn in Path('{}/data/llamaft_gt_labels/{}'.format(project_dir, category)).glob('*.txt')]
        train_ids = list(set(train_ids) & set(valid_ids))[:]
        val_ids = list(set(val_ids) & set(valid_ids))[:]


        image_paths = {
            'train': [f'{project_dir}/data/blender_renderings/{fn}.png' for fn in train_ids],
            'val': [f'{project_dir}/data/blender_renderings/{fn}.png' for fn in val_ids],
        }

        # Read prompt once and reuse
        prompt_text = open(prompt_path).read()
        text_inputs = {
            'train': [prompt_text] * len(image_paths['train']),
            'val': [prompt_text] * len(image_paths['val']),
        }

        text_outputs = {
            'train': [open(os.path.join(project_dir, 'data', 'llamaft_gt_labels', category, '{}.txt'.format(fn))).read() for fn in train_ids],
            'val': [open(os.path.join(project_dir, 'data', 'llamaft_gt_labels', category, '{}.txt'.format(fn))).read() for fn in val_ids],
        }

        train_dataset = []
        for i in range(len(image_paths['train'])):
            sample = {
                "image_path": image_paths['train'][i],
                "input": text_inputs['train'][i],
                "response": text_outputs['train'][i],
            }
            train_dataset.append(format_data(sample))

        val_dataset = []
        for i in range(len(image_paths['val'])):
            sample = {
                "image_path": image_paths['val'][i],
                "input": text_inputs['val'][i],
                "response": text_outputs['val'][i],
            }
            val_dataset.append(format_data(sample))

    print(f'Dataset: {args.dataset}; category: {category}; train: {len(train_dataset)}; validation: {len(val_dataset)}')

    if not train_dataset or not val_dataset:
        raise ValueError('Training and validation datasets must both be nonempty')
    if args.smoke_test:
        train_dataset = train_dataset[:4]
        val_dataset = val_dataset[:4]
    checkpoint_root = args.output_dir or ('data/ckpts/llamaft' if category == 'scene' or args.dataset == 'original' else 'data/ckpts/llamaft_scenes')
    if args.task == 'placement' and args.output_dir is None:
        checkpoint_root = 'data/ckpts/llamaft_placement'
    if args.smoke_test:
        checkpoint_root += '_smoke'
    output_dir = '{}/{}'.format(checkpoint_root, category)
    if args.object_count is not None:
        output_dir = os.path.join(output_dir, f'{args.object_count}_objects')
    elif category == 'scene':
        output_dir = os.path.join(output_dir, 'all_objects')
    os.makedirs(output_dir, exist_ok=True)

    task_config = dict(task=args.task, category=category, object_count=args.object_count,
                       dataset_root=str(args.dataset_root.resolve()), supplied_boxes=True,
                       supplied_camera=True, transform='normalized_object_local_to_scene')
    if args.task == 'placement':
        task_config['placement_statistics'] = placement_statistics
    config_path = Path(output_dir)/'task_config.json'
    if args.task == 'placement' and not config_path.is_file() and any(Path(output_dir).glob('*/adapter_config.json')):
        raise ValueError('Refusing to mark existing untagged CAD checkpoints as placement models; choose a placement output root')
    if config_path.is_file() and json.loads(config_path.read_text()).get('task') != args.task:
        raise ValueError('Output folder belongs to a different task; keep placement and structure checkpoints separate')
    if config_path.is_file() and args.resume_from_checkpoint:
        previous = json.loads(config_path.read_text())
        for key in ('task', 'category', 'object_count', 'dataset_root'):
            if previous.get(key) != task_config[key]:
                raise ValueError(f'Resume configuration mismatch: {key}')
        if args.task == 'placement' and previous.get('placement_statistics', {}).get('data_signature') != placement_statistics['data_signature']:
            raise ValueError('Placement training data changed since the checkpoint')
    config_path.write_text(json.dumps(task_config, indent=2)+'\n')

    resume_checkpoint = args.resume_from_checkpoint
    if resume_checkpoint == 'latest':
        from transformers.trainer_utils import get_last_checkpoint
        resume_checkpoint = get_last_checkpoint(output_dir)
        if resume_checkpoint is None:
            raise ValueError(f'No checkpoint found to resume in {output_dir}')
    if resume_checkpoint and not (Path(resume_checkpoint)/'trainer_state.json').is_file():
        raise ValueError(f'Not a resumable Trainer checkpoint: {resume_checkpoint}')

    if not torch.cuda.is_available():
        raise RuntimeError('Run training inside a GPU allocation')
    supports_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    if args.precision == 'bf16' and not supports_bf16:
        raise ValueError('This GPU does not support BF16; use --precision fp16')
    use_bf16 = supports_bf16 if args.precision == 'auto' else args.precision == 'bf16'
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f'GPU: {torch.cuda.get_device_name()}; compute dtype: {compute_dtype}')

    # Note: Set your Hugging Face token as environment variable HF_TOKEN
    # or call login() with your token
    # login('your_hf_token_here')

    # Hugging Face model id
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute_dtype

    )

    # Load model and tokenizer
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        device_map={"": torch.cuda.current_device()},
        torch_dtype=compute_dtype,
        quantization_config=bnb_config
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model.config.use_cache = False

    # LoRA configuration based on QLoRA paper
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM"
    )

    if args.init_adapter:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={'use_reentrant':False})
        model = PeftModel.from_pretrained(model, str(args.init_adapter), is_trainable=True)
        peft_config = None

    # Training arguments
    smoke_test = args.smoke_test
    batch_size = args.batch_size
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=nepochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        max_steps=2 if smoke_test else -1,
        gradient_accumulation_steps=1,  # Increased to maintain effective batch size
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        logging_steps=1 if smoke_test else 5,
        logging_nan_inf_filter=False,
        save_strategy="steps",
        save_steps=1 if smoke_test else args.save_steps,
        save_total_limit=2,
        eval_strategy="steps" if smoke_test else "epoch",
        eval_steps=1 if smoke_test else None,
        learning_rate=2e-4,
        bf16=use_bf16,
        fp16=not use_bf16,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        push_to_hub=False,
        report_to="none" if smoke_test else "tensorboard",
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True}
    )

    # Trainer setup
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        tokenizer=processor.tokenizer,
        peft_config=peft_config
    )

    # Check that a trainable LoRA matrix actually changes during the smoke test.
    probe = None
    if smoke_test:
        probe_name, probe = next((name, parameter) for name, parameter in trainer.model.named_parameters()
                                 if parameter.requires_grad and 'lora_B' in name)
        probe_before = probe.detach().cpu().clone()
    result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    if not torch.isfinite(torch.tensor(result.training_loss)):
        raise RuntimeError('Training produced non-finite loss')
    if smoke_test:
        if torch.equal(probe_before, probe.detach().cpu()):
            raise RuntimeError(f'Smoke test did not update {probe_name}')
        evaluations = [row['eval_loss'] for row in trainer.state.log_history if 'eval_loss' in row]
        if not evaluations or not all(torch.isfinite(torch.tensor(value)) for value in evaluations):
            raise RuntimeError('Smoke test requires finite validation loss')
        print('Smoke test passed: finite training/validation loss and LoRA weights updated')
    final_dir = os.path.join(output_dir, 'final')
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)
    (Path(final_dir)/'task_config.json').write_text(json.dumps(task_config, indent=2)+'\n')
    print(f'Completed {trainer.state.global_step} training steps; saved adapter to {final_dir}')
