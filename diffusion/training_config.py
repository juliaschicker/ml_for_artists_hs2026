from dataclasses import dataclass

@dataclass
class TrainingConfig:
    image_size = 64  # the generated image resolution
    train_batch_size = 8
    eval_batch_size = 8  # how many images to sample during evaluation
    num_epochs = 2
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 50
    save_model_epochs = 30
    mixed_precision = "no"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "ddpm-butterflies-64"  # the folder where the model, samples and logs are saved

    # Fine-tuning: continue training an already trained model instead of starting from scratch.
    # None = start from scratch. To fine-tune, set it to the folder of a trained model, e.g. "ddpm-butterflies-64".
    # Use a NEW output_dir when fine-tuning, so the original model is not overwritten.
    # For fine-tuning, a lower learning_rate (e.g. 5e-5) and fewer lr_warmup_steps (e.g. 100) usually work better.
    resume_from = None

    seed = 0


config = TrainingConfig()
