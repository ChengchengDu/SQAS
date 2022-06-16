import importlib
import io


def is_fairscale_available():
    return importlib.util.find_spec("fairscale") is not None


def is_deepspeed_available():
    return importlib.util.find_spec("deepspeed") is not None


def init_deepspeed(trainer, num_training_steps):
    """
    Init DeepSpeed, after converting any relevant Trainer's args into DeepSpeed configuration

    Args:
        trainer: Trainer object
        num_training_steps: per single gpu

    Returns: model, optimizer, lr_scheduler
    """
    import deepspeed

    args = trainer.args
    ds_config_file = args.deepspeed
    model = trainer.model

    with io.open(ds_config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    # The following code translates relevant trainer's cl args into the DS config

    # First to ensure that there is no mismatch between cl args values and presets in the config
    # file, ask to not set in ds config file:
    # - "train_batch_size",
    # - "train_micro_batch_size_per_gpu",
    # - "gradient_accumulation_steps"
    bs_keys = ["train_batch_size", "train_micro_batch_size_per_gpu"]
    if len([x for x in bs_keys if x in config.keys()]):
        raise ValueError(
            f"Do not include {bs_keys} entries in the ds config file, as they will be set via --per_device_train_batch_size or its default"
        )
    if "gradient_accumulation_steps" in config.keys():
        raise ValueError(
            "Do not include gradient_accumulation_steps entries in the ds config file, as they will be set via --gradient_accumulation_steps or its default"
        )

    # DeepSpeed does:
    #   train_batch_size = n_gpus * train_micro_batch_size_per_gpu * gradient_accumulation_steps
    # therefore we just need to set:
    config["train_micro_batch_size_per_gpu"] = args.per_device_train_batch_size
    config["gradient_accumulation_steps"] = args.gradient_accumulation_steps

    if "gradient_clipping" in config:
        logger.info(
            f"Keeping the `gradient_clipping` config from {ds_config_file} intact, ignoring any gradient clipping-specific cl args"
        )
    else:  # override only if the ds config doesn't already have this section
        config["gradient_clipping"] = args.max_grad_norm

    if "optimizer" in config:
        logger.info(
            f"Keeping the `optimizer` config from {ds_config_file} intact, ignoring any optimizer-specific cl args"
        )
    else:  # override only if the ds config doesn't already have this section
        # ds supports Adam, OneBitAdam, and Lamb optimizers and can import other optimizers from torch.
        # But trainer uses AdamW by default.
        # To use other optimizers so using a different scheduler requires voiding warranty with: `zero_allow_untested_optimizer`

        optimizer_configs = {
            "AdamW": {
                "lr": args.learning_rate,
                "betas": [args.adam_beta1, args.adam_beta2],
                "eps": args.adam_epsilon,
                "weight_decay": args.weight_decay,
            }
        }
        optimizer = "AdamW"

        config["zero_allow_untested_optimizer"] = True
        config["optimizer"] = {
            "type": optimizer,
            "params": optimizer_configs[optimizer],
        }

    # DS schedulers (deepspeed/runtime/lr_schedules.py):
    #
    # DS name      | --lr_scheduler_type  | HF func                           | Notes
    # -------------| ---------------------|-----------------------------------|--------------------
    # LRRangeTest  | na                   | na                                | LRRT
    # OneCycle     | na                   | na                                | 1CLR
    # WarmupLR     | constant_with_warmup | get_constant_schedule_with_warmup | w/ warmup_min_lr=0
    # WarmupDecayLR| linear               | get_linear_schedule_with_warmup   |
    if "scheduler" in config:
        logger.info(
            f"Keeping the `scheduler` config from {ds_config_file} intact, ignoring any scheduler-specific cl args"
        )
    else:  # override only if the ds config doesn't already have this section
        if args.lr_scheduler_type == SchedulerType.LINEAR:
            scheduler = "WarmupDecayLR"
            params = {
                "last_batch_iteration": -1,
                "total_num_steps": num_training_steps,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.learning_rate,
                "warmup_num_steps": args.warmup_steps,
            }
        elif args.lr_scheduler_type == SchedulerType.CONSTANT_WITH_WARMUP:
            scheduler = "WarmupLR"
            params = {
                "warmup_min_lr": 0,
                "warmup_max_lr": args.learning_rate,
                "warmup_num_steps": args.warmup_steps,
            }
        else:
            raise ValueError(f"{args.lr_scheduler_type} scheduler type is not supported by DeepSpeed")

        config["scheduler"] = {
            "type": scheduler,
            "params": params,
        }

    # fp16
    if trainer.fp16_backend is not None:
        # Deepspeed has 2 possible fp16 config entries:
        # - `fp16`: for the native amp - it has a bunch of optional params but we won't set any here unless the user did the work
        # - `amp`: which delegates amp work to apex (which needs to be available), but it cannot be used with any ZeRO features, so probably best to be avoided.
        if trainer.fp16_backend == "apex":
            if "amp" in config:
                logger.info(
                    f"Keeping the `amp` config from {ds_config_file} intact, ignoring any amp-specific cl args"
                )
            else:
                config["amp"] = {
                    "enabled": True,
                    "opt_level": args.fp16_opt_level,
                }
        elif trainer.fp16_backend == "amp":
            if "fp16" in config:
                logger.info(
                    f"Keeping the `fp16` config from {ds_config_file} intact, ignoring any fp16-specific cl args"
                )
            else:
                config["fp16"] = {
                    "enabled": True,
                }

    # for clarity extract the specific cl args that are being passed to deepspeed
    ds_args = dict(local_rank=args.local_rank)

    # init that takes part of the config via `args`, and the bulk of it via `config_params`
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        args=SimpleNamespace(**ds_args),  # expects an obj
        model=model,
        model_parameters=model_parameters,
        config_params=config,
    )

    return model, optimizer, lr_scheduler