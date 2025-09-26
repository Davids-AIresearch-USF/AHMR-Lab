from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# ---- Global cache ----
_model_cache = {}

def load_model(model_name):
    """Load model+tokenizer once per model name, reuse for later calls."""
    if model_name in _model_cache:
        return _model_cache[model_name]   # return already-loaded copy

    # save_path = f"/nvme/AMHR/onurbilgin/models/{model_name}/"
    save_path = f"{model_name}"
    tokenizer = AutoTokenizer.from_pretrained(save_path)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype="bfloat16"
    )

    model = AutoModelForCausalLM.from_pretrained(
        save_path,
        quantization_config=quant_config,
        device_map="auto",
    )

    # store in cache
    _model_cache[model_name] = (tokenizer, model)
    return tokenizer, model


def get_output(tokenizer, model, msg):

    text = tokenizer.apply_chat_template(
        msg, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    prompt_token_count = inputs.input_ids.shape[1]

    outputs = model.generate(
        **inputs,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        do_sample=True,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
    )

    generated_token_count = outputs.shape[1] - prompt_token_count
    token_count = {"prompt_token": prompt_token_count, "generated_token": generated_token_count}

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return result, token_count



def get_belief_output(tokenizer, model, msg):

    text = tokenizer.apply_chat_template(
        msg, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        do_sample=True,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
    )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return result


