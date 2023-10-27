import torch
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from routing_model import Routing, load_model

gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2")
max_len =gpt_tokenizer.model_max_length


def load_models(
    base_model_path,
    bbq_adapter_path,
    cnn_adapter_path,
    math_adapter_path,
    routing_model_path,
    device=None
):
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    nf4_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_quant_type="nf4",
       bnb_4bit_use_double_quant=True,
       bnb_4bit_compute_dtype=torch.bfloat16
    )

    with torch.cuda.device(device):
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map={"": device},
            torch_dtype=torch.float16,
            quantization_config=nf4_config
        )

#     base_model = AutoModelForCausalLM.from_pretrained(
#         base_model_path,
#         device_map="auto",
#         torch_dtype=torch.float16,
#         quantization_config=nf4_config
#     )
    routing = load_model(routing_model_path).to(device)
    routing.eval()

    target_map = {
        "others": base_model,
        "bbq": PeftModel.from_pretrained(base_model, bbq_adapter_path, device_map={"": device}),
        "cnn": PeftModel.from_pretrained(base_model, cnn_adapter_path, device_map={"": device}),
        "math": PeftModel.from_pretrained(base_model, math_adapter_path, device_map={"": device}),
        "routing": routing
    }

    return target_map


def get_model(model_map, tokenizer, prompt, params, device="cuda:0"):
    model_lookup = {
        0: "bbq",
        1: "cnn",
        2: "math",
        3: "others"
    }
    temp = gpt_tokenizer(prompt, return_tensors="pt")
    temp = {k: v.to(device) for k, v in temp.items()}

    model_name = model_lookup[model_map["routing"](temp["input_ids"][:max_len]).top_gate.item()]

    return model_map[model_name]
