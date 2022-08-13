from transformers import AutoTokenizer


def get_tokenizer(params):
    return AutoTokenizer.from_pretrained(
        params.architecture
    )
