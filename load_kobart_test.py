
from transformers import BartModel
from kobart import get_pytorch_kobart_model, get_kobart_tokenizer


if __name__ == '__main__':
    kobart_tokenizer = get_kobart_tokenizer()
    model = BartModel.from_pretrained(get_pytorch_kobart_model())
    # print(model)
    inputs = kobart_tokenizer(['안녕하세요.'], return_tensors='pt')
    print(inputs)
    out = model(inputs['input_ids'])
    print(out.keys())


