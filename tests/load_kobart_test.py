
from transformers import BartModel
from kobart import get_pytorch_kobart_model, get_kobart_tokenizer


if __name__ == '__main__':
    kobart_tokenizer = get_kobart_tokenizer()
    model = BartModel.from_pretrained(get_pytorch_kobart_model())
    print(model)
    inputs = kobart_tokenizer(['안녕하세요. 저는 이번 프로젝트를 진행하는 이지명입니다.'], return_tensors='pt')
    print(inputs)
    out = model(inputs['input_ids'])
    print(out.keys())
    print(out['last_hidden_state'].shape)
    print(out['past_key_values'][0])


