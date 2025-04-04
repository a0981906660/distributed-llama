import sys
import json
import os
from sentencepiece import SentencePieceProcessor
from transformers import PreTrainedTokenizerFast
writer = __import__('tokenizer-writer')

def openJson(path):
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)

def unicodeToBytes():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(cs, bs))

class TokensResolver:
    def __init__(self, dirPath, tokenizerConfig):
        self.dirPath = dirPath
        self.tokenizerConfig = tokenizerConfig
        self.bosId = None
        self.eosIds = None
        self.tokens = []
        self.scores = []

    def resolvePreTrainedTokenizerFast(self):
        utb = unicodeToBytes()
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(self.dirPath, 'tokenizer.json'))
        vocabLen = len(tokenizer.get_vocab())
        for i in range(vocabLen):
            tokenChars = list(tokenizer.convert_ids_to_tokens([i])[0])
            tokenBytes = []
            for chr in tokenChars:
                if chr in utb:
                    tokenBytes.append(utb[chr])
                else:
                    tokenBytes += list(chr.encode('utf-8'))
            self.tokens.append(bytes(tokenBytes))
            self.scores.append(-float(i))

        self.bosId = tokenizer.bos_token_id
        if tokenizer.eos_token_id:
            self.eosIds = [tokenizer.eos_token_id]
        if self.bosId is None or self.eosIds is None:
            config = openJson(os.path.join(self.dirPath, 'config.json'))
            if self.bosId is None:
                self.bosId = config.get('bos_token_id', 2)
            if self.eosIds is None:
                eos = config.get('eos_token_id', 1)
                if isinstance(eos, list):
                    self.eosIds = eos[:2]
                else:
                    self.eosIds = [eos]

    def resolveLlamaTokenizer(self):
        modelPath = os.path.join(self.dirPath, 'tokenizer.model')
        processor = SentencePieceProcessor(model_file=modelPath)

        assert processor.vocab_size() == processor.get_piece_size()
        self.bosId = processor.bos_id()
        self.eosIds = [processor.eos_id()]
        vocabSize = processor.vocab_size()
        for i in range(vocabSize):
            t = processor.id_to_piece(i)
            s = processor.get_score(i)
            t = t.replace('▁', ' ')
            if len(t) == 6 and t.startswith('<0x') and t.endswith('>'):
                b = bytearray.fromhex(t[3:-1])
            else:
                b = t.encode('utf-8')
            self.tokens.append(b)
            self.scores.append(s)

    def resolve(self):
        cls = self.tokenizerConfig['tokenizer_class']
        if cls in ['PreTrainedTokenizerFast', 'LlamaTokenizerFast', 'GemmaTokenizerFast', 'GemmaTokenizer']:
            return self.resolvePreTrainedTokenizerFast()
        if cls == 'LlamaTokenizer':
            return self.resolveLlamaTokenizer()
        raise Exception(f'Tokenizer {cls} is not supported')

def printUsage():
    print('Usage: python convert-tokenizer-hf.py <tokenizerFolderPath> <name>')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        printUsage()
        exit(1)

    dirPath = sys.argv[1]
    name = sys.argv[2]
    tokenizerConfig = openJson(os.path.join(dirPath, 'tokenizer_config.json'))

    resolver = TokensResolver(dirPath, tokenizerConfig)
    resolver.resolve()

    if resolver.bosId is None or resolver.eosIds is None:
        raise Exception('Cannot resolve bosId or eosIds')
    print(f'bosId: {resolver.bosId} ({resolver.tokens[resolver.bosId]})')
    for eosId in resolver.eosIds:
        print(f'eosId: {eosId} ({resolver.tokens[eosId]})')

    chatTemplate = None
    chatExtraStop = None
    if 'chat_template' in tokenizerConfig:
        chatTemplate = tokenizerConfig['chat_template'].encode('utf-8')
        input_value = input('⏩ Enter value for chat extra stop (enter to skip): ')
        if input_value:
            chatExtraStop = input_value.encode('utf-8')

    # Gemma3 actual tokenizer vocab size from HuggingFace tokenizer.json:
    target_vocab_size = 262145
    current_vocab_size = len(resolver.tokens)

    if current_vocab_size != target_vocab_size:
        raise Exception(f"Tokenizer vocab size ({current_vocab_size}) "
                        f"does NOT match model vocab size ({target_vocab_size}). "
                        f"Check your tokenizer files.")

    outputFileName = f'dllama_tokenizer_{name}.t'
    with open(outputFileName, 'wb') as outputFile:
        special_tokens_map = {
            b'<bos>': resolver.bosId,
            b'<eos>': resolver.eosIds[0],
            b'<end_of_turn>': resolver.eosIds[1] if len(resolver.eosIds) > 1 else resolver.eosIds[0],
        }

        special_tokens = {idx: token.decode('utf-8', errors='replace') 
                        for idx, token in enumerate(resolver.tokens) 
                        if token in special_tokens_map}

        writer.writeTokenizer(outputFile, {
            'bos_id': resolver.bosId,
            'eos_id': resolver.eosIds[0],
            'chat_eos_id': resolver.eosIds[1] if len(resolver.eosIds) > 1 else resolver.eosIds[0],
            'special_tokens': special_tokens,
        }, resolver.tokens, resolver.scores, chatTemplate, chatExtraStop)

    print(f'✅ Created {outputFileName}')

# (venv-py39) chenbo@KQCFDQ44W2 distributed-llama % python converter/convert-tokenizer-hf.py /Users/chenbo/repo/huggingface/gemma-3-4b-it gemma3-4b-it

# bosId: 2 (b'<bos>')
# eosId: 1 (b'<eos>')
# eosId: 106 (b'<end_of_turn>')
# ⏩ Enter value for chat extra stop (enter to skip):
# Unknown header key: special_tokens
# ⭐ Params:
# {'bos_id': 2, 'eos_id': 1, 'chat_eos_id': 106, 'special_tokens': {1: '<eos>', 2: '<bos>', 106: '<end_of_turn>'}, 'version': 1, 'vocab_size': 262145, 'max_token_length': 93, 'chat_template': 1532}
# ⭐ Chat template:
# b'{{ bos_token }}\n{%- if messages[0][\'role\'] == \'system\' -%}\n    {%- if messages[0][\'content\'] is string -%}\n        {%- set first_user_prefix = messages[0][\'content\'] + \'\n\n\' -%}\n    {%- else -%}\n        {%- set first_user_prefix = messages[0][\'content\'][0][\'text\'] + \'\n\n\' -%}\n    {%- endif -%}\n    {%- set loop_messages = messages[1:] -%}\n{%- else -%}\n    {%- set first_user_prefix = "" -%}\n    {%- set loop_messages = messages -%}\n{%- endif -%}\n{%- for message in loop_messages -%}\n    {%- if (message[\'role\'] == \'user\') != (loop.index0 % 2 == 0) -%}\n        {{ raise_exception("Conversation roles must alternate user/assistant/user/assistant/...") }}\n    {%- endif -%}\n    {%- if (message[\'role\'] == \'assistant\') -%}\n        {%- set role = "model" -%}\n    {%- else -%}\n        {%- set role = message[\'role\'] -%}\n    {%- endif -%}\n    {{ \'<start_of_turn>\' + role + \'\n\' + (first_user_prefix if loop.first else "") }}\n    {%- if message[\'content\'] is string -%}\n        {{ message[\'content\'] | trim }}\n    {%- elif message[\'content\'] is iterable -%}\n        {%- for item in message[\'content\'] -%}\n            {%- if item[\'type\'] == \'image\' -%}\n                {{ \'<start_of_image>\' }}\n            {%- elif item[\'type\'] == \'text\' -%}\n                {{ item[\'text\'] | trim }}\n            {%- endif -%}\n        {%- endfor -%}\n    {%- else -%}\n        {{ raise_exception("Invalid content type") }}\n    {%- endif -%}\n    {{ \'<end_of_turn>\n\' }}\n{%- endfor -%}\n{%- if add_generation_prompt -%}\n    {{\'<start_of_turn>model\n\'}}\n{%- endif -%}\n'
# ✅ Created dllama_tokenizer_gemma3-4b-it.t

