# convert-hf.py
import gc
import json
import sys
import os
from writer import parseFloatType, writeTensor, writeHeader, FloatType
from safetensors import safe_open

class ArchType:
    LLAMA = 0xABCD00
    GEMMA3 = 0xABCD03

def permute(tensor, nHeads: int, nKvHeads: int):
    if nHeads != nKvHeads:
        nHeads = nKvHeads
    return (tensor.reshape(nHeads, 2, tensor.shape[0] // nHeads // 2, *tensor.shape[1:]).swapaxes(1, 2).reshape(tensor.shape))

class Processor:
    def __init__(self, config):
        self.config = config
        self.currentModelIndex = None
        self.currentModel = None
        self.currentModelKeys = None
        self.layerMap = {}
        self.plan = []

    def __unloadModel(self):
        if self.currentModel:
            del self.currentModel
            self.currentModel = None
            gc.collect()

    def __loadModel(self, index: int):
        if self.currentModelIndex == index:
            return
        self.__unloadModel()
        filePath = self.config['files'][index]
        print(f'💿 Loading {filePath}...')
        self.currentModel = safe_open(filePath, framework='pt', device='cpu')
        self.currentModelKeys = list(self.currentModel.keys())
        for key in self.currentModelKeys:
            self.layerMap[key] = index
        self.currentModelIndex = index

    def __permuteQ(self, tensor):
        return permute(tensor, self.config['n_heads'], self.config['n_heads'])
        # return tensor

    def __permuteK(self, tensor):
        return permute(tensor, self.config['n_heads'], self.config['n_kv_heads'])
        # return tensor 

    def __preparePlan(self):
        wt = self.config['weights_float_type']
        p = self.plan

        p.append([FloatType.F32, 'language_model.model.embed_tokens.weight'])

        for l in range(0, self.config['n_layers']):
            p.append([wt, f'language_model.model.layers.{l}.self_attn.q_proj.weight'])
            p.append([wt, f'language_model.model.layers.{l}.self_attn.k_proj.weight'])
            p.append([wt, f'language_model.model.layers.{l}.self_attn.v_proj.weight'])
            p.append([wt, f'language_model.model.layers.{l}.self_attn.o_proj.weight'])

            p.append([wt, f'language_model.model.layers.{l}.mlp.gate_proj.weight'])
            p.append([wt, f'language_model.model.layers.{l}.mlp.down_proj.weight'])
            p.append([wt, f'language_model.model.layers.{l}.mlp.up_proj.weight'])

            p.append([FloatType.F32, f'language_model.model.layers.{l}.input_layernorm.weight'])
            p.append([FloatType.F32, f'language_model.model.layers.{l}.post_attention_layernorm.weight'])

        p.append([FloatType.F32, 'language_model.model.norm.weight'])
        p.append([wt, 'lm_head.weight', 'language_model.model.embed_tokens.weight'])


    def ___preparePlan(self):
        wt = self.config['weights_float_type']
        p = self.plan
        p.append([FloatType.F32, 'model.embed_tokens.weight'])
        for l in range(0, self.config['n_layers']):
            p.append([wt, self.__permuteQ, f'model.layers.{l}.self_attn.q_proj.weight'])
            p.append([wt, self.__permuteK, f'model.layers.{l}.self_attn.k_proj.weight'])
            p.append([wt, f'model.layers.{l}.self_attn.v_proj.weight'])
            p.append([wt, f'model.layers.{l}.self_attn.o_proj.weight'])

            p.append([wt, f'model.layers.{l}.mlp.gate_proj.weight'])
            p.append([wt, f'model.layers.{l}.mlp.down_proj.weight'])
            p.append([wt, f'model.layers.{l}.mlp.up_proj.weight'])

            p.append([FloatType.F32, f'model.layers.{l}.input_layernorm.weight'])
            p.append([FloatType.F32, f'model.layers.{l}.post_attention_layernorm.weight'])

        p.append([FloatType.F32, 'model.norm.weight'])
        p.append([wt, 'lm_head.weight', 'model.embed_tokens.weight'])

    def write(self, outputFile: str):
        self.__preparePlan()
        for planItem in self.plan:
            lookup = planItem[1:]
            transform = None
            if callable(lookup[0]):
                transform = lookup[0]
                lookup = lookup[1:]

            tensor = None
            for modelIndex in range(len(self.config['files'])):
                self.__loadModel(modelIndex)
                for layerName in lookup:
                    if layerName in self.currentModelKeys:
                        tensor = self.currentModel.get_tensor(layerName)
                        break
                if tensor is not None:
                    break

            if tensor is None:
                raise Exception(f'Layer {lookup[0]} not found in any files')

            print(f'🔶 Writing {layerName} {tensor.shape}...')

            floatType = planItem[0]
            if transform:
                tensor = transform(tensor)
            
            writeTensor(outputFile, tensor, floatType)

            # Explicitly free memory immediately after writing
            del tensor
            gc.collect()

        # Ensure model is unloaded at the end
        self.__unloadModel()

    def __write(self, outputFile: str):
        self.__preparePlan()
        for planItem in self.plan:
            lookup = planItem[1:]
            transform = None
            if callable(lookup[0]):
                transform = lookup[0]
                lookup = lookup[1:]

            tensor = None
            # Iterate through all files if tensor is not yet found
            for modelIndex in range(len(self.config['files'])):
                self.__loadModel(modelIndex)
                for layerName in lookup:
                    if layerName in self.currentModelKeys:
                        tensor = self.currentModel.get_tensor(layerName)
                        break
                if tensor is not None:
                    break  # Found the tensor, exit loop

            if tensor is None:
                raise Exception(f'Layer {lookup[0]} not found in any files')
            print(f'🔶 Writing {layerName} {tensor.shape}...')

            floatType = planItem[0]
            if transform:
                tensor = transform(tensor)
            writeTensor(outputFile, tensor, floatType)


def parseArchType(type: str):
    archType = {
        'llama': ArchType.LLAMA,
        'mistral': ArchType.LLAMA,
        'gemma3': ArchType.GEMMA3,
    }.get(type)
    if archType is None:
        raise Exception(f'Unsupported arch type: {type}')
    return archType

def parseHiddenAct(act: str):
    hiddenAct = {
        'gelu': 0,
        'silu': 1
    }.get(act)
    if hiddenAct is None:
        raise Exception(f'Unsupported hidden act: {act}')
    return hiddenAct

def parseRopeType(rt: str):
    ropeType = {
        'llama3': 2,
        'linear': 3,
    }.get(rt)
    if ropeType is None:
        raise Exception(f'Unsupported rope type: {rt}')
    return ropeType

def loadConfig(folderPath: str, weightsFloatType: int):
    with open(os.path.join(folderPath, 'config.json')) as fc:
        config = json.load(fc)

    text_cfg = config['text_config']
    files = sorted([os.path.join(folderPath, f) for f in os.listdir(folderPath) if f.endswith('.safetensors')])

    result = {
        'version': 0,
        'arch_type': parseArchType(config['model_type']),
        'hidden_act': 0,  # Gemma uses GELU
        'dim': text_cfg['hidden_size'],
        'hidden_dim': text_cfg['intermediate_size'],
        'n_layers': text_cfg['num_hidden_layers'],
        'n_heads': text_cfg['hidden_size'] // 128,
        'n_kv_heads': (text_cfg['hidden_size'] // 128) // 4,
        'weights_float_type': weightsFloatType,
        'max_seq_len': 4096,
        # 'vocab_size': 256000,
        'vocab_size': config['image_token_index'] + 1,
        'files': files,
        'n_experts': 0,
        'rope_scaling_factor': int(text_cfg['rope_scaling']['factor']),
        'rope_type': parseRopeType(text_cfg['rope_scaling']['rope_type'])
    }

    return result

def printUsage():
    print('Usage: python convert-hf.py <sourceFolderPath> <weightsFloatType> <name>')

if __name__ == '__main__':
    if len(sys.argv) < 4:
        printUsage()
        exit(1)

    sourceFolderPath = sys.argv[1]
    weightsFloatType = parseFloatType(sys.argv[2])
    name = sys.argv[3]
    outputFileName = f'dllama_model_{name}_{sys.argv[2]}.m'

    print(f'Output file: {outputFileName}')
    config = loadConfig(sourceFolderPath, weightsFloatType)

    with open(outputFileName, 'wb') as outputFile:
        writeHeader(outputFile, config)
        processor = Processor(config)
        processor.write(outputFile)

    print(f'✅ {outputFileName} created successfully')

# python converter/convert-hf.py /Users/chenbo/repo/huggingface/gemma-3-4b-it q40 gemma3-4b-it
# python converter/convert-hf.py /Users/chenbo/repo/huggingface/gemma-3-4b-it f32 gemma3-4b-it
