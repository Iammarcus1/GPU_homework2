#验证 ModelScope token
from modelscope.hub.api import HubApi
api = HubApi()
api.login('ms-a4cc4194-058d-4cb0-b1d7-6319c988b7aa')

model_name = 'Iammarcus/Qwen3-0.6B-GPU-Pro'
local_model_path = './local-model'

#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download(
    model_id=model_name,
    cache_dir=local_model_path,
    revision="master"
)